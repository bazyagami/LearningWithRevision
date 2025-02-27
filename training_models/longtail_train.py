import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import time
from utils import log_memory, plot_metrics, plot_metrics_test, plot_accuracy_time_multi, plot_accuracy_time_multi_test
from tqdm import tqdm
from imbalance_cifar import IMBALANCECIFAR100

# cls_num_list = IMBALANCECIFAR100.get_cls_num_list()

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)
    

def train_with_revision_longtail(model_name, model, train_loader, test_loader, device, epochs, save_path, threshold, start_revision, task, cls_num_list):

    save_path = save_path
    model.to(device)
    
    # criterion = nn.CrossEntropyLoss()
    train_sampler = None
    idx = epochs // 24
    betas = [0, 0.9999]
    effective_num = 1.0 - np.power(betas[idx], cls_num_list)
    per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    #as per implementation LR=0.045, they use 16 GPU. https://discuss.pytorch.org/t/training-mobilenet-on-imagenet/174391/6 from this blog
    #we use the idea to divide the learning rate by the number of GPUs. 
    # optimizer = optim.RMSprop(self.model.parameters(), weight_decay=0.00004, momentum=0.9, lr=0.0028125)   
    # optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
    epoch_losses = []
    epoch_accuracies = []
    epoch_test_accuracies = []
    epoch_test_losses = []
    time_per_epoch = []
    start_time = time.time()
    for epoch in range(epochs):
        if epoch < start_revision : 
            model.train()
            epoch_start_time = time.time()
            running_loss = 0.0
            correct = 0
            total_correct = 0
            total_samples = 0
            total = 0
            print(f"Epoch [{epoch+1/epochs}]")
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
            
            for batch_idx, (inputs, labels) in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                with torch.no_grad():
                    outputs = model(inputs)
                    # if task == "segmentation":
                    #     outputs = outputs['out']
                    preds = torch.argmax(outputs, dim=1)
                    
                    if threshold == 0:
                        mask = preds != labels
                    else:
                        prob = torch.softmax(outputs, dim=1)
                        correct_class = prob[torch.arange(labels.size(0)), labels]
                        mask = correct_class < threshold

                if not mask.any():
                    continue

                inputs_misclassified = inputs[mask]
                labels_misclassified = labels[mask]

                # if inputs_misclassified.size(0) < 2:
                #     continue

                # if inputs_misclassified.size(0) < 2:
                #     required_samples = 2 - inputs_misclassified.size(0)
                #     correctly_classified_mask = ~mask
                #     correct_inputs = inputs[correctly_classified_mask][:required_samples]
                #     correct_labels = labels[correctly_classified_mask][:required_samples]

                #     inputs_misclassified = torch.cat((inputs_misclassified, correct_inputs), dim=0)
                #     labels_misclassified = torch.cat((labels_misclassified, correct_labels), dim=0)

                optimizer.zero_grad()

                outputs_misclassified = model(inputs_misclassified)
                # if task == "segmentation":
                #     outputs_misclassified = outputs_misclassified['out']
                # outputs_misclassified = outputs[mask]
                loss = criterion(outputs_misclassified, labels_misclassified)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # preds_misclassified = torch.argmax(outputs_misclassified, dim=1)
                # correct += (preds_misclassified == labels_misclassified).sum().item()
                # # total += labels_misclassified.size(0)
                # with torch.no_grad():
                    # outputs = model(inputs)
                    # preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                progress_bar.set_postfix({"Loss": loss.item()})

            epoch_loss = running_loss / len(train_loader)
            # epoch_accuracy = correct / total if total > 0 else 0
            epoch_accuracy = total_correct/total_samples if total_samples > 0 else 0 
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

            epoch_end_time = time.time()
            time_per_epoch.append(epoch_end_time-epoch_start_time)

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            model.eval()
            correct = 0
            total = 0
            test_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Evaluating"):
                    inputs = batch[0].to(device)
                    labels = batch[1].to(device)
                    outputs = model(inputs)
                    # if task == "segmentation":
                    #     outputs = outputs['out']

                    batch_loss = criterion(outputs, labels)
                    test_loss+=batch_loss.item()

                    predictions = torch.argmax(outputs, dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            accuracy = correct / total
            val_loss = test_loss / len(test_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
            scheduler.step(val_loss)
            epoch_test_accuracies.append(accuracy)
            epoch_test_losses.append(val_loss)

        else:
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            print(f"Epoch [{epoch+1/epochs}]")
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")


            for batch_idx, (inputs, labels) in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                # if task == "segmentation":
                #     outputs = outputs['out']
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
                outputs = model(inputs)
                # if task == "segmentation":
                #     outputs = outputs['out']
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct / total
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

            epoch_end_time = time.time()
            time_per_epoch.append(epoch_end_time - epoch_start_time)

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            model.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Evaluating"):
                    inputs = batch[0].to(device)
                    labels = batch[1].to(device)
                    outputs = model(inputs)
                    # if task == "segmentation":
                    #     outputs = outputs['out']

                    batch_loss = criterion(outputs, labels)
                    test_loss+=batch_loss.item()

                    predictions = torch.argmax(outputs, dim=-1)
                    test_correct += (predictions == labels).sum().item()
                    test_total += labels.size(0)

            accuracy = test_correct / test_total
            val_loss = test_loss / len(test_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
            scheduler.step(val_loss)
            epoch_test_accuracies.append(accuracy)
            epoch_test_losses.append(val_loss)


    end_time = time.time()
    log_memory(start_time, end_time)

    plot_metrics(epoch_losses, epoch_accuracies, "Revision")
    plot_metrics_test(epoch_test_accuracies, "Revisiong Test")
    # plot_accuracy_time(epoch_accuracies, time_per_epoch, title="Accuracy and Time per Epoch", save_path=save_path)
    plot_accuracy_time_multi(
    model_name=model_name,  
    accuracy=epoch_accuracies,
    time_per_epoch=time_per_epoch,  
    save_path=save_path,
    data_file=save_path
    )
    plot_accuracy_time_multi_test(
        model_name = model_name,
        accuracy=epoch_test_accuracies,
        time_per_epoch=time_per_epoch,
        save_path=save_path,
        data_file=save_path
    )

    return model

def train_baseline_longtail(model_name, model, train_loader, test_loader, device, epochs, save_path, cls_num_list):
    model.to(device)
    
    # criterion = nn.CrossEntropyLoss()
    train_sampler = None
    idx = epoch // 160
    betas = [0, 0.9999]
    effective_num = 1.0 - np.power(betas[idx], cls_num_list)
    per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    # optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
    epoch_losses = []
    epoch_accuracies = []
    epoch_test_accuracies = []
    epoch_test_losses = []
    time_per_epoch = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"Epoch [{epoch+1/epochs}]")
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")


        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

        epoch_end_time = time.time()
        time_per_epoch.append(epoch_end_time - epoch_start_time)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                test_loss+=batch_loss.item()
                predictions = torch.argmax(outputs, dim=-1)
                test_correct += (predictions == labels).sum().item()
                test_total += labels.size(0)

        accuracy = test_correct / test_total
        val_loss = test_loss/len(test_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
        scheduler.step(val_loss)
        epoch_test_accuracies.append(accuracy)
        epoch_test_losses.append(val_loss)

    end_time = time.time()
    log_memory(start_time, end_time)

    plot_metrics(epoch_losses, epoch_accuracies, "Baseline Training")
    plot_metrics_test(epoch_test_accuracies, "Baseline Training")
    plot_accuracy_time_multi(
    model_name=model_name,  
    accuracy=epoch_accuracies,
    time_per_epoch=time_per_epoch,  
    save_path=save_path,
    data_file=save_path
    )
    plot_accuracy_time_multi_test(
        model_name = model_name,
        accuracy=epoch_test_accuracies,
        time_per_epoch=time_per_epoch,
        save_path=save_path,
        data_file=save_path
    )
    return model