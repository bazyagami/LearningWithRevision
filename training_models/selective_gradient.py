import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import time
from utils import log_memory, plot_metrics, plot_metrics_test, plot_accuracy_time_multi, plot_accuracy_time_multi_test
from tqdm import tqdm
import json
import os
import numpy as np

class TrainRevision:
    def __init__(self, model_name, model, train_loader, test_loader, device, epochs, save_path, threshold):
        self.model_name = model_name
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.save_path = save_path
        self.threshold = threshold

    def train_selective(self):
        self.model.to(self.device)
        save_path = self.save_path
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
        # optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        start_time = time.time()
        for epoch in range(self.epochs):
            self.model.train()
            epoch_start_time = time.time()
            running_loss = 0.0
            correct = 0
            total_correct = 0
            total_samples = 0
            total = 0
            print(f"Epoch [{epoch+1/self.epochs}]")
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")

            for batch_idx, (inputs, labels) in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    
                    if self.threshold == 0:
                        mask = preds != labels
                    else:
                        prob = torch.softmax(outputs, dim=1)
                        correct_class = prob[torch.arange(labels.size(0)), labels]
                        mask = correct_class < self.threshold

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

                outputs_misclassified = self.model(inputs_misclassified)
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

            epoch_loss = running_loss / len(self.train_loader)
            # epoch_accuracy = correct / total if total > 0 else 0
            epoch_accuracy = total_correct/total_samples if total_samples > 0 else 0 
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

            epoch_end_time = time.time()
            time_per_epoch.append(epoch_end_time-epoch_start_time)

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in tqdm(self.test_loader, desc="Evaluating"):
                    inputs = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                    outputs = self.model(inputs)
                    predictions = torch.argmax(outputs, dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            accuracy = correct / total
            val_loss = criterion(correct, total)

            print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
            scheduler.step(val_loss)
            epoch_test_accuracies.append(accuracy)
            epoch_test_losses.append(val_loss)
            

        end_time = time.time()
        log_memory(start_time, end_time)

        plot_metrics(epoch_losses, epoch_accuracies, "Selective Training")
        plot_metrics_test(epoch_test_accuracies, "Selective Training")
        # plot_accuracy_time(epoch_accuracies, time_per_epoch, title="Accuracy and Time per Epoch", save_path=save_path)
        plot_accuracy_time_multi(
        model_name= self.model_name,  
        accuracy=epoch_accuracies,
        time_per_epoch=time_per_epoch,  
        save_path=save_path,
        data_file=save_path
        )
        plot_accuracy_time_multi_test(
            model_name = self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            save_path=save_path,
            data_file=save_path
        )

        return self.model

    def train_selective_epoch(self):
        self.model.to(self.device)
        save_path = self.save_path
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        epoch_losses = []
        epoch_accuracies = []
        time_per_epoch = []
        start_time = time.time()

        accumulated_inputs = []
        accumulated_labels = []
        max_accumulated_samples = 128

        for epoch in range(self.epochs):
            self.model.train()
            epoch_start_time = time.time()
            running_loss = 0.0
            total_correct = 0
            total_samples = 0

            print(f"Epoch [{epoch+1/self.epochs}]")
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")

            for batch_idx, (inputs, labels) in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if epoch < self.epochs:
                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)
                        
                        if self.threshold == 0:
                            mask = preds != labels
                            mask_correct = preds == labels
                        else:
                            prob = torch.softmax(outputs, dim=1)
                            correct_class = prob[torch.arange(labels.size(0)), labels]
                            mask = correct_class < self.threshold
                            mask_correct = correct_class > self.threshold

                    accumulated_inputs.append(inputs[mask_correct].cpu())
                    accumulated_labels.append(labels[mask_correct].cpu())

                    if len(accumulated_inputs) >= max_accumulated_samples:
                        reintroduced_inputs = torch.cat(accumulated_inputs, dim=0).to(self.device)
                        reintroduced_labels = torch.cat(accumulated_labels, dim=0).to(self.device)

                        accumulated_inputs = []  
                        accumulated_labels = []

                        inputs_selected = torch.cat((inputs, reintroduced_inputs), dim=0)
                        labels_selected = torch.cat((labels, reintroduced_labels), dim=0)

                    else:
                        if not mask.any():
                            continue

                        inputs_selected = inputs[mask]
                        labels_selected = labels[mask]

                    if not mask.any():
                        continue

                    inputs_selected = inputs[mask]
                    labels_selected = labels[mask]
                else:
                    if accumulated_inputs:
                        reintroduced_inputs = torch.cat(accumulated_inputs, dim=0).to(self.device)
                        reintroduced_labels = torch.cat(accumulated_labels, dim=0).to(self.device)

                        accumulated_inputs = []
                        accumulated_labels = []

                        inputs_selected = torch.cat((inputs, reintroduced_inputs), dim=0)
                        labels_selected = torch.cat((labels, reintroduced_labels), dim=0)
                    else:
                        print("No accumulated samples")
                        inputs_selected = inputs
                        labels_selected = labels

                optimizer.zero_grad()
                outputs_selected = self.model(inputs_selected)
                loss = criterion(outputs_selected, labels_selected)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                with torch.no_grad():
                    outputs = self.model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    total_correct += (preds == labels).sum().item()
                    total_samples += labels.size(0)
                
                progress_bar.set_postfix({"Loss": loss.item()})

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

            epoch_end_time = time.time()
            time_per_epoch.append(epoch_end_time - epoch_start_time)

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        end_time = time.time()
        log_memory(start_time, end_time)

        plot_metrics(epoch_losses, epoch_accuracies, "Selective Training with Reintroduction")
        plot_accuracy_time_multi(
            model_name=self.model_name,
            accuracy=epoch_accuracies,
            time_per_epoch=time_per_epoch,
            save_path=save_path,
            data_file=save_path
        )

        return self.model


    def train_with_revision(self, start_revision, task):

        save_path = self.save_path
        self.model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        #as per implementation LR=0.045, they use 16 GPU. https://discuss.pytorch.org/t/training-mobilenet-on-imagenet/174391/6 from this blog
        #we use the idea to divide the learning rate by the number of GPUs. 
        # optimizer = optim.RMSprop(self.model.parameters(), weight_decay=0.00004, momentum=0.9, lr=0.0028125)   
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        start_time = time.time()
        num_step = 0
        samples_used_per_epoch = []
        for epoch in range(self.epochs):
            samples_used = 0
            if epoch < start_revision : 
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                correct = 0
                total_correct = 0
                total_samples = 0
                total = 0
                print(f"Epoch [{epoch+1/self.epochs}]")
                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")
                
                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)
                        
                        if self.threshold == 0:
                            mask = preds != labels
                        else:
                            prob = torch.softmax(outputs, dim=1)
                            correct_class = prob[torch.arange(labels.size(0)), labels]
                            mask = correct_class < self.threshold

                    if not mask.any():
                        continue

                    inputs_misclassified = inputs[mask]
                    labels_misclassified = labels[mask]

                    optimizer.zero_grad()

                    outputs_misclassified = self.model(inputs_misclassified)
                    loss = criterion(outputs_misclassified, labels_misclassified)
                    num_step+=len(outputs_misclassified)
                    samples_used+=len(outputs_misclassified)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    total_correct += (preds == labels).sum().item()
                    total_samples += labels.size(0)
                    progress_bar.set_postfix({"Loss": loss.item()})

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = total_correct/total_samples if total_samples > 0 else 0 
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time-epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                self.model.eval()
                correct = 0
                total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss+=batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)

                accuracy = correct / total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)

            else:
                self.model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                print(f"Epoch [{epoch+1/self.epochs}]")
                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")


                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    num_step+=len(outputs)
                    samples_used+=len(outputs)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    
                    outputs = self.model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = correct / total
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                self.model.eval()
                test_correct = 0
                test_total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss+=batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        test_correct += (predictions == labels).sum().item()
                        test_total += labels.size(0)

                accuracy = test_correct / test_total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)
            
            samples_used_per_epoch.append(samples_used)


        end_time = time.time()
        log_memory(start_time, end_time)
        print(num_step)

        total_wall_time = end_time - start_time
        print(f"\n✅ Total Wall Time for {self.epochs} epochs: {total_wall_time:.2f} seconds "
            f"({total_wall_time / 60:.2f} minutes)")

        plot_accuracy_time_multi(
        model_name=self.model_name,  
        accuracy=epoch_accuracies,
        time_per_epoch=time_per_epoch,  
        save_path=save_path,
        data_file=save_path
        )
        plot_accuracy_time_multi_test(
            model_name = self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path
        )

        return self.model, num_step
    

    def train_with_random(self, start_revision, task):

        save_path = self.save_path
        self.model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        samples_used_per_epoch = []

        start_time = time.time()
        num_step = 0

        for epoch in range(self.epochs):
            samples_used = 0
            print(f"Epoch [{epoch+1}/{self.epochs}]")

            if epoch < start_revision:
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                total_correct = 0
                total_samples = 0

                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")

                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)

                        if self.threshold == 0:
                            mask = preds != labels
                        else:
                            prob = torch.softmax(outputs, dim=1)
                            correct_class = prob[torch.arange(labels.size(0)), labels]
                            mask = correct_class < self.threshold

                        num_to_select = mask.sum().item()

                    # Skip batch if no samples pass threshold
                    if num_to_select == 0:
                        continue

                    # 🔁 Random sampling based on how many passed threshold
                    indices = torch.randperm(inputs.size(0))[:num_to_select]
                    inputs_sampled = inputs[indices]
                    labels_sampled = labels[indices]

                    optimizer.zero_grad()
                    outputs_sampled = self.model(inputs_sampled)
                    loss = criterion(outputs_sampled, labels_sampled)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    num_step += len(outputs_sampled)
                    samples_used += len(outputs_sampled)

                    # Stats on original batch
                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)
                        total_correct += (preds == labels).sum().item()
                        total_samples += labels.size(0)

                    progress_bar.set_postfix({"Loss": loss.item()})

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                # Evaluation
                self.model.eval()
                correct = 0
                total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss += batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)

                accuracy = correct / total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)

            else:
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                correct = 0
                total = 0

                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")
                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    num_step += len(outputs)
                    samples_used += len(outputs)

                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)

                    progress_bar.set_postfix({"Loss": loss.item()})

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = correct / total
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                # Evaluation
                self.model.eval()
                test_correct = 0
                test_total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss += batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        test_correct += (predictions == labels).sum().item()
                        test_total += labels.size(0)

                accuracy = test_correct / test_total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)

            samples_used_per_epoch.append(samples_used)

        end_time = time.time()
        log_memory(start_time, end_time)
        print(num_step)

        # Visualization
        plot_accuracy_time_multi(
            model_name=self.model_name,  
            accuracy=epoch_accuracies,
            time_per_epoch=time_per_epoch,  
            save_path=save_path,
            data_file=save_path
        )
        plot_accuracy_time_multi_test(
            model_name=self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path
        )

        return self.model, num_step

    def train_with_revision_3d(self, start_revision, task):

        save_path = self.save_path
        self.model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        #as per implementation LR=0.045, they use 16 GPU. https://discuss.pytorch.org/t/training-mobilenet-on-imagenet/174391/6 from this blog
        #we use the idea to divide the learning rate by the number of GPUs. 
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        start_time = time.time()
        num_step = 0
        samples_used_per_epoch = []
        for epoch in range(self.epochs):
            samples_used = 0
            if epoch < start_revision : 
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                correct = 0
                total_correct = 0
                total_samples = 0
                total = 0
                print(f"Epoch [{epoch+1/self.epochs}]")
                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")
                
                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device).float(), labels.to(self.device).long().view(-1)
                    
                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)
                        
                        if self.threshold == 0:
                            mask = preds != labels
                        else:
                            prob = torch.softmax(outputs, dim=1)
                            correct_class = prob[torch.arange(labels.size(0)), labels]
                            mask = correct_class < self.threshold

                    if not mask.any():
                        continue

                    inputs_misclassified = inputs[mask]
                    labels_misclassified = labels[mask]

                    optimizer.zero_grad()

                    outputs_misclassified = self.model(inputs_misclassified)
                    loss = criterion(outputs_misclassified, labels_misclassified)
                    num_step+=len(outputs_misclassified)
                    samples_used+=len(outputs_misclassified)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    total_correct += (preds == labels).sum().item()
                    total_samples += labels.size(0)
                    progress_bar.set_postfix({"Loss": loss.item()})

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = total_correct/total_samples if total_samples > 0 else 0 
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time-epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                self.model.eval()
                correct = 0
                total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device).float()
                        labels = batch[1].to(self.device).long().view(-1)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss+=batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)

                accuracy = correct / total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)

            else:
                self.model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                print(f"Epoch [{epoch+1/self.epochs}]")
                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")


                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device).float(), labels.to(self.device).long().view(-1)

                    optimizer.zero_grad()

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    num_step+=len(outputs)
                    samples_used+=len(outputs)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    
                    outputs = self.model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = correct / total
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                self.model.eval()
                test_correct = 0
                test_total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device).float()
                        labels = batch[1].to(self.device).long().view(-1)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss+=batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        test_correct += (predictions == labels).sum().item()
                        test_total += labels.size(0)

                accuracy = test_correct / test_total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)
            
            samples_used_per_epoch.append(samples_used)


        end_time = time.time()
        log_memory(start_time, end_time)
        print(num_step)




        # plot_metrics(epoch_losses, epoch_accuracies, "Revision")
        # plot_metrics_test(epoch_test_accuracies, "Revision Test")
        plot_accuracy_time_multi(
        model_name=self.model_name,  
        accuracy=epoch_accuracies,
        time_per_epoch=time_per_epoch,  
        save_path=save_path,
        data_file=save_path
        )
        plot_accuracy_time_multi_test(
            model_name = self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path
        )

        return self.model, num_step
    

    def train_with_percentage(self, start_revision):
        save_path = self.save_path
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        samples_used_per_epoch = []
        num_step = 0
        start_time = time.time()

        for epoch in range(self.epochs):
            samples_used = 0
            self.model.train()
            epoch_start_time = time.time()
            running_loss = 0.0
            correct = 0
            total = 0
            total_correct = 0
            total_samples = 0

            print(f"Epoch [{epoch+1}/{self.epochs}]")
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")

            if epoch < start_revision:
                decay_factor = 0.95 ** epoch  ##percentage to be sampled
                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    batch_size = inputs.size(0)
                    selected_count = int(decay_factor * batch_size)

                    if selected_count == 0:
                        continue

                    selected_indices = torch.randperm(batch_size)[:selected_count]
                    inputs_selected = inputs[selected_indices]
                    labels_selected = labels[selected_indices]

                    optimizer.zero_grad()
                    outputs = self.model(inputs_selected)
                    loss = criterion(outputs, labels_selected)
                    num_step += selected_count
                    samples_used += selected_count
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    with torch.no_grad():
                        preds = torch.argmax(self.model(inputs), dim=1)
                        total_correct += (preds == labels).sum().item()
                        total_samples += labels.size(0)

                    progress_bar.set_postfix({"Loss": loss.item()})
            else:
                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    num_step += inputs.size(0)
                    samples_used += inputs.size(0)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = (
                total_correct / total_samples if epoch < start_revision else correct / total
            )
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

            epoch_end_time = time.time()
            time_per_epoch.append(epoch_end_time - epoch_start_time)

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            self.model.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(self.test_loader, desc="Evaluating"):
                    inputs = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                    outputs = self.model(inputs)
                    batch_loss = criterion(outputs, labels)
                    test_loss += batch_loss.item()
                    predictions = torch.argmax(outputs, dim=-1)
                    test_correct += (predictions == labels).sum().item()
                    test_total += labels.size(0)

            accuracy = test_correct / test_total
            val_loss = test_loss / len(self.test_loader)
            print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
            scheduler.step(val_loss)
            epoch_test_accuracies.append(accuracy)
            epoch_test_losses.append(val_loss)
            samples_used_per_epoch.append(samples_used)

        end_time = time.time()
        log_memory(start_time, end_time)
        print(num_step)

        plot_accuracy_time_multi(
            model_name=self.model_name,
            accuracy=epoch_accuracies,
            time_per_epoch=time_per_epoch,
            save_path=save_path,
            data_file=save_path,
        )
        plot_accuracy_time_multi_test(
            model_name=self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path,
        )

        return self.model, num_step

    def inverse_linear(self, epoch, alpha):
        if epoch == 200:
            return 50000
        x = np.arange(1, 200)
        y = 1 / (x + alpha)
        y_scaled = (y / np.max(y)) * 50000
        return y_scaled[epoch - 1]

    def train_with_inverse_linear(self, start_revision, data_size):
        save_path = self.save_path
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        samples_used_per_epoch = []
        num_step = 0
        alpha=2
        start_time = time.time()

        for epoch in range(self.epochs):
            samples_used = 0
            self.model.train()
            epoch_start_time = time.time()
            running_loss = 0.0
            correct = 0
            total = 0
            total_correct = 0
            total_samples = 0

            print(f"Epoch [{epoch+1}/{self.epochs}]")
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")

            if epoch < start_revision:
                # Use inverse linear decay
                scaled_value = self.inverse_linear(epoch + 1, alpha)  # epoch+1 to match 1-based indexing
                sample_ratio = scaled_value / data_size

                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    batch_size = inputs.size(0)
                    selected_count = int(sample_ratio * batch_size)

                    if selected_count == 0:
                        continue

                    selected_indices = torch.randperm(batch_size)[:selected_count]
                    inputs_selected = inputs[selected_indices]
                    labels_selected = labels[selected_indices]

                    optimizer.zero_grad()
                    outputs = self.model(inputs_selected)
                    loss = criterion(outputs, labels_selected)
                    num_step += selected_count
                    samples_used += selected_count
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    with torch.no_grad():
                        preds = torch.argmax(self.model(inputs), dim=1)
                        total_correct += (preds == labels).sum().item()
                        total_samples += labels.size(0)

                    progress_bar.set_postfix({"Loss": loss.item()})
            else:
                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    num_step += inputs.size(0)
                    samples_used += inputs.size(0)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = (
                total_correct / total_samples if epoch < start_revision else correct / total
            )
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

            epoch_end_time = time.time()
            time_per_epoch.append(epoch_end_time - epoch_start_time)

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            self.model.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(self.test_loader, desc="Evaluating"):
                    inputs = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                    outputs = self.model(inputs)
                    batch_loss = criterion(outputs, labels)
                    test_loss += batch_loss.item()
                    predictions = torch.argmax(outputs, dim=-1)
                    test_correct += (predictions == labels).sum().item()
                    test_total += labels.size(0)

            accuracy = test_correct / test_total
            val_loss = test_loss / len(self.test_loader)
            print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
            scheduler.step(val_loss)
            epoch_test_accuracies.append(accuracy)
            epoch_test_losses.append(val_loss)
            samples_used_per_epoch.append(samples_used)

        end_time = time.time()
        log_memory(start_time, end_time)
        print(num_step)

        plot_accuracy_time_multi(
            model_name=self.model_name,
            accuracy=epoch_accuracies,
            time_per_epoch=time_per_epoch,
            save_path=save_path,
            data_file=save_path,
        )
        plot_accuracy_time_multi_test(
            model_name=self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path,
        )

        return self.model, num_step
    
    def log_schedule(self, epoch, data_size, alpha):
        x = np.arange(1, 200)
        y = 1 / np.log(x + alpha)  # make sure alpha > 1 to avoid log(0)
        y_scaled = (y / np.max(y)) * data_size
        return y_scaled[epoch - 1]

    def train_with_log(self, start_revision, data_size):
        save_path = self.save_path
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        samples_used_per_epoch = []
        num_step = 0
        alpha=2
        start_time = time.time()

        for epoch in range(self.epochs):
            samples_used = 0
            self.model.train()
            epoch_start_time = time.time()
            running_loss = 0.0
            correct = 0
            total = 0
            total_correct = 0
            total_samples = 0

            print(f"Epoch [{epoch+1}/{self.epochs}]")
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")

            if epoch < start_revision:
                # Use inverse linear decay
                scaled_value = self.log_schedule(epoch + 1, data_size, alpha)  # epoch+1 to match 1-based indexing
                sample_ratio = scaled_value / data_size

                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    batch_size = inputs.size(0)
                    selected_count = int(sample_ratio * batch_size)

                    if selected_count == 0:
                        continue

                    selected_indices = torch.randperm(batch_size)[:selected_count]
                    inputs_selected = inputs[selected_indices]
                    labels_selected = labels[selected_indices]

                    optimizer.zero_grad()
                    outputs = self.model(inputs_selected)
                    loss = criterion(outputs, labels_selected)
                    num_step += selected_count
                    samples_used += selected_count
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    with torch.no_grad():
                        preds = torch.argmax(self.model(inputs), dim=1)
                        total_correct += (preds == labels).sum().item()
                        total_samples += labels.size(0)

                    progress_bar.set_postfix({"Loss": loss.item()})
            else:
                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    num_step += inputs.size(0)
                    samples_used += inputs.size(0)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = (
                total_correct / total_samples if epoch < start_revision else correct / total
            )
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

            epoch_end_time = time.time()
            time_per_epoch.append(epoch_end_time - epoch_start_time)

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            self.model.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(self.test_loader, desc="Evaluating"):
                    inputs = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                    outputs = self.model(inputs)
                    batch_loss = criterion(outputs, labels)
                    test_loss += batch_loss.item()
                    predictions = torch.argmax(outputs, dim=-1)
                    test_correct += (predictions == labels).sum().item()
                    test_total += labels.size(0)

            accuracy = test_correct / test_total
            val_loss = test_loss / len(self.test_loader)
            print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
            scheduler.step(val_loss)
            epoch_test_accuracies.append(accuracy)
            epoch_test_losses.append(val_loss)
            samples_used_per_epoch.append(samples_used)

        end_time = time.time()
        log_memory(start_time, end_time)
        print(num_step)

        plot_accuracy_time_multi(
            model_name=self.model_name,
            accuracy=epoch_accuracies,
            time_per_epoch=time_per_epoch,
            save_path=save_path,
            data_file=save_path,
        )
        plot_accuracy_time_multi_test(
            model_name=self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path,
        )

        return self.model, num_step 