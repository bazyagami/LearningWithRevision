import torch
import torch.nn as nn
import torch.optim as optim
import time
from utils import log_memory, plot_metrics, plot_metrics_test, plot_accuracy_time_multi, plot_accuracy_time_multi_test
from tqdm import tqdm

def train_selective(model_name, model, train_loader, test_loader, device, epochs, save_path, threshold):
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    epoch_losses = []
    epoch_accuracies = []
    epoch_test_accuracies = []
    removed_samples_batch = []
    time_per_epoch = []
    start_time = time.time()
    for epoch in range(epochs):
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
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Test Accuracy: {accuracy:.4f}")
        epoch_test_accuracies.append(accuracy)
        

    end_time = time.time()
    log_memory(start_time, end_time)

    plot_metrics(epoch_losses, epoch_accuracies, "Selective Training")
    plot_metrics_test(epoch_test_accuracies, "Selective Training")
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

def train_selective_epoch(model_name, model, train_loader, device, epochs, save_path, threshold):
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    epoch_losses = []
    epoch_accuracies = []
    time_per_epoch = []
    start_time = time.time()

    accumulated_inputs = []
    accumulated_labels = []
    max_accumulated_samples = 128

    for epoch in range(epochs):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        print(f"Epoch [{epoch+1/epochs}]")
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")

        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            if epoch < epochs:
                with torch.no_grad():
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    
                    if threshold == 0:
                        mask = preds != labels
                        mask_correct = preds == labels
                    else:
                        prob = torch.softmax(outputs, dim=1)
                        correct_class = prob[torch.arange(labels.size(0)), labels]
                        mask = correct_class < threshold
                        mask_correct = correct_class > threshold

                accumulated_inputs.append(inputs[mask_correct].cpu())
                accumulated_labels.append(labels[mask_correct].cpu())

                if len(accumulated_inputs) >= max_accumulated_samples:
                    reintroduced_inputs = torch.cat(accumulated_inputs, dim=0).to(device)
                    reintroduced_labels = torch.cat(accumulated_labels, dim=0).to(device)

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
                    reintroduced_inputs = torch.cat(accumulated_inputs, dim=0).to(device)
                    reintroduced_labels = torch.cat(accumulated_labels, dim=0).to(device)

                    accumulated_inputs = []
                    accumulated_labels = []

                    inputs_selected = torch.cat((inputs, reintroduced_inputs), dim=0)
                    labels_selected = torch.cat((labels, reintroduced_labels), dim=0)
                else:
                    print("No accumulated samples")
                    inputs_selected = inputs
                    labels_selected = labels

            optimizer.zero_grad()
            outputs_selected = model(inputs_selected)
            loss = criterion(outputs_selected, labels_selected)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            with torch.no_grad():
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
            
            progress_bar.set_postfix({"Loss": loss.item()})

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

        epoch_end_time = time.time()
        time_per_epoch.append(epoch_end_time - epoch_start_time)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    end_time = time.time()
    log_memory(start_time, end_time)

    plot_metrics(epoch_losses, epoch_accuracies, "Selective Training with Reintroduction")
    plot_accuracy_time_multi(
        model_name=model_name,
        accuracy=epoch_accuracies,
        time_per_epoch=time_per_epoch,
        save_path=save_path,
        data_file=save_path
    )

    return model


def train_with_revision(model_name, model, train_loader, test_loader, device, epochs, save_path, threshold, start_revision):
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    epoch_losses = []
    epoch_accuracies = []
    epoch_test_accuracies = []
    removed_samples_batch = []
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
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Evaluating"):
                    inputs = batch[0].to(device)
                    labels = batch[1].to(device)
                    outputs = model(inputs)
                    predictions = torch.argmax(outputs, dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            accuracy = correct / total
            print(f"Epoch {epoch + 1}/{epochs}, Test Accuracy: {accuracy:.4f}")
            epoch_test_accuracies.append(accuracy)

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
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Evaluating"):
                    inputs = batch[0].to(device)
                    labels = batch[1].to(device)
                    outputs = model(inputs)
                    predictions = torch.argmax(outputs, dim=-1)
                    test_correct += (predictions == labels).sum().item()
                    test_total += labels.size(0)

            accuracy = test_correct / test_total
            print(f"Epoch {epoch + 1}/{epochs}, Test Accuracy: {accuracy:.4f}")
            epoch_test_accuracies.append(accuracy)


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