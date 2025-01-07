import torch
import torch.nn as nn
import torch.optim as optim
import time
from utils import log_memory, plot_metrics, plot_accuracy_time, plot_accuracy_time_multi

def train_selective(model_name, model, train_loader, device, epochs, save_path, threshold):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    epoch_losses = []
    epoch_accuracies = []
    removed_samples_batch = []
    time_per_epoch = []
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0
        total_correct = 0
        total_samples = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
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

            if inputs_misclassified.size(0) < 2:
                required_samples = 2 - inputs_misclassified.size(0)
                correctly_classified_mask = ~mask
                correct_inputs = inputs[correctly_classified_mask][:required_samples]
                correct_labels = labels[correctly_classified_mask][:required_samples]

                inputs_misclassified = torch.cat((inputs_misclassified, correct_inputs), dim=0)
                labels_misclassified = torch.cat((labels_misclassified, correct_labels), dim=0)

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

        epoch_loss = running_loss / len(train_loader)
        # epoch_accuracy = correct / total if total > 0 else 0
        epoch_accuracy = total_correct/total_samples if total_samples > 0 else 0 
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

        epoch_end_time = time.time()
        time_per_epoch.append(epoch_end_time-epoch_start_time)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    end_time = time.time()
    log_memory(start_time, end_time)

    plot_metrics(epoch_losses, epoch_accuracies, "Selective Training")
    # plot_accuracy_time(epoch_accuracies, time_per_epoch, title="Accuracy and Time per Epoch", save_path=save_path)
    plot_accuracy_time_multi(
    model_name=model_name,  
    accuracy=epoch_accuracies,
    time_per_epoch=time_per_epoch,  
    save_path=save_path,
    data_file=save_path
    )

    return model

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_accuracy = correct / total if total > 0 else 0
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy



    