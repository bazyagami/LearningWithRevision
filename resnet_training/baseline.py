import torch
import torch.nn as nn
import torch.optim as optim
import time
from utils import log_memory, plot_metrics, plot_accuracy_time_multi


def train_baseline(model_name, model, train_loader, device, epochs, save_path):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    epoch_losses = []
    epoch_accuracies = []
    time_per_epoch = []
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
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

    end_time = time.time()
    log_memory(start_time, end_time)

    plot_metrics(epoch_losses, epoch_accuracies, "Baseline Training")
    plot_accuracy_time_multi(
    model_name=model_name,  
    accuracy=epoch_accuracies,
    time_per_epoch=time_per_epoch,  
    save_path=save_path,
    data_file=save_path
    )
    return model