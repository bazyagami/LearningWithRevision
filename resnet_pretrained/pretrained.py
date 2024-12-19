import torch
import torch.nn as nn
import torch.optim as optim
import time
from utils import log_memory, plot_metrics


# def train_pretrained_baseline(model, train_loader, device, epochs=10):
#     model.to(device)
#     model.train()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
#     start_time = time.time()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
#     end_time = time.time()
#     log_memory(start_time, end_time)

# def train_pretrained_selective(model, train_loader, device, epochs=10):
#     model.to(device)
#     model.train()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
#     start_time = time.time()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             with torch.no_grad():
#                 outputs = model(inputs)
#                 preds = torch.argmax(outputs, dim=1)
#                 mask = preds != labels
#             if not mask.any():
#                 continue
#             inputs_misclassified = inputs[mask]
#             labels_misclassified = labels[mask]
#             optimizer.zero_grad()
#             outputs = model(inputs_misclassified)
#             loss = criterion(outputs, labels_misclassified)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
#     end_time = time.time()
#     log_memory(start_time, end_time)


def train_pretrained_baseline(model, train_loader, device, epochs=10):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Metrics storage
    epoch_losses = []
    epoch_accuracies = []
    start_time = time.time()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Calculate and store metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    end_time = time.time()
    log_memory(start_time, end_time)

    # Plot and save metrics
    plot_metrics(epoch_losses, epoch_accuracies, "Pretrained Baseline Training")


def train_pretrained_selective(model, train_loader, device, epochs=10):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Metrics storage
    epoch_losses = []
    epoch_accuracies = []
    start_time = time.time()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass to identify misclassified samples
            with torch.no_grad():
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                mask = preds != labels

            # Skip batch if all samples are correctly classified
            if not mask.any():
                continue

            inputs_misclassified = inputs[mask]
            labels_misclassified = labels[mask]

            optimizer.zero_grad()

            # Forward and backward pass on misclassified samples
            outputs = model(inputs_misclassified)
            loss = criterion(outputs, labels_misclassified)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy on misclassified samples
            preds_misclassified = torch.argmax(outputs, dim=1)
            correct += (preds_misclassified == labels_misclassified).sum().item()
            total += labels_misclassified.size(0)

        # Calculate and store metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total if total > 0 else 0
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    end_time = time.time()
    log_memory(start_time, end_time)

    # Plot and save metrics
    plot_metrics(epoch_losses, epoch_accuracies, "Pretrained Selective Training")