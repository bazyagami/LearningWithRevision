import torch
import torch.nn as nn
import torch.optim as optim
import time
from utils import log_memory, plot_metrics


def train_selective(model, train_loader, device, epochs=10):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    epoch_losses = []
    epoch_accuracies = []
    removed_samples_batch = []
    start_time = time.time()
    log_file = "removed_samples_log.txt"
    with open(log_file, "w") as f:
        f.write("Epoch,Batch,Removed Samples\n")
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                with torch.no_grad():
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    mask = preds != labels

                removed_samples = (~mask).sum().item()
                removed_samples_batch.append(removed_samples)
                # print(f"Batch [{batch_idx+1}/{len(train_loader)}], Removed Samples: {removed_samples}")
                f.write(f"{epoch+1},{batch_idx+1},{removed_samples}\n")

                if not mask.any():
                    continue

                inputs_misclassified = inputs[mask]
                labels_misclassified = labels[mask]

                optimizer.zero_grad()

                outputs = model(inputs_misclassified)
                loss = criterion(outputs, labels_misclassified)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                preds_misclassified = torch.argmax(outputs, dim=1)
                correct += (preds_misclassified == labels_misclassified).sum().item()
                total += labels_misclassified.size(0)

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct / total if total > 0 else 0
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    end_time = time.time()
    log_memory(start_time, end_time)

    plot_metrics(epoch_losses, epoch_accuracies, "Selective Training")