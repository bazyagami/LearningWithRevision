import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import time
from utils import log_memory, plot_metrics, plot_metrics_test, plot_accuracy_time_multi, plot_accuracy_time_multi_test
from tqdm import tqdm


def train_baseline(model_name, model, train_loader, test_loader, device, epochs, save_path):
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
    epoch_losses = []
    epoch_accuracies = []
    epoch_test_accuracies = []
    epoch_test_losses = []
    time_per_epoch = []
    start_time = time.time()
    num_step = 0
    samples_used_per_epoch = []

    for epoch in range(epochs):
        samples_used = 0
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
            num_step+=len(outputs)
            samples_used+=len(outputs)

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
    samples_used_per_epoch.append(samples_used)
    log_memory(start_time, end_time)
    print(num_step)

    # plot_metrics(epoch_losses, epoch_accuracies, "Baseline Training")
    # plot_metrics_test(epoch_test_accuracies, "Baseline Training")
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
        samples_per_epoch=samples_used_per_epoch,
        threshold=None,
        save_path=save_path,
        data_file=save_path
    )
    return model