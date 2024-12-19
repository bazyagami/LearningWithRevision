import psutil
import os
import matplotlib.pyplot as plt

def log_memory(start_time, end_time):
    process = psutil.Process(os.getpid())
    print(f"Training Time: {end_time - start_time:.2f} seconds")
    print(f"Memory Consumption: {process.memory_info().rss / (1024 * 1024):.2f} MB")

def plot_training_time(times_dict, save_path="training_time.png"):
    """
    Plot a bar chart comparing total training time for different experiments.
    Args:
        times_dict (dict): Dictionary with experiment names as keys and training time (in seconds) as values.
        save_path (str): File path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.bar(times_dict.keys(), times_dict.values(), color=['blue', 'orange'])
    plt.xlabel("Experiments")
    plt.ylabel("Training Time (seconds)")
    plt.title("Total Training Time Comparison")
    plt.savefig(save_path)
    print(f"Training time plot saved to {save_path}")
    plt.close()

def plot_metrics(losses, accuracies, title):
    epochs = range(1, len(losses) + 1)
    
    # Plotting loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{title} - Loss')
    plt.legend()
    
    # Plotting accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend()
    
    # Save and show plots
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_metrics.png')
    plt.show()
