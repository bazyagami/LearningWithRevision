import psutil
import os
import matplotlib.pyplot as plt
import json

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

def plot_accuracy_time(accuracy, time_per_epoch, title="Accuracy and Time per Epoch", save_path=None):
    epochs = range(1, len(accuracy) + 1)

    fig, ax1 = plt.subplots()

    # Plot accuracy on the left y-axis
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy", color="tab:blue")
    ax1.plot(epochs, accuracy, label="Accuracy", color="tab:blue", marker="o")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Create a second y-axis to plot time
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time (seconds)", color="tab:orange")
    ax2.plot(epochs, time_per_epoch, label="Time", color="tab:orange", marker="s")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Add title and grid
    plt.title(title)
    fig.tight_layout()
    plt.grid(visible=True, which="both", linestyle="--", linewidth=0.5)

    # Show the plot
    if save_path:
        plt.savefig(save_path)
    plt.show()



def plot_accuracy_time_multi(model_name, accuracy, time_per_epoch, save_path="accuracy_vs_time_plot.png", data_file="model_data.json"):
    # Calculate the cumulative training time
    cumulative_time = [0] + [sum(time_per_epoch[:i + 1]) for i in range(len(time_per_epoch))]
    
    # If the data file exists, load existing data
    if os.path.exists(data_file):
        with open(data_file, "r") as f:
            all_model_data = json.load(f)
    else:
        all_model_data = {}

    # Add the new model data
    all_model_data[model_name] = {
        "cumulative_time": cumulative_time[1:],  # Exclude the initial "0"
        "accuracy": accuracy
    }

    # Save the updated data back to the file
    with open(data_file, "w") as f:
        json.dump(all_model_data, f, indent=4)

    # Plot all models
    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10.colors  # Use a colormap for distinct model plots
    for idx, (name, data) in enumerate(all_model_data.items()):
        plt.plot(
            data["cumulative_time"],
            data["accuracy"],
            label=name,
            color=colors[idx % len(colors)],
            marker="o"
        )

    # Add labels, title, legend, and grid
    plt.xlabel("Time (seconds)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Time for Multiple Models")
    plt.legend()
    plt.grid(visible=True, which="both", linestyle="--", linewidth=0.5)

    # Save the updated plot
    plt.savefig(save_path)
    plt.show()
