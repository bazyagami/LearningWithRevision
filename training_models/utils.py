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
    # Ensure directory for the plot image exists
    plot_output_dir = os.path.dirname(save_path)
    if plot_output_dir and not os.path.exists(plot_output_dir):
        os.makedirs(plot_output_dir, exist_ok=True)
        print(f"Info: Created directory '{plot_output_dir}' for plot image.")

    plt.figure(figsize=(8, 6))
    plt.bar(times_dict.keys(), times_dict.values(), color=['blue', 'orange'])
    plt.xlabel("Experiments")
    plt.ylabel("Training Time (seconds)")
    plt.title("Total Training Time Comparison")
    try:
        plt.savefig(save_path)
        print(f"Training time plot saved to {save_path}")
    except Exception as e:
        print(f"Error: Could not save plot to '{save_path}': {e}")
    plt.close()

def plot_metrics(losses, accuracies, title):
    epochs = range(1, len(losses) + 1)
    save_filename = f'{title.lower().replace(" ", "_")}_metrics.png'
    # Ensure directory for the plot image exists
    plot_output_dir = os.path.dirname(save_filename)
    if plot_output_dir and not os.path.exists(plot_output_dir):  # Check if plot_output_dir is not empty
        os.makedirs(plot_output_dir, exist_ok=True)
        print(f"Info: Created directory '{plot_output_dir}' for plot image '{save_filename}'.")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{title} - Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend()

    plt.tight_layout()
    try:
        plt.savefig(save_filename)
        print(f"Info: Metrics plot saved to '{save_filename}'.")
    except Exception as e:
        print(f"Error: Could not save metrics plot to '{save_filename}': {e}")
    # plt.show()
    plt.close()


def plot_metrics_test(accuracies, title):
    epochs = range(1, len(accuracies) + 1)
    save_filename = f'{title.lower().replace(" ", "_")}_metrics_test.png'
    # Ensure directory for the plot image exists
    plot_output_dir = os.path.dirname(save_filename)
    if plot_output_dir and not os.path.exists(plot_output_dir):  # Check if plot_output_dir is not empty
        os.makedirs(plot_output_dir, exist_ok=True)
        print(f"Info: Created directory '{plot_output_dir}' for plot image '{save_filename}'.")

    # Note: Original code had subplot(1,2,2) which might be part of a larger figure.
    # For a standalone plot, plt.figure() might be needed if not already called.
    # Assuming it's okay as is, or part of an existing figure setup.
    plt.figure(figsize=(6, 5))  # Added figure for standalone plotting
    plt.plot(epochs, accuracies, label='Accuracy', color='green')  # Removed subplot for single plot
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend()

    plt.tight_layout()
    try:
        plt.savefig(save_filename)
        print(f"Info: Test metrics plot saved to '{save_filename}'.")
    except Exception as e:
        print(f"Error: Could not save test metrics plot to '{save_filename}': {e}")
    # plt.show()
    plt.close()


def plot_accuracy_time(accuracy, time_per_epoch, title="Accuracy and Time per Epoch", save_path=None):
    epochs = range(1, len(accuracy) + 1)

    if save_path:
        # Ensure directory for the plot image exists
        plot_output_dir = os.path.dirname(save_path)
        if plot_output_dir and not os.path.exists(plot_output_dir):
            os.makedirs(plot_output_dir, exist_ok=True)
            print(f"Info: Created directory '{plot_output_dir}' for plot image '{save_path}'.")

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy", color="tab:blue")
    ax1.plot(epochs, accuracy, label="Accuracy", color="tab:blue", marker="o")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Time (seconds)", color="tab:orange")
    ax2.plot(epochs, time_per_epoch, label="Time", color="tab:orange", marker="s")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title(title)
    fig.tight_layout()
    plt.grid(visible=True, which="both", linestyle="--", linewidth=0.5)

    if save_path:
        try:
            plt.savefig(save_path)
            print(f"Info: Accuracy/Time plot saved to '{save_path}'.")
        except Exception as e:
            print(f"Error: Could not save accuracy/time plot to '{save_path}': {e}")
    # plt.show()
    plt.close(fig)


def plot_accuracy_time_multi(model_name, accuracy, time_per_epoch, save_path="accuracy_vs_time_plot.png",
                             data_file="model_data.json"):
    cumulative_time = [0] + [sum(time_per_epoch[:i + 1]) for i in range(len(time_per_epoch))]

    # Determine the actual path for the JSON data file
    default_json_filename = "model_data.json"
    actual_json_file_path = data_file

    if os.path.isdir(data_file):
        # data_file argument is an existing directory.
        # The JSON file should be created/updated inside this directory.
        print(f"Info: '{data_file}' is a directory. Using '{default_json_filename}' inside it for data storage.")
        actual_json_file_path = os.path.join(data_file, default_json_filename)

    # Ensure the directory for the JSON file exists
    json_dir = os.path.dirname(actual_json_file_path)
    if json_dir and not os.path.exists(json_dir):  # Check if json_dir is not empty
        os.makedirs(json_dir, exist_ok=True)
        print(f"Info: Created directory '{json_dir}' for JSON data file.")

    # Load existing data if the JSON file exists and is a file
    all_model_data = {}
    if os.path.exists(actual_json_file_path):
        if os.path.isfile(actual_json_file_path):
            try:
                with open(actual_json_file_path, "r") as f:
                    content = f.read()
                    if content.strip():  # Ensure file is not empty
                        all_model_data = json.loads(content)
                    else:
                        all_model_data = {}
                        print(f"Info: JSON file '{actual_json_file_path}' is empty. Initializing fresh data.")
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not decode JSON from '{actual_json_file_path}'. File might be corrupted or malformed. Initializing fresh data.")
                all_model_data = {}
            except Exception as e:
                print(
                    f"Warning: An error occurred while reading '{actual_json_file_path}': {e}. Initializing fresh data.")
                all_model_data = {}
        else:
            print(
                f"Warning: Path '{actual_json_file_path}' intended for JSON file is a directory. Data will be initialized fresh.")
            all_model_data = {}
    else:
        print(f"Info: JSON file '{actual_json_file_path}' not found. Initializing fresh data.")
        all_model_data = {}

    # Update data
    all_model_data[model_name] = {
        "cumulative_time": cumulative_time[1:],
        "accuracy": accuracy
    }

    # Write data to JSON file
    try:
        with open(actual_json_file_path, "w") as f:
            json.dump(all_model_data, f, indent=4)
        print(f"Info: Data for '{model_name}' saved to '{actual_json_file_path}'.")
    except Exception as e:
        print(f"Error: Could not write JSON data to '{actual_json_file_path}': {e}")

    # Ensure directory for the plot image exists
    plot_output_dir = os.path.dirname(save_path)
    if plot_output_dir and not os.path.exists(plot_output_dir):  # Check if plot_output_dir is not empty
        os.makedirs(plot_output_dir, exist_ok=True)
        print(f"Info: Created directory '{plot_output_dir}' for plot image '{save_path}'.")

    # Plotting
    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10.colors
    has_data_to_plot = False
    for idx, (name, data) in enumerate(all_model_data.items()):
        if isinstance(data, dict) and "cumulative_time" in data and "accuracy" in data:
            plt.plot(
                data["cumulative_time"],
                data["accuracy"],
                label=name,
                color=colors[idx % len(colors)],
                marker="o"
            )
            has_data_to_plot = True
        else:
            print(
                f"Warning: Missing 'cumulative_time' or 'accuracy' for model '{name}' in data, or data is not a dictionary. Skipping plot for this model entry.")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Time for Multiple Models")
    if has_data_to_plot:
        plt.legend()
    plt.grid(visible=True, which="both", linestyle="--", linewidth=0.5)

    try:
        plt.savefig(save_path)
        print(f"Info: Plot saved to '{save_path}'.")
    except Exception as e:
        print(f"Error: Could not save plot to '{save_path}': {e}")
    # plt.show()
    plt.close()


def plot_accuracy_time_multi_test(model_name, accuracy, time_per_epoch, samples_per_epoch, threshold,
                                  save_path="accuracy_vs_time_plot.png", data_file="model_data.json"):
    cumulative_time = [0] + [sum(time_per_epoch[:i + 1]) for i in range(len(time_per_epoch))]

    # Modify paths as per original logic, these will be treated as file paths
    # The directory part of these paths will be created if it doesn't exist.
    processed_data_file = data_file + "_test"
    processed_save_path = save_path + "_test"

    if threshold is not None:  # Check for None explicitly
        try:
            threshold_str = f"{int(threshold * 100):02d}"
        except (ValueError, TypeError):
            print(f"Warning: Invalid threshold value '{threshold}'. Using 'unknown_threshold'.")
            threshold_str = "unknown_threshold"  # Fallback for invalid threshold
    else:
        threshold_str = "None"  # Or "no_threshold"

    # Ensure the epoch_save path ends with a common image extension like .png
    base_epoch_save_path = processed_save_path + f"_epochs_{threshold_str}"
    if not base_epoch_save_path.lower().endswith(('.png', '.jpg', '.jpeg', '.svg')):
        processed_epoch_save_path = base_epoch_save_path + ".png"
    else:
        processed_epoch_save_path = base_epoch_save_path

    # --- Directory creation and file handling for processed_data_file (JSON) ---
    data_file_dir = os.path.dirname(processed_data_file)
    if data_file_dir and not os.path.exists(data_file_dir):  # Check if data_file_dir is not empty
        os.makedirs(data_file_dir, exist_ok=True)
        print(f"Info: Created directory '{data_file_dir}' for test data file '{processed_data_file}'.")

    all_model_data = {}
    if os.path.exists(processed_data_file):
        if os.path.isfile(processed_data_file):
            try:
                with open(processed_data_file, "r") as f:
                    content = f.read()
                    if content.strip():
                        all_model_data = json.loads(content)
                    else:
                        all_model_data = {}
                        print(f"Info: Test data JSON file '{processed_data_file}' is empty. Initializing fresh data.")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from '{processed_data_file}'. Initializing fresh data.")
                all_model_data = {}
            except Exception as e:
                print(f"Warning: Error reading '{processed_data_file}': {e}. Initializing fresh data.")
                all_model_data = {}
        else:
            print(f"Warning: Path '{processed_data_file}' for test data JSON is a directory. Initializing fresh data.")
            all_model_data = {}
    else:
        print(f"Info: Test data JSON file '{processed_data_file}' not found. Initializing fresh data.")
        all_model_data = {}

    all_model_data[model_name] = {
        "cumulative_time": cumulative_time[1:],
        "accuracy": accuracy
    }

    try:
        with open(processed_data_file, "w") as f:
            json.dump(all_model_data, f, indent=4)
        print(f"Info: Test data for '{model_name}' saved to '{processed_data_file}'.")
    except Exception as e:
        print(f"Error: Could not write test data to '{processed_data_file}': {e}")

    # --- Plotting Test Accuracy vs Time (main plot) ---
    save_path_dir = os.path.dirname(processed_save_path)
    if save_path_dir and not os.path.exists(save_path_dir):  # Check if save_path_dir is not empty
        os.makedirs(save_path_dir, exist_ok=True)
        print(f"Info: Created directory '{save_path_dir}' for test plot image '{processed_save_path}'.")

    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10.colors
    has_data_to_plot_main = False
    for idx, (name, data) in enumerate(all_model_data.items()):
        if isinstance(data, dict) and "cumulative_time" in data and "accuracy" in data:
            plt.plot(
                data["cumulative_time"],
                data["accuracy"],
                label=name,
                color=colors[idx % len(colors)],
                marker="o"
            )
            has_data_to_plot_main = True
        else:
            print(f"Warning: Missing data for main test plot for model '{name}'. Skipping.")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy vs Time for Multiple Models")
    if has_data_to_plot_main:
        plt.legend()
    plt.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    try:
        plt.savefig(processed_save_path)
        print(f"Info: Test accuracy plot saved to '{processed_save_path}'.")
    except Exception as e:
        print(f"Error: Could not save test accuracy plot to '{processed_save_path}': {e}")
    plt.close()

    # --- Plotting Number of Samples Used Per Epoch (samples plot) ---
    epoch_save_dir = os.path.dirname(processed_epoch_save_path)
    if epoch_save_dir and not os.path.exists(epoch_save_dir):  # Check if epoch_save_dir is not empty
        os.makedirs(epoch_save_dir, exist_ok=True)
        print(f"Info: Created directory '{epoch_save_dir}' for epoch samples plot '{processed_epoch_save_path}'.")

    plt.figure(figsize=(8, 5))
    if samples_per_epoch:  # Check if samples_per_epoch is not empty
        plt.plot(range(1, len(samples_per_epoch) + 1), samples_per_epoch, marker="o", linestyle="-", color="b",
                 label="Samples Used")
        plt.xlabel("Epoch")
        plt.ylabel("Number of Samples Used")
        plt.title("Number of Samples Used Per Epoch")
        plt.legend()
        plt.grid(True)
        try:
            plt.savefig(processed_epoch_save_path)
            print(f"Info: Epoch samples plot saved to '{processed_epoch_save_path}'.")
        except Exception as e:
            print(f"Error: Could not save epoch samples plot to '{processed_epoch_save_path}': {e}")
    else:
        print("Info: No data for samples_per_epoch. Skipping samples plot.")
    plt.close()


# def plot_epoch_samples(samples_per_epoch, save_path, threshold):
#     threshold_str = f"{int(threshold * 100):02d}"
#     # save_path = save_path + f"\samples_per_epoch_{threshold_str}"
#     save_path = os.path.join(save_path, f"samples_per_epoch_{threshold_str}")

#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.figure(figsize=(8, 5))
#     plt.plot(range(1, len(samples_per_epoch) + 1), samples_per_epoch, marker="o", linestyle="-", color="b", label="Samples Used")
#     plt.xlabel("Epoch")
#     plt.ylabel("Number of Samples Used")
#     plt.title("Number of Samples Used Per Epoch")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(save_path)