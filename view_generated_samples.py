import numpy as np
import matplotlib.pyplot as plt

def save_ecg_image(numpy_array, output_file_name):
    """
    Saves an ECG-like image showing the middle channels [0, i, :] from a NumPy array.
    
    Parameters:
    - numpy_array: NumPy array of shape [400, 12, 1000]
    - output_file_name: Name of the file to save the ECG-like image
    """
    # Number of channels
    num_channels = numpy_array.shape[1]
    
    # Create a figure with subplots arranged in a grid (2 columns, 6 rows)
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(10, 15), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # Plot each middle channel
    for i in range(num_channels):
        sample = numpy_array[0, i, :]
        axes[i].plot(sample)
        axes[i].set_title(f'Channel {i}')
        axes[i].grid(True, color='red', linestyle='-', linewidth=1.5)
        axes[i].label_outer()  # Hide x and y labels for outer subplots

    # Adjust layout
    plt.tight_layout()
    
    # Save the plot to the specified file
    plt.savefig(output_file_name)
    plt.close()



# Example usage:
# numpy_array = np.random.rand(400, 12, 1000)  # Example array
numpy_array = np.load("src/sssd/sssd_label_cond/ch256_T200_betaT0.02/0_samples.npy")
save_ecg_image(numpy_array, 'sample_plot.png')

