import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def save_ecg_image(numpy_array, output_file_name):
    """
    Saves an ECG-like image showing the middle channels [0, i, :] from a NumPy array.
    
    Parameters:
    - numpy_array: NumPy array of shape [400, 12, 1000]
    - output_file_name: Name of the file to save the ECG-like image (e.g., 'ecg_image.jpg')
    """
    # Number of channels
    num_channels = numpy_array.shape[1]
    
    # Create a figure with subplots arranged in a grid (2 columns, 6 rows)
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # Define lead names
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # Plot each middle channel
    for i in range(num_channels):
        sample = numpy_array[0, i, :]
        time = np.linspace(0, 10, sample.shape[0])
        axes[i].plot(time, sample, color='black')
        axes[i].set_title(leads[i], loc='right', fontsize=12)
        axes[i].grid(True, which='both', color='red', linestyle='-', linewidth=0.5)

        # Minor ticks every 0.04 (assuming 250 samples/sec)
        axes[i].xaxis.set_minor_locator(MultipleLocator(0.04))
        axes[i].yaxis.set_minor_locator(MultipleLocator(0.1))

        # Major ticks every 1 second and 0.5 mV
        axes[i].xaxis.set_major_locator(MultipleLocator(1))
        axes[i].yaxis.set_major_locator(MultipleLocator(0.5))

        # Customizing the grid lines to look like ECG paper
        axes[i].grid(which='major', color='red', linestyle='-', linewidth=0.75)
        axes[i].grid(which='minor', color='red', linestyle='-', linewidth=0.25)

        # Set the limits for y-axis
        axes[i].set_ylim([-1, 1])

    # Add labels to the entire figure
    fig.text(0.5, 0.04, 'Time in seconds', ha='center', fontsize=14)
    fig.text(0.04, 0.5, 'Amplitude (mV)', va='center', rotation='vertical', fontsize=14)

    # Adjust layout
    plt.tight_layout()
    
    # Save the plot to the specified file with JPEG format
    plt.savefig(output_file_name, format='jpeg')
    plt.close()
# Example usage:
# numpy_array = np.random.rand(400, 12, 1000)  # Example array
numpy_array = np.load("src/sssd/sssd_label_cond/ch256_T200_betaT0.02/1_samples.npy")
save_ecg_image(numpy_array, 'sample_plot.png')

