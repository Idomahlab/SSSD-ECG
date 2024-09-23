import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import ecg_plot

def save_ecg_image(numpy_array, output_file_name):
    """
    Saves an ECG-like image showing all 12 channels from a NumPy array.
    
    Parameters:
    - numpy_array: NumPy array of shape [12, length] representing the ECG data.
    - output_file_name: Name of the file to save the ECG-like image (e.g., 'ecg_image.png')
    """
    
    # Ensure that numpy_array is of shape [12, length]
    if numpy_array.shape[0] != 12:
        raise ValueError("Input array must have 12 channels.")

    # Plot and save the ECG data
    ecg_plot.plot_12(numpy_array, sample_rate=100, title='ECG 12 Leads (Synthetic Diffusion generated sample)')
    ecg_plot.save_as_png(output_file_name, '.')

def get_sample_indices_from_file(file_path, label_index):
    """
    Loads a numpy array from a .npy file and returns the indices of samples 
    that are labeled with a '1' at the specified label index.

    Parameters:
    file_path (str): The path to the .npy file containing the binary labels array.
    label_index (int): The index of the label (0-70).

    Returns:
    np.ndarray: An array of indices of the samples that have a '1' at the specified label index.
    """
    # Load the numpy array from the file
    labels_array = np.load(file_path)
    
    # Check if the label_index is within the valid range
    if label_index < 0 or label_index >= labels_array.shape[1]:
        raise ValueError(f"label_index must be between 0 and {labels_array.shape[1]-1}")

    # Get the indices of samples where the specified label index is 1
    indices = np.where(labels_array[:, label_index] == 1)[0]
    
    return indices

# Example usage:
labels_file_path = "/home/ido.mahlab/SSSD-ECG/sssd_label_cond/ch256_T200_betaT0.02/0_labels.npy"
ecg_data_file_path = "/home/ido.mahlab/SSSD-ECG/sssd_label_cond/ch256_T200_betaT0.02/0_samples.npy"

# Get the indices of samples with the specified label
sample_indices = get_sample_indices_from_file(labels_file_path, 4)

# Choose one index from the returned indices (e.g., the first one)
chosen_index = sample_indices[15]  # You can choose a different index if needed

# Load the ECG data
ECG_data = np.load(ecg_data_file_path)

# Plot and save the ECG image for the chosen sample
save_ecg_image(ECG_data[chosen_index], 'sample_plot.png')
