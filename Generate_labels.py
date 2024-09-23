import numpy as np
import os

def generate_and_save_labels(wanted_label_index, array_size, output_directory):
    """
    Generates an array of binary labels with 1 at the specified index and 0 elsewhere,
    then saves the array as a .npy file in the specified output directory.

    Parameters:
    wanted_label_index (int): The index of the label that should be set to 1 (0-70).
    array_size (int): The number of samples (rows) in the output array.
    output_directory (str): The directory where the numpy array will be saved.

    Returns:
    str: The file path where the array was saved.
    """
    # Validate the wanted_label_index
    if wanted_label_index < 0 or wanted_label_index >= 71:
        raise ValueError("wanted_label_index must be between 0 and 70")

    # Create the binary label array
    labels_array = np.zeros((array_size, 71), dtype=int)
    labels_array[:, wanted_label_index] = 1

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Define the output file name
    output_file_name = f"label_{wanted_label_index}_array.npy"
    output_file_path = os.path.join(output_directory, output_file_name)

    # Save the array to the specified directory
    np.save(output_file_path, labels_array)

    return output_file_path

# Example usage:
wanted_label_index = 16
array_size = 100  # Number of samples
output_directory = "/home/ido.mahlab/SSSD-ECG/generated_labels"

output_file_path = generate_and_save_labels(wanted_label_index, array_size, output_directory)
print(f"Labels array saved at: {output_file_path}")
