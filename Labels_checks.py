import numpy as np

def load_and_print_numpy_file(file_path):
    """
    Loads a NumPy array from a .npy file and prints its contents.

    Parameters:
    file_path (str): The path to the .npy file.

    Returns:
    numpy.ndarray: The loaded NumPy array.
    """
    try:
        # Load the NumPy array from the file
        data = np.load(file_path)
        
        # Print the shape of the array
        print("Data shape:", data.shape)
        
        # Print the contents of the array
        print("Data contents:")
        print(data[211])
        
        return data
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

# Example usage:
file_path = '/home/ido.mahlab/SSSD-ECG/sssd_label_cond/ch256_T200_betaT0.02/0_labels.npy'  # Replace with your file path
data = load_and_print_numpy_file(file_path)

if data is not None:
    print("File loaded and printed successfully!")
else:
    print("Failed to load the file.")