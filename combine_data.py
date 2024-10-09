import numpy as np

samples_data_files_to_combine = [
     #"/home/ido.mahlab/SSSD-ECG/sssd_label_cond/ch256_T200_betaT0.02/0_samples.npy",
     #"/home/ido.mahlab/SSSD-ECG/sssd_label_cond/ch256_T200_betaT0.02/1_samples.npy",
     #'/home/ido.mahlab/SSSD-ECG/sssd_label_cond/ch256_T200_betaT0.02/2_samples.npy',
     #'/home/ido.mahlab/SSSD-ECG/sssd_label_cond/ch256_T200_betaT0.02/3_samples.npy',
     #'/home/ido.mahlab/SSSD-ECG/sssd_label_cond/ch256_T200_betaT0.02/4_samples.npy',
     #'/home/ido.mahlab/SSSD-ECG/sssd_label_cond/ch256_T200_betaT0.02/5_samples.npy',
     #'/home/ido.mahlab/SSSD-ECG/sssd_label_cond/ch256_T200_betaT0.02/6_samples.npy',
     #'/home/ido.mahlab/SSSD-ECG/sssd_label_cond/ch256_T200_betaT0.02/7_samples.npy',
    # '/home/ido.mahlab/SSSD-ECG/sssd_label_cond/ch256_T200_betaT0.02/8_samples.npy',
    '/home/ido.mahlab/SSSD-ECG/data/ptbxl_validation_data.npy',
    "/home/ido.mahlab/SSSD-ECG/data/ptbxl_train_data.npy"
]

labels_data_files_to_combine = [
     #"/home/ido.mahlab/SSSD-ECG/sssd_label_cond/ch256_T200_betaT0.02/0_labels.npy",
     #"/home/ido.mahlab/SSSD-ECG/sssd_label_cond/ch256_T200_betaT0.02/1_labels.npy",
     #'/home/ido.mahlab/SSSD-ECG/sssd_label_cond/ch256_T200_betaT0.02/2_labels.npy',
     #'/home/ido.mahlab/SSSD-ECG/sssd_label_cond/ch256_T200_betaT0.02/3_labels.npy',
     #'/home/ido.mahlab/SSSD-ECG/sssd_label_cond/ch256_T200_betaT0.02/4_labels.npy',
     #'/home/ido.mahlab/SSSD-ECG/sssd_label_cond/ch256_T200_betaT0.02/5_labels.npy',
     #'/home/ido.mahlab/SSSD-ECG/sssd_label_cond/ch256_T200_betaT0.02/6_labels.npy',
     #'/home/ido.mahlab/SSSD-ECG/sssd_label_cond/ch256_T200_betaT0.02/7_labels.npy',
    # '/home/ido.mahlab/SSSD-ECG/sssd_label_cond/ch256_T200_betaT0.02/8_labels.npy',
    '/home/ido.mahlab/SSSD-ECG/labels/ptbxl_validation_labels.npy',
    "/home/ido.mahlab/SSSD-ECG/labels/ptbxl_train_labels.npy"
]

combined_train_data_path = "/home/ido.mahlab/SSSD-ECG/"

def load_numpy_file(file_path: str) -> np.ndarray:
    return np.load(file_path)

def save_numpy_file(file_path: str, file_name: str, array: np.ndarray) -> None:
    file_path_name = file_path + file_name
    np.save(file_path_name, array)
    print(f"Saved array of size {array.shape}")

def combined_arrays(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.concatenate([x, y], axis=0)

def shuffle_data(samples: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray):
    rng = np.random.default_rng(seed=10)  # Ensure reproducibility with a random seed
    indices = rng.permutation(len(samples))
    return samples[indices], labels[indices]

if __name__ == "__main__":
    # Load initial data
    all_samples = load_numpy_file(samples_data_files_to_combine[0])
    all_labels = load_numpy_file(labels_data_files_to_combine[0])
    
    # Combine additional data
    for samples_file, labels_file in zip(samples_data_files_to_combine[1:], labels_data_files_to_combine[1:]):
        samples = load_numpy_file(samples_file)
        labels = load_numpy_file(labels_file)
        all_samples = combined_arrays(all_samples, samples)
        all_labels = combined_arrays(all_labels, labels)
    
    # Shuffle the combined data
    all_samples, all_labels = shuffle_data(all_samples, all_labels)
    
    # Save the shuffled and combined data
    save_numpy_file(combined_train_data_path, "combined_samples.npy", all_samples)
    save_numpy_file(combined_train_data_path, "combined_labels.npy", all_labels)
