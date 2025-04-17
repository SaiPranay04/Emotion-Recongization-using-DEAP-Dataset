import os
import pickle
import numpy as np

def load_eeg_data(dataset_path):
    all_data = []
    all_labels = []
    skipped_files = []

    for file_name in os.listdir(dataset_path):
        if file_name.endswith('.dat'):
            file_path = os.path.join(dataset_path, file_name)
            try:
                with open(file_path, 'rb') as file:
                    participant_data = pickle.load(file, encoding='latin1')

                data = participant_data.get('data')  # Shape: (40, 40, 8064)
                labels = participant_data.get('labels')  # Shape: (40, 4)

                if data is not None and labels is not None:
                    all_data.append(data)
                    all_labels.append(labels)
                else:
                    skipped_files.append(file_name)
            except Exception as e:
                skipped_files.append(file_name)

    if all_data and all_labels:
        all_data = np.vstack(all_data)
        all_labels = np.vstack(all_labels)
    else:
        raise ValueError("No valid data/labels found in the dataset.")

    return all_data, all_labels, skipped_files

if __name__ == "__main__":
    dataset_path = "D:/EEG/src/data/data_preprocessed_python"

    data, labels, skipped_files = load_eeg_data(dataset_path)

    print(f"Data shape: {data.shape}")  # Should be (32 * 40, 40, 8064)
    print(f"Labels shape: {labels.shape}")  # Should be (32 * 40, 4)

    if skipped_files:
        print(f"Skipped files: {skipped_files}")

    np.save("D:/EEG/src/data/combined_data.npy", data)
    np.save("D:/EEG/src/data/combined_labels.npy", labels)


