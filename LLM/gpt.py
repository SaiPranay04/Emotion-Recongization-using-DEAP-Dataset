import numpy as np

def segment_combined_data(data, labels, epoch_duration=20, sampling_rate=128):
    
    epoch_length = epoch_duration * sampling_rate  # e.g., 20 * 128 = 2560 timepoints
    segmented_data = []
    segmented_labels = []
    
    # Iterate over each original segment in the combined data
    for seg, lab in zip(data, labels):
        n_timepoints = seg.shape[1]
        n_epochs = n_timepoints // epoch_length  # How many full epochs can be extracted
        for i in range(n_epochs):
            start = i * epoch_length
            end = start + epoch_length
            epoch_data = seg[:, start:end]  # Shape: (channels, epoch_length)
            segmented_data.append(epoch_data)
            segmented_labels.append(lab)  # Inherit the label from the parent segment

    segmented_data = np.array(segmented_data)
    segmented_labels = np.array(segmented_labels)
    
    return segmented_data, segmented_labels

if __name__ == "__main__":
    # Load the combined data and labels from previously saved .npy files
    combined_data = np.load("D:/EEG/src/data/combined_data.npy")
    combined_labels = np.load("D:/EEG/src/data/combined_labels.npy")
    
    segmented_data, segmented_labels = segment_combined_data(
        combined_data, combined_labels, epoch_duration=20, sampling_rate=128
    )
    
    print(f"Segmented data shape: {segmented_data.shape}")    # Expected: (total_epochs, channels, 2560)
    print(f"Segmented labels shape: {segmented_labels.shape}")  # Expected: (total_epochs, label_dim)
    
    # Save the segmented data and labels with new names
    np.save("D:/EEG/src/data/segmented_data.npy", segmented_data)
    np.save("D:/EEG/src/data/segmented_labels.npy", segmented_labels)
