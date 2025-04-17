import numpy as np

def convert_features_to_verbal(features, epoch_duration=20, channel_names=None):
    """
    Convert a feature array into natural language descriptions using a standardized template.
    
    Parameters:
      features (np.ndarray): Array with shape (n_epochs, n_channels*6)
      epoch_duration (int): Duration of the epoch in seconds.
      channel_names (list): List of names for each channel. If None, default names are generated.
      
    Returns:
      verbal_representations (list): List of strings, one for each epoch.
    """
    n_channels = features.shape[1] // 6  # 6 features per channel
    if channel_names is None:
        # Generate default channel names if not provided.
        channel_names = [f"Channel {i+1}" for i in range(n_channels)]
    
    verbal_representations = []
    
    # Iterate over each epoch (row) in the feature matrix
    for feature_vector in features:
        description = f"Quantitative EEG: In a {epoch_duration}-second epoch, "
        # Process each channel in the epoch
        for i, ch_name in enumerate(channel_names):
            start_idx = i * 6
            perc_90 = feature_vector[start_idx]
            std_val = feature_vector[start_idx + 1]
            kurt = feature_vector[start_idx + 2]
            alpha_delta = feature_vector[start_idx + 3]
            theta_alpha = feature_vector[start_idx + 4]
            delta_theta = feature_vector[start_idx + 5]
            
            # Append the description for the current channel using the template
            channel_text = (
                f"at channel {ch_name}: the 90th percentile is {perc_90:.2f} µV, "
                f"standard deviation is {std_val:.2f}, kurtosis is {kurt:.2f}, "
                f"alpha:delta ratio is {alpha_delta:.2f}, theta:alpha ratio is {theta_alpha:.2f}, "
                f"and delta:theta ratio is {delta_theta:.2f}; "
            )
            description += channel_text
        verbal_representations.append(description)
    
    return verbal_representations

if __name__ == "__main__":
    # Load the extracted features (assumed to be saved from previous steps)
    features = np.load("D:/EEG/src/data/extracted_features.npy")
    
    # Define channel names based on the channels you selected (update as needed)
    channel_names = ["Fz", "F3", "F4", "Cz"]
    
    # Convert features into natural language descriptions
    verbal_reps = convert_features_to_verbal(features, epoch_duration=20, channel_names=channel_names)
    
    # Print the first verbal representation as an example
    print(verbal_reps[0])
    
    # Optionally, save all verbal representations to a text file for later use
    with open("D:/EEG/src/data/verbal_representations.txt", "w") as f:
        for rep in verbal_reps:
            f.write(rep + "\n")
