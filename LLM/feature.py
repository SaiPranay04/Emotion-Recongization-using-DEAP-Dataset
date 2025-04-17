import numpy as np
from scipy.stats import kurtosis

def compute_power_bands(signal, sampling_rate):
    """
    Compute the power in delta, theta, and alpha bands for a given 1D signal.
    
    Parameters:
      signal (np.ndarray): 1D array of EEG timepoints.
      sampling_rate (int): Sampling rate in Hz.
      
    Returns:
      delta_power, theta_power, alpha_power (floats): Summed power in each frequency band.
    """
    # Compute the FFT and its power spectrum
    fft_vals = np.fft.rfft(signal)
    fft_power = np.abs(fft_vals)**2
    freqs = np.fft.rfftfreq(signal.shape[0], d=1/sampling_rate)
    
    # Define frequency bands (you can adjust the band limits as needed)
    delta_idx = np.where((freqs >= 1) & (freqs < 4))[0]
    theta_idx = np.where((freqs >= 4) & (freqs < 8))[0]
    alpha_idx = np.where((freqs >= 8) & (freqs < 13))[0]
    
    delta_power = np.sum(fft_power[delta_idx])
    theta_power = np.sum(fft_power[theta_idx])
    alpha_power = np.sum(fft_power[alpha_idx])
    
    return delta_power, theta_power, alpha_power

def extract_features(segmented_data, sampling_rate=128, selected_channels=None):
    """
    Extract features for each epoch and each selected channel.
    
    For each channel, compute:
      - 90th percentile
      - Standard deviation
      - Kurtosis
      - Alpha:Delta, Theta:Alpha, and Delta:Theta power ratios
    
    Parameters:
      segmented_data (np.ndarray): Array of shape (n_epochs, n_channels, n_timepoints)
      sampling_rate (int): Sampling rate in Hz.
      selected_channels (list): List of channel indices to process. If None, all channels are used.
      
    Returns:
      features (np.ndarray): Array of shape (n_epochs, n_features) where n_features = len(selected_channels) * 6.
    """
    if selected_channels is None:
        selected_channels = np.arange(segmented_data.shape[1])
    
    all_features = []
    for epoch in segmented_data:
        epoch_features = []
        for ch in selected_channels:
            signal = epoch[ch, :]  # Extract the 1D time series for this channel
            
            # Compute basic statistical features
            perc_90 = np.percentile(signal, 90)
            std_val = np.std(signal)
            kurt = kurtosis(signal)
            
            # Compute spectral power features for the defined bands
            delta_power, theta_power, alpha_power = compute_power_bands(signal, sampling_rate)
            
            # Calculate power ratios while avoiding division by zero
            alpha_delta_ratio = alpha_power / delta_power if delta_power != 0 else 0
            theta_alpha_ratio = theta_power / alpha_power if alpha_power != 0 else 0
            delta_theta_ratio = delta_power / theta_power if theta_power != 0 else 0
            
            # Append the 6 features for this channel
            ch_features = [perc_90, std_val, kurt, alpha_delta_ratio, theta_alpha_ratio, delta_theta_ratio]
            epoch_features.extend(ch_features)
        all_features.append(epoch_features)
    
    features = np.array(all_features)
    return features

if __name__ == "__main__":
    # Load the segmented data from the file produced in the segmentation step
    segmented_data = np.load("D:/EEG/src/data/segmented_data.npy")
    
    # Define the channels you want to extract features from.
    # For example, if frontal channels are most informative for emotion,
    # specify their indices (adjust these indices based on your data).
    selected_channels = [0, 1, 2, 3]  # Example: change as needed
    
    # Extract features
    features = extract_features(segmented_data, sampling_rate=128, selected_channels=selected_channels)
    
    print("Extracted features shape:", features.shape)
    # Expected shape: (n_epochs, len(selected_channels) * 6)
    
    # Save the extracted features for later use
    np.save("D:/EEG/src/data/extracted_features.npy", features)
