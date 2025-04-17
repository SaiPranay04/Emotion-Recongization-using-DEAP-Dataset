import numpy as np
import scipy.signal as signal
from scipy.integrate import trapezoid
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# -------------------------------
# 1. Load Data and Labels
# -------------------------------
all_data = np.load("D:/EEG/src/data/combined_data.npy")    # Shape: (1240, 40, 8064)
all_labels = np.load("D:/EEG/src/data/combined_labels.npy")  # Shape: (1240, 4)

# Liking
liking = all_labels[:, 3]
y = (liking > 5).astype(np.int32)

# -------------------------------
# 2. Feature Extraction: Frequency Band Power
# -------------------------------
fs = 128  # Sampling frequency (data already downsampled to 128Hz)
bands = [(4, 8), (8, 13), (13, 30), (30, 45)]  # Theta, Alpha, Beta, Gamma

n_trials, n_channels, n_samples = all_data.shape
features = []

# Compute band power for each trial and each channel
for trial in range(n_trials):
    trial_features = []
    for ch in range(n_channels):
        # Extract the signal for one channel in the current trial.
        data = all_data[trial, ch, :]  # shape: (8064,)
        # Compute power spectral density using Welch's method.
        freqs, psd = signal.welch(data, fs=fs, nperseg=256)
        # For each frequency band, compute the area under the PSD curve using trapezoid integration.
        for band in bands:
            idx_band = (freqs >= band[0]) & (freqs <= band[1])
            band_power = trapezoid(psd[idx_band], freqs[idx_band])
            trial_features.append(band_power)
    features.append(trial_features)

features = np.array(features)  # Expected shape: (1240, 160) -> 40 channels x 4 bands

# -------------------------------
# 3. Standardize and Split Data
# -------------------------------
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, y, test_size=0.2, random_state=42)

# -------------------------------
# 4. SVM Hyperparameter Tuning with GridSearchCV
# -------------------------------
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

svm = SVC()
grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=1)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)

best_svm = grid.best_estimator_
y_pred = best_svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy*100:.2f}%")
