import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GaussianNoise
from tensorflow.keras.optimizers import Adam
# from scipy.signal import resample

all_data = np.load("D:/EEG/src/data/combined_data.npy")  # Shape: (1240, 40, 8064)
all_labels = np.load("D:/EEG/src/data/combined_labels.npy")  # Shape: (1240, 4)

#Arousal
arousal = all_labels[:, 1]
y = (arousal > 5).astype(np.int32)

# Preprocess EEG data:
# Transpose each trial from (channels, time) to (time, channels) for Conv1D input.
X = np.transpose(all_data, (0, 2, 1))  # new shape: (1280, 8064, 40)

# Normalize each trial (z-score normalization)
X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

# (Optional) Downsample along the time axis to reduce sequence length (e.g., by factor of 2)
X = X[:, ::2, :]  # new time dimension: 8064/2 = 4032

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model (using 1D convolutions over time)
model = Sequential()

# Add GaussianNoise layer to inject noise during training (only active during training)
model.add(GaussianNoise(0.1, input_shape=X_train.shape[1:]))  # 0.1 is the noise std deviation

# First Conv1D layer: reduce time dimension with a kernel of size 128 and stride 2
model.add(Conv1D(filters=64, kernel_size=128, strides=2, activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling1D(pool_size=4))
# Second Conv1D layer
model.add(Conv1D(filters=128, kernel_size=64, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
# Flatten and add fully connected layers
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
# Final layer for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")






