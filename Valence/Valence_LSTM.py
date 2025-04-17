import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GaussianNoise
from tensorflow.keras.optimizers import Adam

# Load the data
all_data = np.load("D:/EEG/src/data/combined_data.npy")   # Shape: (1240, 40, 8064)
all_labels = np.load("D:/EEG/src/data/combined_labels.npy") # Shape: (1240, 4)

# Binary classification on the valence rating.
# Ratings are on a 1-9 scale; threshold at 5: >5 is high valence.
valence = all_labels[:, 0]
y = (valence > 5).astype(np.int32)

# Preprocess EEG data:
# 1. Transpose each trial from (channels, time) to (time, channels) for LSTM input.
X = np.transpose(all_data, (0, 2, 1))  # New shape: (1240, 8064, 40)

# 2. Normalize each trial (z‑score normalization)
X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

# 3. Downsample the time axis by a factor of 32 to reduce sequence length
X = X[:, ::32, :]  # New shape: (1240, ~252, 40)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model with noise injection
model = Sequential()

# Add GaussianNoise layer to inject noise during training (only active during training)
model.add(GaussianNoise(0.1, input_shape=X_train.shape[1:]))  # 0.1 is the noise std deviation

# First LSTM layer with 64 units. return_sequences=True allows stacking another LSTM.
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.3))

# Second LSTM layer with 64 units
model.add(LSTM(64))
model.add(Dropout(0.3))

# Fully connected layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Final layer for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model using binary crossentropy loss and the Adam optimizer.
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train,
                    epochs=20,
                    batch_size=16,
                    validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
