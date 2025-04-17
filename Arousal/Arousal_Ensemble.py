import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D,
                                     LSTM, Dense, Dropout, BatchNormalization, GaussianNoise,
                                     concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# -------------------------------
# 1. Load Data and Labels
# -------------------------------
all_data = np.load("D:/EEG/src/data/combined_data.npy")    # Shape: (1240, 40, 8064)
all_labels = np.load("D:/EEG/src/data/combined_labels.npy")  # Shape: (1240, 4)

#arousal
arousal = all_labels[:, 1]
y = (arousal > 5).astype(np.int32)

# -------------------------------
# 2. Preprocess EEG Data
# -------------------------------
# Transpose from (channels, time) to (time, channels)
X = np.transpose(all_data, (0, 2, 1))  # New shape: (1240, 8064, 40)

# Z-score normalization per trial
X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

# Downsample the time axis by a factor of 32 (~252 timesteps)
X = X[:, ::32, :]  # New shape: (1240, ~252, 40)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 3. Build the Ensemble Model (CNN + LSTM)
# -------------------------------
input_shape = X_train.shape[1:]  # (252, 40)
inputs = Input(shape=input_shape)

# --- CNN Branch ---
cnn = GaussianNoise(0.05)(inputs)
cnn = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same',
             kernel_regularizer=l2(1e-4))(cnn)
cnn = MaxPooling1D(pool_size=2)(cnn)
cnn = BatchNormalization()(cnn)
cnn = Dropout(0.3)(cnn)
cnn = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',
             kernel_regularizer=l2(1e-4))(cnn)
cnn = MaxPooling1D(pool_size=2)(cnn)
cnn = BatchNormalization()(cnn)
cnn = Dropout(0.3)(cnn)
cnn = GlobalAveragePooling1D()(cnn)
cnn_branch = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(cnn)
cnn_branch = Dropout(0.3)(cnn_branch)

# --- LSTM Branch ---
lstm = LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4))(inputs)
lstm = Dropout(0.3)(lstm)
lstm = LSTM(64, kernel_regularizer=l2(1e-4))(lstm)
lstm = Dropout(0.3)(lstm)
lstm_branch = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(lstm)
lstm_branch = Dropout(0.3)(lstm_branch)

# --- Ensemble ---
combined = concatenate([cnn_branch, lstm_branch])
combined = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(combined)
combined = BatchNormalization()(combined)
combined = Dropout(0.4)(combined)
outputs = Dense(1, activation='sigmoid')(combined)

model = Model(inputs=inputs, outputs=outputs)

# -------------------------------
# 4. Compile and Train
# -------------------------------
# Use a lower learning rate for finer updates
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=16,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks)

# -------------------------------
# 5. Evaluate the Model
# -------------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
