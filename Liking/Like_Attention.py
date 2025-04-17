import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

all_data = np.load("D:/EEG/src/data/combined_data.npy")  # Shape: (1240, 40, 8064)
all_labels = np.load("D:/EEG/src/data/combined_labels.npy")  # Shape: (1240, 4)

# --- Custom Attention Layer ---
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # input_shape: (batch_size, timesteps, features)
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="random_normal",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True
        )
        self.u = self.add_weight(
            name="att_u",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True
        )
        super(Attention, self).build(input_shape)
    
    def call(self, x):
        # x shape: (batch_size, timesteps, features)
        # 1) Apply tanh activation: (batch_size, timesteps, features)
        u_it = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        # 2) Compute attention scores: (batch_size, timesteps, 1)
        a_it = tf.nn.softmax(tf.tensordot(u_it, self.u, axes=1), axis=1)
        # 3) Weighted sum of x by the attention scores
        weighted_input = x * a_it
        output = tf.reduce_sum(weighted_input, axis=1)
        return output


#Liking
liking = all_labels[:, 3]
y = (liking > 5).astype(np.int32)

# --- Preprocess EEG Data ---
# Transpose from (trials, channels, time) -> (trials, time, channels)
X = np.transpose(all_data, (0, 2, 1))  # shape: (1280, 8064, 40)

# Z-score normalization (per trial)
X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

# Downsample by factor of 32 (optional)
X = X[:, ::32, :]  # shape: (1280, ~252, 40)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Define Callbacks ---
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,              # Stop after 5 epochs with no improvement
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,              # Reduce learning rate by factor of 10
    patience=3,              # After 3 epochs of no improvement
    min_lr=1e-7
)

# --- Define the LSTM Model with Attention + L2 regularization ---
model = Sequential()

# Add GaussianNoise layer to inject noise during training (only active during training)
model.add(GaussianNoise(0.1, input_shape=X_train.shape[1:]))  # 0.1 is the noise std deviation


# First LSTM layer (64 units) with L2 regularization; returns sequences for attention
model.add(
    LSTM(
        64,
        return_sequences=True,
        kernel_regularizer=l2(1e-5),
        bias_regularizer=l2(1e-5),
        input_shape=X_train.shape[1:]
    )
)
model.add(Dropout(0.3))

# Second LSTM layer (32 units) with L2 regularization; also returns sequences
model.add(
    LSTM(
        32,
        return_sequences=True,
        kernel_regularizer=l2(1e-5),
        bias_regularizer=l2(1e-5)
    )
)
model.add(Dropout(0.3))

# Attention layer
model.add(Attention())

# Fully connected layer with L2 regularization
model.add(Dense(64, activation='relu', kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5)))
model.add(Dropout(0.5))

# Final output for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Train the Model ---
history = model.fit(
    X_train, y_train,
    epochs=50,             # Increased epochs for potential better convergence
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr]
)

# --- Evaluate the Model ---
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")




