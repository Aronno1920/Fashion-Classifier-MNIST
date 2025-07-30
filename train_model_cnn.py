######## Import required library
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization

# TensorFlow Fashion MNIST dataset
from tensorflow.keras.datasets import fashion_mnist

# Callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Model config class
from model_config import ModelConfig
config = ModelConfig(model_type="cnn")

######## Load data and preprocess
print("--- Building and Training Convolutional Neural Network (CNN) with Callbacks ---")
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize image data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# ✅ One-Hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Reshape for CNN input
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

# Dataset check
print("\n--- Dataset Initial Inspection ---")
print(f"Training images shape -> {x_train.shape}")
print(f"Test images shape -> {x_test.shape}")
print(f"Image shape (single sample) -> {x_train[0].shape}")

######## Build CNN Model
cnn_model = Sequential([
    Input(shape=(28, 28, 1)),

    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # ✅ Correct loss for One-Hot labels
                  metrics=['accuracy'])

print("\nCNN Model Architecture:")
cnn_model.summary()

######## Define Callbacks
early_stopping_cnn = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print("\n--- Model Path: ", config.get_model_path(), " ---")

model_checkpoint_cnn = ModelCheckpoint(
    filepath=config.get_model_path(),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-5,
    verbose=1
)

######## Train CNN Model
print("\nTraining CNN Model with EarlyStopping and ModelCheckpoint...")
cnn_history = cnn_model.fit(
    x_train_cnn, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(x_test_cnn, y_test),
    callbacks=[early_stopping_cnn, model_checkpoint_cnn, lr_scheduler],
    verbose=2
)

print("\nCNN Model training complete.")
cnn_model.load_weights(config.get_model_path())
