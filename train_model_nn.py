######## Import required library
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization

# tensorflow fashion dataset
from tensorflow.keras.datasets import fashion_mnist

# Best epochs number using EarlyStopping & ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
####################
from model_config import ModelConfig


######## Load data
config = ModelConfig(model_type="nn")

print("--- Building and Training Neural Network (NN) with Callbacks ---")
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train_nn = x_train.reshape(-1, 28*28)
x_test_nn = x_test.reshape(-1, 28*28)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


print("\n--- Dataset Initial Inspection ---")
print(f"Training images shape -> {x_train.shape}")
print(f"Test images shape -> {x_test.shape}")
print(f"Image shape (single sample) -> {x_train[0].shape}")

nn_model = Sequential([
    Input(shape=(784,)),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(10, activation='softmax')
])

nn_model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

print("\n--- Neural Network Model Architecture ---")
nn_model.summary()

# --- Start: Callbacks ---
early_stopping_nn = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print("\n--- Model Path: ",config.get_model_path()," ---")

model_checkpoint_nn = ModelCheckpoint(
    filepath=config.get_model_path(),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-5,
    verbose=1)
# --- End: Callbacks ---

print("\n--- Training Neural Network Model with EarlyStopping and ModelCheckpoint ---")
nn_history = nn_model.fit(
    x_train_nn, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping_nn, model_checkpoint_nn, lr_scheduler],
    verbose=2
)

print("--- Neural Network Model training complete. ---")
nn_model.load_weights(config.get_model_path())
####################