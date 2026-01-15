import tensorflow as tf
from tensorflow.keras import layers, models

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
IMG_SIZE = 64
MAX_FRAMES = 75

# 26 letters + space + CTC blank (index 0)
NUM_CLASSES = 29

# -------------------------------------------------
# CNN FEATURE EXTRACTOR
# -------------------------------------------------
def build_cnn():
    cnn = models.Sequential(name="Lip_CNN")

    # Block 1
    cnn.add(layers.Conv2D(64, (3, 3), padding="same"))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.Activation("relu"))
    cnn.add(layers.MaxPooling2D((2, 2)))

    # Block 2
    cnn.add(layers.Conv2D(128, (3, 3), padding="same"))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.Activation("relu"))
    cnn.add(layers.MaxPooling2D((2, 2)))

    # Block 3
    cnn.add(layers.Conv2D(256, (3, 3), padding="same"))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.Activation("relu"))
    cnn.add(layers.MaxPooling2D((2, 2)))

    cnn.add(layers.Flatten())

    return cnn

# -------------------------------------------------
# CNN + LSTM MODEL
# -------------------------------------------------
def build_model():
    cnn = build_cnn()

    model = models.Sequential(name="LipReading_CNN_LSTM")

    # Apply CNN on each frame
    model.add(
        layers.TimeDistributed(
            cnn,
            input_shape=(MAX_FRAMES, IMG_SIZE, IMG_SIZE, 1)
        )
    )

    # Temporal modeling (STACKED LSTM)
    model.add(layers.LSTM(256, return_sequences=True))
    model.add(layers.LSTM(128, return_sequences=True))

    # Regularization
    model.add(layers.Dropout(0.3))

    # Character probabilities
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

    return model
