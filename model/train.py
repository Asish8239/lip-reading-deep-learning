import os
import sys
import numpy as np
import tensorflow as tf

# -------------------------------------------------
# Add project root to Python path
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from model.cnn_lstm_model import build_model
from preprocessing.data_loader import load_lip_sequence
from preprocessing.label_parser import read_align_file
from model.text_encoder import encode_text

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
LIPS_ROOT = "data/lips/s1"
LABELS_ROOT = "data/labels/s1"
EPOCHS = 3            # 👈 keep small (enough for demo)
BATCH_SIZE = 2
IMG_SIZE = 64
MAX_FRAMES = 75

# -------------------------------------------------
# LOAD LIP DATA
# -------------------------------------------------
print("🔹 Loading lip sequences...")

video_names = sorted(os.listdir(LIPS_ROOT))
X = []

for video in video_names:
    seq = load_lip_sequence(os.path.join(LIPS_ROOT, video))
    X.append(seq)

X = np.array(X)
X = X[..., np.newaxis]   # (batch, time, h, w, 1)

print("✔ X shape:", X.shape)

# -------------------------------------------------
# LOAD & ENCODE LABELS
# -------------------------------------------------
print("🔹 Loading labels...")

labels = []
label_lengths = []

for video in video_names:
    align_path = os.path.join(LABELS_ROOT, f"{video}.align")
    text = read_align_file(align_path)
    encoded = encode_text(text)

    labels.append(encoded)
    label_lengths.append(len(encoded))

max_label_len = max(label_lengths)
y = np.zeros((len(labels), max_label_len), dtype=np.int32)

for i, seq in enumerate(labels):
    y[i, :len(seq)] = seq

print("✔ y shape:", y.shape)

# -------------------------------------------------
# BUILD MODEL
# -------------------------------------------------
model = build_model()
model.summary()

# -------------------------------------------------
# CTC LOSS FUNCTION
# -------------------------------------------------
def ctc_loss(y_true, y_pred):
    batch_size = tf.shape(y_pred)[0]

    input_length = tf.ones((batch_size, 1), dtype="int32") * tf.shape(y_pred)[1]
    label_length = tf.reduce_sum(
        tf.cast(tf.not_equal(y_true, 0), tf.int32),
        axis=1
    )
    label_length = tf.expand_dims(label_length, 1)

    return tf.keras.backend.ctc_batch_cost(
        y_true, y_pred, input_length, label_length
    )

model.compile(
    optimizer="adam",
    loss=ctc_loss
)

# -------------------------------------------------
# TRAIN
# -------------------------------------------------
print("🚀 Starting training...")
model.fit(
    X,
    y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# -------------------------------------------------
# SAVE MODEL (ONCE, CLEANLY)
# -------------------------------------------------
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/lip_reader.h5")
print("✅ Model saved at saved_model/lip_reader.h5")
