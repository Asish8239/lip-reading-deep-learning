import os
import sys
import numpy as np
import tensorflow as tf
import cv2

# -------------------------------------------------
# Add project root to Python path
# -------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.text_encoder import decode_text

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_PATH = "model/lip_reading_model.h5"
LIPS_ROOT = "data/lips/s1"
IMG_SIZE = 64
MAX_FRAMES = 75

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# -------------------------------------------------
# LOAD LIP SEQUENCE
# -------------------------------------------------
def load_lip_sequence(folder_path):
    frames = sorted(os.listdir(folder_path))
    sequence = []

    for frame in frames[:MAX_FRAMES]:
        img = cv2.imread(os.path.join(folder_path, frame), cv2.IMREAD_GRAYSCALE)
        img = img.astype("float32") / 255.0
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        sequence.append(img)

    while len(sequence) < MAX_FRAMES:
        sequence.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype="float32"))

    X = np.array(sequence)
    X = X[np.newaxis, ..., np.newaxis]
    return X

# -------------------------------------------------
# CTC GREEDY DECODER
# -------------------------------------------------
def ctc_decode(pred):
    pred = np.argmax(pred, axis=-1)[0]

    decoded = []
    prev = -1
    for p in pred:
        if p != prev and p != 0:
            decoded.append(p)
        prev = p

    return decode_text(decoded)

# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    # ▶ Get video name from command line
    if len(sys.argv) < 2:
        print("❌ Please provide a video name")
        print("Usage: python model/predict.py <video_name>")
        print("Example: python model/predict.py bbaf2n")
        sys.exit(1)

    video_name = sys.argv[1]
    lip_folder = os.path.join(LIPS_ROOT, video_name)

    if not os.path.exists(lip_folder):
        print(f"❌ Lip folder not found: {lip_folder}")
        sys.exit(1)

    print("🔹 Predicting for video:", video_name)

    X = load_lip_sequence(lip_folder)
    prediction = model.predict(X)

    text = ctc_decode(prediction)
    print("✅ Predicted Text:", text)
