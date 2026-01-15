import os
import cv2
import numpy as np

IMG_SIZE = 64
MAX_FRAMES = 75   # fixed sequence length

def load_lip_sequence(folder_path):
    frames = sorted(os.listdir(folder_path))
    sequence = []

    for frame in frames[:MAX_FRAMES]:
        img = cv2.imread(os.path.join(folder_path, frame), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0   # normalize
        sequence.append(img)

    # Padding if frames < MAX_FRAMES
    while len(sequence) < MAX_FRAMES:
        sequence.append(np.zeros((IMG_SIZE, IMG_SIZE)))

    return np.array(sequence)



def load_dataset(lips_root):
    X = []
    video_names = sorted(os.listdir(lips_root))

    for video in video_names:
        seq = load_lip_sequence(os.path.join(lips_root, video))
        X.append(seq)

    return np.array(X), video_names
