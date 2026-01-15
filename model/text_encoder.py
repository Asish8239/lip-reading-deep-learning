import string
import numpy as np

CHARS = string.ascii_lowercase + " "
char_to_idx = {c: i+1 for i, c in enumerate(CHARS)}
idx_to_char = {i+1: c for i, c in enumerate(CHARS)}

def encode_text(text):
    return np.array([char_to_idx[c] for c in text])

def decode_text(indices):
    return "".join([idx_to_char[i] for i in indices if i in idx_to_char])
