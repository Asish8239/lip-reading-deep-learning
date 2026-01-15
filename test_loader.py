from preprocessing.data_loader import load_dataset

X = load_dataset("data/lips/s1")
print("Dataset shape:", X.shape)
