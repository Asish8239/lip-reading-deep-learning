def read_align_file(path):
    words = []

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and parts[2] != "sil":
                words.append(parts[2])

    return " ".join(words)
