import os

path = "data/lips/s1"

print("Does s1 exist?", os.path.exists(path))
print("Absolute path:", os.path.abspath(path))

if os.path.exists(path):
    print("Folders inside s1:")
    for f in os.listdir(path):
        print(" -", f)
