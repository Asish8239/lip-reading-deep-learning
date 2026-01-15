import os

video_dir = "data/raw_videos/s1"
label_dir = "data/labels/s1"

print("Videos:", len(os.listdir(video_dir)))
print("Labels:", len(os.listdir(label_dir)))

