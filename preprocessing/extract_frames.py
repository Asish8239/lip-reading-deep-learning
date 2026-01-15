import os
import cv2

RAW_VIDEO_DIR = "data/raw_videos/s1"
FRAMES_DIR = "data/frames/s1"

os.makedirs(FRAMES_DIR, exist_ok=True)

videos = [v for v in os.listdir(RAW_VIDEO_DIR) if v.endswith(".mpg")]

for video in videos:
    video_name = video.replace(".mpg", "")
    video_path = os.path.join(RAW_VIDEO_DIR, video)
    out_dir = os.path.join(FRAMES_DIR, video_name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(out_dir, f"frame_{count:03d}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1

    cap.release()
    print(f"✔ Extracted {count} frames from {video}")
