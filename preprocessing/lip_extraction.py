import os
import cv2

# --------------------------------------------------
# PATHS
# --------------------------------------------------
FRAMES_ROOT = "data/frames/s1"
LIPS_ROOT = "data/lips/s1"
os.makedirs(LIPS_ROOT, exist_ok=True)

# --------------------------------------------------
# LOAD HAAR CASCADES
# --------------------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

mouth_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

# --------------------------------------------------
# PROCESS ALL VIDEOS
# --------------------------------------------------
videos = sorted(os.listdir(FRAMES_ROOT))

for video in videos:
    print(f"✔ Processing video: {video}")

    frames_dir = os.path.join(FRAMES_ROOT, video)
    lips_dir = os.path.join(LIPS_ROOT, video)
    os.makedirs(lips_dir, exist_ok=True)

    frames = sorted(os.listdir(frames_dir))
    saved = 0

    for f in frames:
        img_path = os.path.join(frames_dir, f)
        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            continue

        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]

        mouths = mouth_cascade.detectMultiScale(face_roi, 1.5, 7)
        if len(mouths) == 0:
            continue

        mx, my, mw, mh = mouths[0]

        # Mouth region (lower half bias)
        lip = face_roi[my+mh//4 : my+mh, mx : mx+mw]

        if lip.size == 0:
            continue

        lip = cv2.resize(lip, (64, 64))
        cv2.imwrite(os.path.join(lips_dir, f), lip)
        saved += 1

    print(f"✅ Saved {saved} lip frames for {video}")

print("🎉 LIP EXTRACTION COMPLETED FOR ALL VIDEOS")
