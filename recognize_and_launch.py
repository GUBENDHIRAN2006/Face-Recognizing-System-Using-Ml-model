import cv2
import json
import os
import sys
import time
import subprocess
from pathlib import Path
import argparse
import webbrowser  # ✅ added

"""
Usage:
python recognize_and_launch.py --target "YourName" --open "C:/path/to/file_or_video.mp4"
python recognize_and_launch.py --open "https://github.com/GUBENDHIRAN2006?tab=repositories"
"""

def open_path(path):
    path = str(path).strip()

    # ✅ If it's a URL, open in browser
    if path.startswith("http://") or path.startswith("https://"):
        webbrowser.open(path)
        return

    # ✅ Otherwise, treat as file/app path
    path = str(Path(path).expanduser().resolve())
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.call(["open", path])
    else:
        subprocess.call(["xdg-open", path])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=r"G:\face\models\lbph_face_model.xml")
    parser.add_argument("--labels", default=r"G:\face\models\labels.json")

    parser.add_argument("--target", default="Gubendhiran")
    parser.add_argument("--open", dest="open_target",
                        default=r"https://github.com/GUBENDHIRAN2006?tab=repositories")

    parser.add_argument("--threshold", type=float, default=60.0, help="LBPH confidence threshold (lower is better)")
    parser.add_argument("--cooldown", type=float, default=10.0, help="Seconds to wait before allowing another launch")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}. Train it with train_lbph.py.")
    if not Path(args.labels).exists():
        raise FileNotFoundError(f"Labels not found: {args.labels}. Train it with train_lbph.py.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(args.model)

    with open(args.labels, "r") as f:
        label_map = json.load(f)
    name_to_id = {v: int(k) for k, v in label_map.items()}

    if args.target not in name_to_id:
        raise ValueError(f"Target '{args.target}' not found in labels: {list(name_to_id.keys())}")

    target_id = name_to_id[args.target]

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # ✅ Using IP camera (your phone as webcam)
    cap = cv2.VideoCapture("http://192.168.29.238:8080/video")
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    last_launch = 0.0
    stable_hits = 0
    REQUIRED_STABLE = 8  # consecutive frames required

    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        recognized_now = False
        best_conf = 9999.0
        for (x, y, w, h) in faces:
            x0 = max(0, x - 10); y0 = max(0, y - 10)
            x1 = min(gray.shape[1], x + w + 10); y1 = min(gray.shape[0], y + h + 10)
            roi = gray[y0:y1, x0:x1]
            roi = cv2.resize(roi, (200, 200))

            label_id, confidence = recognizer.predict(roi)
            name = label_map.get(str(label_id), "Unknown")

            color = (0, 255, 0) if confidence < args.threshold else (0, 0, 255)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            cv2.putText(frame, f"{name} ({confidence:.1f})", (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if label_id == target_id and confidence < args.threshold:
                recognized_now = True
                best_conf = min(best_conf, confidence)

        if recognized_now:
            stable_hits += 1
        else:
            stable_hits = 0

        now = time.time()
        if stable_hits >= REQUIRED_STABLE and (now - last_launch) > args.cooldown:
            print(f"[ACTION] Recognized {args.target} with confidence {best_conf:.1f}. Opening: {args.open_target}")
            open_path(args.open_target)
            last_launch = now
            stable_hits = 0

        cv2.imshow("Recognize & Launch (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
