import cv2
import os
from pathlib import Path
import argparse

"""
Usage:
python collect_faces.py --name "YourName" --count 120
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="Gubendhiran")
    parser.add_argument("--count", type=int, default=120, help="How many face images to capture")
    parser.add_argument("--dataset_dir", default=r"G:\face\My face")
    args = parser.parse_args()

    person_dir = Path(args.dataset_dir) / args.name
    person_dir.mkdir(parents=True, exist_ok=True)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture("http://192.168.29.238:8080/video")

    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try a different index (0/1) or check permissions.")

    print(f"[INFO] Capturing {args.count} face images for: {args.name}")
    saved = 0

    while saved < args.count:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed."); continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            x0 = max(0, x-10); y0 = max(0, y-10)
            x1 = min(gray.shape[1], x + w + 10); y1 = min(gray.shape[0], y + h + 10)
            face_roi = gray[y0:y1, x0:x1]
            face_roi = cv2.resize(face_roi, (200, 200))
            filename = person_dir / f"{args.name}_{saved:04d}.png"
            cv2.imwrite(str(filename), face_roi)
            saved += 1
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, f"Saved: {saved}/{args.count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Collect Faces - Press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Saved {saved} images to {person_dir}")

if __name__ == "__main__":
    main()