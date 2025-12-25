import cv2
import os
import json
import numpy as np
from pathlib import Path
import argparse

"""
Usage:
python train_lbph.py --dataset_dir dataset --model_out models/lbph_face_model.xml --labels_out models/labels.json
"""
dataset_dir='G:\face\My face'
def load_images_and_labels(dataset_dir):
    images = []
    labels = []
    label_map = {}

    dataset_dir = Path(dataset_dir)
    class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    class_dirs.sort()

    current_label = 0
    for d in class_dirs:
        label_map[current_label] = d.name
        for img_file in list(d.glob("*.png")) + list(d.glob("*.jpg")) + list(d.glob("*.jpeg")):


            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200, 200))
            images.append(img)
            labels.append(current_label)
        current_label += 1

    return images, np.array(labels, dtype=np.int32), label_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=r"G:\face\My face")

    parser.add_argument("--model_out", default="models/lbph_face_model.xml")
    parser.add_argument("--labels_out", default="models/labels.json")
    args = parser.parse_args()

    images, labels, label_map = load_images_and_labels(args.dataset_dir)
    if len(images) == 0:
        raise RuntimeError("No training images found. Run collect_faces.py first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)
    print("[INFO] Training LBPH recognizer...")
    recognizer.train(images, labels)
    recognizer.save(args.model_out)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.labels_out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.labels_out, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"[DONE] Model saved to {args.model_out}")
    print(f"[DONE] Label map saved to {args.labels_out} -> {label_map}")

if __name__ == "__main__":
    main()