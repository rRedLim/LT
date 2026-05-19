from __future__ import annotations
from pathlib import Path
import random
import shutil
import cv2
import numpy as np


def add_glare(img: np.ndarray, n_spots: int = 2, seed: int | None = None) -> np.ndarray:
    rng = random.Random(seed)
    H, W = img.shape[:2]
    overlay = np.zeros_like(img)
    for _ in range(n_spots):
        cx, cy = rng.randint(0, W), rng.randint(0, H)
        rx, ry = rng.randint(40, 160), rng.randint(20, 80)
        angle = rng.uniform(0, 180)
        cv2.ellipse(overlay, (cx, cy), (rx, ry), angle, 0, 360, (255, 255, 255), -1)
    overlay = cv2.GaussianBlur(overlay, (51, 51), 0)
    alpha = 0.45
    return cv2.addWeighted(img, 1.0, overlay, alpha, 0)


def augment_glare_dataset(src_images: Path, src_labels: Path,
                          dst_images: Path, dst_labels: Path,
                          n_copies: int = 1, seed: int = 42) -> int:
    rng = random.Random(seed)
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    n = 0
    images = sorted(src_images.glob("*.jpg"))
    from src.utils.progress import progress
    for img_path in progress(images, "glare"):
        lbl_path = src_labels / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        for i in range(n_copies):
            out = add_glare(img, n_spots=rng.randint(1, 3), seed=rng.randint(0, 1 << 31))
            stem = f"{img_path.stem}_glare{i}"
            cv2.imwrite(str(dst_images / f"{stem}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
            shutil.copy(lbl_path, dst_labels / f"{stem}.txt")
            n += 1
    return n


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--src-images", type=Path, required=True)
    p.add_argument("--src-labels", type=Path, required=True)
    p.add_argument("--dst-images", type=Path, required=True)
    p.add_argument("--dst-labels", type=Path, required=True)
    p.add_argument("--n-copies", type=int, default=1)
    args = p.parse_args()
    print(augment_glare_dataset(args.src_images, args.src_labels,
                                args.dst_images, args.dst_labels, args.n_copies))
