from __future__ import annotations
from pathlib import Path
import random
import shutil
import cv2
import numpy as np


def motion_blur(img: np.ndarray, kernel: int = 15, angle: float = 0.0) -> np.ndarray:
    k = np.zeros((kernel, kernel), dtype=np.float32)
    k[kernel // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((kernel / 2, kernel / 2), angle, 1.0)
    k = cv2.warpAffine(k, M, (kernel, kernel))
    k /= k.sum() + 1e-9
    return cv2.filter2D(img, -1, k)


def augment_motion_dataset(src_images: Path, src_labels: Path,
                           dst_images: Path, dst_labels: Path,
                           n_copies: int = 1, seed: int = 42) -> int:
    rng = random.Random(seed)
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    n = 0
    images = sorted(src_images.glob("*.jpg"))
    from src.utils.progress import progress
    for img_path in progress(images, "motion-blur"):
        lbl_path = src_labels / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        for i in range(n_copies):
            kernel = rng.randint(9, 21)
            angle = rng.uniform(-30, 30)
            out = motion_blur(img, kernel, angle)
            stem = f"{img_path.stem}_mblur{i}"
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
    print(augment_motion_dataset(args.src_images, args.src_labels,
                                 args.dst_images, args.dst_labels, args.n_copies))
