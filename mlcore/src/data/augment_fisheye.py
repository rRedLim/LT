"""Fisheye-аугментация iPhone-кадров чтобы они «выглядели» как fisheye-кадры с робота.

Ключевая оптимизация: cv2.remap-карта строится ОДИН РАЗ на уникальный
(W, H, strength) и кешируется. Без кеша: 374 кадра × ~10 сек = час+.
С кешем: одна построенная карта применяется ко всем 374 кадрам за секунды.

Применяется ТОЛЬКО к нашим iPhone-видео (у орг fisheye уже есть от камеры робота).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


STRENGTH_PRESETS = {
    "weak":   dict(k1=-0.10, k2=0.00),
    "medium": dict(k1=-0.25, k2=0.05),
    "strong": dict(k1=-0.45, k2=0.15),
}


def _build_K(W: int, H: int) -> np.ndarray:
    fx = fy = max(W, H) * 0.55
    return np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=np.float64)


def _build_D(strength: dict) -> np.ndarray:
    return np.array([strength["k1"], strength["k2"], 0.0, 0.0], dtype=np.float64)


def _pixels_to_normalized(pts_px: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Пиксельные координаты (N,1,2) → нормированные для fisheye.distortPoints."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    out = pts_px.copy()
    out[..., 0] = (pts_px[..., 0] - cx) / fx
    out[..., 1] = (pts_px[..., 1] - cy) / fy
    return out


# Глобальный кеш карт ремаппинга. Ключ — (W, H, strength_name).
# Эти карты — float32 массивы размера H×W×2; они большие (для 1920×1080 ≈ 16 MB),
# но переиспользуются между всеми кадрами с тем же размером.
_REMAP_CACHE: dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray]] = {}


def _get_remap_maps(W: int, H: int, strength_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Строит и кеширует карту ремаппинга (map_x, map_y) для cv2.remap."""
    key = (W, H, strength_name)
    cached = _REMAP_CACHE.get(key)
    if cached is not None:
        return cached
    strength = STRENGTH_PRESETS[strength_name]
    K = _build_K(W, H)
    D = _build_D(strength)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    pts_px = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2).astype(np.float64)
    pts_norm = _pixels_to_normalized(pts_px, K)
    dist = cv2.fisheye.distortPoints(pts_norm, K, D).reshape(H, W, 2).astype(np.float32)
    map_x = dist[..., 0]
    map_y = dist[..., 1]
    _REMAP_CACHE[key] = (map_x, map_y)
    return map_x, map_y


def apply_fisheye_image(img: np.ndarray, strength: dict | str) -> np.ndarray:
    """Применяет fisheye-дисторсию.

    `strength` принимается двумя форматами:
      - dict {'k1': ..., 'k2': ...} (legacy для тестов)
      - str "weak"/"medium"/"strong" (рекомендуется — задействует кеш)

    При dict-форме карта НЕ кешируется (нет имени), при str-форме кешируется.
    """
    H, W = img.shape[:2]
    if isinstance(strength, str):
        map_x, map_y = _get_remap_maps(W, H, strength)
    else:
        # Без кеша — для обратной совместимости с тестами
        K = _build_K(W, H)
        D = _build_D(strength)
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        pts_px = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2).astype(np.float64)
        pts_norm = _pixels_to_normalized(pts_px, K)
        dist = cv2.fisheye.distortPoints(pts_norm, K, D).reshape(H, W, 2).astype(np.float32)
        map_x = dist[..., 0]
        map_y = dist[..., 1]
    return cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def distort_bbox(bbox: Tuple[float, float, float, float], W: int, H: int,
                 strength: dict) -> Tuple[float, float, float, float]:
    """Пересчёт bbox через 4 угла + bounding rect. Дёшево (4 точки), кеш не нужен."""
    x1, y1, x2, y2 = bbox
    K = _build_K(W, H)
    D = _build_D(strength)
    corners_px = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                           dtype=np.float64).reshape(-1, 1, 2)
    corners_norm = _pixels_to_normalized(corners_px, K)
    distorted = cv2.fisheye.distortPoints(corners_norm, K, D).reshape(-1, 2)
    nx1 = max(0.0, float(distorted[:, 0].min()))
    ny1 = max(0.0, float(distorted[:, 1].min()))
    nx2 = min(float(W), float(distorted[:, 0].max()))
    ny2 = min(float(H), float(distorted[:, 1].max()))
    return nx1, ny1, nx2, ny2


def augment_yolo_dataset(
    src_images: Path, src_labels: Path,
    dst_images: Path, dst_labels: Path,
    strength_name: str = "medium",
    suffix: str = "_fish_med",
) -> int:
    """Применяет fisheye ко всем парам (img, label) и сохраняет в dst с суффиксом.

    Карта ремаппинга строится один раз на (W, H, strength_name) — все 374 кадра
    с одинаковым размером используют одну и ту же карту, очень быстро.
    """
    strength = STRENGTH_PRESETS[strength_name]
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(src_images.glob("*.jpg"))
    if not images:
        return 0

    # Прогресс через print() — tqdm не работает на cloud-jupyter без TTY.
    from src.utils.progress import progress

    n = 0
    for img_path in progress(images, f"fisheye[{strength_name}]"):
        lbl_path = src_labels / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        # Используем str-форму strength → кеш карты по (W, H, strength_name)
        out = apply_fisheye_image(img, strength_name)
        out_img = dst_images / f"{img_path.stem}{suffix}.jpg"
        out_lbl = dst_labels / f"{img_path.stem}{suffix}.txt"
        cv2.imwrite(str(out_img), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(lbl_path, encoding="utf-8") as lf:
            lines = lf.read().splitlines()
        with open(out_lbl, "w", encoding="utf-8") as wf:
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, w, h = parts
                cx, cy, w, h = map(float, [cx, cy, w, h])
                x1 = (cx - w / 2) * W
                y1 = (cy - h / 2) * H
                x2 = (cx + w / 2) * W
                y2 = (cy + h / 2) * H
                nx1, ny1, nx2, ny2 = distort_bbox((x1, y1, x2, y2), W, H, strength)
                nw = max(0.0, nx2 - nx1) / W
                nh = max(0.0, ny2 - ny1) / H
                if nw <= 0 or nh <= 0:
                    continue
                ncx = (nx1 + nx2) / 2 / W
                ncy = (ny1 + ny2) / 2 / H
                wf.write(f"{cls} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}\n")
        n += 1
    return n


def preview_strengths(input_image: Path, out_dir: Path) -> None:
    """Создаёт N копий для визуальной калибровки силы."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(input_image))
    if img is None:
        raise IOError(f"Cannot read {input_image}")
    for name in STRENGTH_PRESETS:
        out = apply_fisheye_image(img, name)
        cv2.imwrite(str(out_dir / f"{input_image.stem}_{name}.jpg"), out)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("augment")
    a.add_argument("--src-images", type=Path, required=True)
    a.add_argument("--src-labels", type=Path, required=True)
    a.add_argument("--dst-images", type=Path, required=True)
    a.add_argument("--dst-labels", type=Path, required=True)
    a.add_argument("--strength", choices=list(STRENGTH_PRESETS), default="medium")
    a.add_argument("--suffix", default="_fish_med")

    pv = sub.add_parser("preview")
    pv.add_argument("--input", type=Path, required=True)
    pv.add_argument("--out", type=Path, required=True)

    args = p.parse_args()
    if args.cmd == "augment":
        n = augment_yolo_dataset(args.src_images, args.src_labels,
                                 args.dst_images, args.dst_labels,
                                 args.strength, args.suffix)
        print(f"Wrote {n} augmented pairs")
    else:
        preview_strengths(args.input, args.out)
        print(f"Previews in {args.out}")
