from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import cv2


@dataclass
class OrganizerCsvRow:
    ts_ms: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def parse_organizer_csv(csv_path: Path) -> List[OrganizerCsvRow]:
    """Reads organizer CSV. Handles comma-decimal coords ("2063,1" -> float)."""
    rows: List[OrganizerCsvRow] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append(OrganizerCsvRow(
                    ts_ms=int(r["frame_timestamp"]),
                    x_min=float(r["x_min"].replace(",", ".")),
                    y_min=float(r["y_min"].replace(",", ".")),
                    x_max=float(r["x_max"].replace(",", ".")),
                    y_max=float(r["y_max"].replace(",", ".")),
                ))
            except (KeyError, ValueError):
                continue
    return rows


def csv_to_yolo_bbox(row: OrganizerCsvRow, frame_w: int, frame_h: int) -> Tuple[float, float, float, float]:
    x1, y1 = max(0.0, row.x_min), max(0.0, row.y_min)
    x2, y2 = min(float(frame_w), row.x_max), min(float(frame_h), row.y_max)
    cx = ((x1 + x2) / 2) / frame_w
    cy = ((y1 + y2) / 2) / frame_h
    w = (x2 - x1) / frame_w
    h = (y2 - y1) / frame_h
    return cx, cy, w, h


def _offsets(window_ms: int) -> List[int]:
    if window_ms <= 0:
        return [0]
    half = window_ms // 2
    return [-half, 0, half]


def import_video(
    video_path: Path,
    csv_path: Path,
    out_images: Path,
    out_labels: Path,
    window_ms: int = 500,
    cls_id: int = 0,
) -> int:
    """Extracts frames at timestamps from CSV (+optional window ±window_ms) and writes YOLO labels.

    window_ms=0  -> GT frame only.
    window_ms=500 -> also frames at t±250 (same bbox).
    If all ts=0 in the file -> extract only one frame (guards against overfitting on a single scene).
    Returns the number of (image, label) pairs written.
    """
    rows = parse_organizer_csv(csv_path)
    if not rows:
        return 0
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    all_zero = all(r.ts_ms == 0 for r in rows)

    by_ts: dict[int, List[OrganizerCsvRow]] = {}
    for r in rows:
        by_ts.setdefault(r.ts_ms, []).append(r)

    pairs_written = 0
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return 0
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        from src.utils.progress import progress
        for ts, rows_at_ts in progress(list(by_ts.items()), f"import {video_path.stem}"):
            ts_offsets = [0] if all_zero else _offsets(window_ms)
            for off in ts_offsets:
                tgt = ts + off
                if tgt < 0:
                    continue
                cap.set(cv2.CAP_PROP_POS_MSEC, tgt)
                ok, frame = cap.read()
                if not ok:
                    continue
                stem = f"{video_path.stem}_t{tgt:07d}"
                img_path = out_images / f"{stem}.jpg"
                lbl_path = out_labels / f"{stem}.txt"
                cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                with open(lbl_path, "w", encoding="utf-8") as lf:
                    for r in rows_at_ts:
                        cx, cy, w, h = csv_to_yolo_bbox(r, W, H)
                        if w > 0 and h > 0:
                            lf.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                pairs_written += 1
    finally:
        cap.release()
    return pairs_written


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help="Root data folder (Данные/)")
    p.add_argument("--videos", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--window-ms", type=int, default=500)
    args = p.parse_args()
    total = 0
    for v in args.videos:
        vd = args.data / v
        total += import_video(
            video_path=vd / f"{v}.mp4",
            csv_path=vd / f"{v}.csv",
            out_images=args.out / "images" / args.split,
            out_labels=args.out / "labels" / args.split,
            window_ms=args.window_ms,
        )
    print(f"Wrote {total} pairs to {args.out}")
