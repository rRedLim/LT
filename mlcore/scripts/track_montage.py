"""Создаёт montage (плакат) всех кадров одного трека для визуальной диагностики.

Каждый кадр — реальный bbox-кроп, подписан ts_ms / ширина / sharpness / EDGE.
Раскладывается в сетку, чтобы за один взгляд понять, есть ли в треке полный
кадр или вся серия — половинки.

Запуск:
    py scripts/track_montage.py \\
        --video ../dataset/dataset_myself/IMG_6511.mov \\
        --yolo runs/yolo/v3/weights/best.pt \\
        --track 33 \\
        --out /tmp/montage_33.jpg
"""
import sys as _sys
from pathlib import Path as _Path
_ML_CORE = str(_Path(__file__).resolve().parents[1])
if _ML_CORE not in _sys.path:
    _sys.path.insert(0, _ML_CORE)

import argparse
from math import ceil, sqrt
from pathlib import Path

import cv2
import numpy as np

from src.inference.detector import detect_and_track
from src.inference.frame_picker import _sharpness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--yolo", type=Path, required=True)
    ap.add_argument("--track", type=int, required=True)
    ap.add_argument("--cell", type=int, default=240,
                    help="ширина одной ячейки сетки в пикселях")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tracks = detect_and_track(args.video, args.yolo, keep_frames=True)
    target = next((t for t in tracks if t.track_id == args.track), None)
    if target is None:
        ids_with_len = sorted(((t.track_id, len(t.frames)) for t in tracks),
                              key=lambda x: -x[1])
        print(f"trk {args.track} не найден. Top по длине:", ids_with_len[:10])
        return

    n = len(target.frames)
    cols = ceil(sqrt(n))
    rows = ceil(n / cols)
    cell_w = args.cell
    cell_h = int(args.cell * 0.65)

    canvas = np.full((rows * cell_h, cols * cell_w, 3), 30, dtype=np.uint8)
    for i, fd in enumerate(target.frames):
        if fd.frame is None:
            continue
        r, c = divmod(i, cols)
        x0, y0 = c * cell_w, r * cell_h
        # ресайз кропа в ячейку с сохранением пропорций
        crop = fd.frame
        ch, cw = crop.shape[:2]
        scale = min((cell_w - 4) / cw, (cell_h - 28) / ch)
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas[y0:y0+nh, x0:x0+nw] = resized
        # подпись
        bw = int(fd.bbox[2] - fd.bbox[0])
        bh = int(fd.bbox[3] - fd.bbox[1])
        s = int(_sharpness(crop))
        edge = "EDGE" if getattr(fd, "touches_edge", False) else "    "
        label = f"i{i:02d} ts{fd.ts_ms} {bw}x{bh} s={s} {edge}"
        cv2.putText(canvas, label, (x0 + 2, y0 + nh + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 200), 1,
                    cv2.LINE_AA)
        # рамка
        cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1),
                      (60, 60, 60), 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), canvas)
    print(f"trk {args.track}: {n} frames → {args.out} ({cols}×{rows} grid)")


if __name__ == "__main__":
    main()
