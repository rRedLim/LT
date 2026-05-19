"""Прогоняет YOLO по одному кадру видео и сохраняет картинку с bbox.

Нужен чтобы проверить: реально ли детектор НАХОДИТ ценник на этом конкретном
кадре. Если на резком кадре #15 он не детектит — проблема в детекторе.

Запуск:
    py scripts/detect_only_at_timestamp.py \\
        --video ../dataset/dataset_myself/1.mov \\
        --yolo runs/yolo/v3/weights/best.pt \\
        --ts-ms 5000 \\
        --out /tmp/det_5000.jpg
"""
import sys as _sys
from pathlib import Path as _Path
_ML_CORE = str(_Path(__file__).resolve().parents[1])
if _ML_CORE not in _sys.path:
    _sys.path.insert(0, _ML_CORE)

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--yolo", type=Path, required=True)
    ap.add_argument("--ts-ms", type=int, required=True,
                    help="таймстемп в мс — какой кадр посмотреть")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_MSEC, args.ts_ms)
    ok, frame = cap.read()
    if not ok:
        print("Не удалось прочитать кадр")
        return
    cap.release()

    model = YOLO(str(args.yolo))
    results = model.predict(frame, conf=args.conf, imgsz=1280, verbose=False)
    annotated = frame.copy()
    n_det = 0
    for r in results:
        if r.boxes is None:
            continue
        for box, cf in zip(r.boxes.xyxy.cpu().tolist(),
                           r.boxes.conf.cpu().tolist()):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(annotated, f"{cf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            n_det += 1
    cv2.imwrite(str(args.out), annotated)
    print(f"ts={args.ts_ms}ms: detected {n_det} pricetags → {args.out}")


if __name__ == "__main__":
    main()
