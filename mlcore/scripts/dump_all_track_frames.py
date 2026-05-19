"""Дампит ВСЕ кадры одного трека из видео в папку, чтобы глазами посмотреть.

Полезно когда pick_A выдаёт плохой кадр — глянуть, есть ли вообще нормальный
кадр в этом треке или ByteTrack отрезал трек на blur-куске.

Запуск:
    py scripts/dump_all_track_frames.py \\
        --video ../dataset/dataset_myself/1.mov \\
        --yolo runs/yolo/v3/weights/best.pt \\
        --track 5 \\
        --out /tmp/track_5_dump
"""
import sys as _sys
from pathlib import Path as _Path
_ML_CORE = str(_Path(__file__).resolve().parents[1])
if _ML_CORE not in _sys.path:
    _sys.path.insert(0, _ML_CORE)

import argparse
from pathlib import Path

import cv2

from src.inference.detector import detect_and_track
from src.inference.frame_picker import _sharpness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--yolo", type=Path, required=True)
    ap.add_argument("--track", type=int, default=None,
                    help="track_id для дампа (если None — дампим все треки длиннее --min-len)")
    ap.add_argument("--min-len", type=int, default=20,
                    help="минимальная длина трека для дампа (если --track не задан)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tracks = detect_and_track(args.video, args.yolo, keep_frames=True)
    print(f"Всего треков: {len(tracks)}")
    by_len = sorted(tracks, key=lambda t: len(t.frames), reverse=True)
    print(f"Top-10 треков по длине: {[(t.track_id, len(t.frames)) for t in by_len[:10]]}")

    targets = [t for t in tracks if t.track_id == args.track] if args.track is not None \
              else [t for t in tracks if len(t.frames) >= args.min_len]

    if not targets:
        print(f"Нет подходящих треков")
        return

    for tr in targets:
        tdir = args.out / f"trk_{tr.track_id:04d}"
        tdir.mkdir(parents=True, exist_ok=True)
        scored = []
        for i, fd in enumerate(tr.frames):
            if fd.frame is None:
                continue
            s = _sharpness(fd.frame)
            scored.append((s, i, fd))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Сохраняем все кадры с префиксом ранга в имени
        for rank, (s, i, fd) in enumerate(scored):
            edge_tag = "_EDGE" if getattr(fd, "touches_edge", False) else ""
            name = (f"rank{rank:03d}_score{s:09.0f}{edge_tag}"
                    f"_idx{i:03d}_ts{fd.ts_ms}.jpg")
            cv2.imwrite(str(tdir / name), fd.frame)
        print(f"trk={tr.track_id:4d} frames={len(tr.frames):3d}  "
              f"top_score={scored[0][0] if scored else 0:.1f}  "
              f"→ {tdir} ({len(scored)} кадров)")


if __name__ == "__main__":
    main()
