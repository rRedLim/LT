"""End-to-end: ../dataset/dataset_orig + LS-export + augs → datasets/detection_v3.

Структура целевых данных:
  <wd>/
    ml_core/                  ← запуск отсюда
    dataset/
      dataset_orig/<video>/<video>.{mp4,csv}
      dataset_myself/*.{mp4,mov}
    labeling/                 ← опционально, если есть LS-разметка наших 13
"""
# sys.path bootstrap для запуска `py scripts/...` (Python добавляет в path
# папку скрипта, а не родителя). Без этого `from src.X import Y` упадёт.
import sys as _sys
from pathlib import Path as _Path
_ML_CORE = str(_Path(__file__).resolve().parents[1])
if _ML_CORE not in _sys.path:
    _sys.path.insert(0, _ML_CORE)

import argparse
from pathlib import Path

from src.data.build_detection_dataset import build


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--organizer-root", type=Path,
                   default=Path("../dataset/dataset_orig"))
    p.add_argument("--ls-export", type=Path,
                   default=Path("../labeling/ls_export.json"))
    p.add_argument("--frames", type=Path,
                   default=Path("../labeling/frames"))
    p.add_argument("--out", type=Path, default=Path("datasets/detection_v3"))
    # fisheye/motion аугментация по умолчанию ВЫКЛЮЧЕНА — synthetic fisheye
    # сильно отличается от реального 140°-HFOV робота и портит модель.
    p.add_argument("--fisheye-strengths", nargs="*",
                   default=[],
                   choices=["weak", "medium", "strong"])
    p.add_argument("--motion-copies", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="Доля train → val (random split по картинкам)")
    p.add_argument("--split-seed", type=int, default=42,
                   help="Seed для воспроизводимости val-split'а")
    args = p.parse_args()
    s = build(
        args.organizer_root, args.ls_export, args.frames, args.out,
        fisheye_strengths=tuple(args.fisheye_strengths),
        motion_copies=args.motion_copies,
        val_frac=args.val_frac,
        split_seed=args.split_seed,
    )
    for k, v in s.items():
        print(f"  {k}: {v}")
