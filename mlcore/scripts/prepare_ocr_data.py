"""End-to-end OCR-разметка: prefill → wait_for_excel → finalize → build.

Использование (3 шага):
  py scripts/prepare_ocr_data.py prefill \\
      --yolo runs/yolo/v3/weights/best.pt \\
      --lora runs/lora/v3/final
  # … вручную правишь labels_myself/*__editable.csv в Excel UTF-8 …
  py scripts/prepare_ocr_data.py finalize
  py scripts/prepare_ocr_data.py build

Структура целевых данных:
  <wd>/ml_core/   ← запуск отсюда
  <wd>/dataset/dataset_myself/*.{mp4,mov}
  <wd>/dataset/dataset_orig/<video>/<video>.{mp4,csv}
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

from src.data.ocr_prefill import run_prefill
from src.data.validate_csv import validate_file
from src.data.build_ocr_dataset import build as build_ocr


def cmd_prefill(args):
    only = (
        [s.strip() for s in args.videos.split(",") if s.strip()]
        if args.videos else None
    )
    n = run_prefill(args.data, args.yolo, args.lora, args.out, only_videos=only)
    print(f"Prefilled {n} videos to {args.out}")
    print(f"\nNOW: открой каждый __editable.csv в Excel UTF-8, поправь поля.")
    print(f"Затем: py scripts/prepare_ocr_data.py finalize")


def cmd_finalize(args):
    in_dir = args.in_dir
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_fail = 0
    for editable in sorted(in_dir.glob("*__editable.csv")):
        rc = validate_file(editable)
        if rc != 0:
            print(f"FAIL {editable.name}: validate rc={rc}")
            n_fail += 1
            continue
        # finalize: используем csv-логику здесь без вызова legacy main()
        # (т.к. legacy main() парсит свой argparse и делает sys.exit)
        import csv
        out_path = out_dir / editable.name.replace("__editable", "")
        with open(editable, encoding="utf-8") as fi, \
             open(out_path, "w", encoding="utf-8", newline="") as fo:
            rdr = csv.reader(fi)
            hdr = next(rdr)
            keep = [i for i, c in enumerate(hdr) if not c.startswith("__")]
            fo.write(",".join(hdr[i] for i in keep) + "\n")
            for row in rdr:
                cells = [row[i] for i in keep]
                fixed = []
                for v in cells:
                    if "," in v and not (v.startswith('"') and v.endswith('"')):
                        v = '"' + v.replace('"', '""') + '"'
                    fixed.append(v)
                fo.write(",".join(fixed) + "\n")
        n_ok += 1
        print(f"OK   {editable.name} → {out_path.name}")
    print(f"\n{n_ok} CSV финализированы, {n_fail} с ошибками. Далее:")
    print(f"  py scripts/prepare_ocr_data.py build")


def cmd_build(args):
    org_vids = (
        [s.strip() for s in args.organizer_videos.split(",") if s.strip()]
        if args.organizer_videos else None
    )
    s = build_ocr(
        args.our_videos, args.our_csvs, args.organizer_root,
        args.out, args.val_frac, args.h_min, args.sharp_min,
        photos_dir=args.photos_dir,
        organizer_videos=org_vids,
    )
    for k, v in s.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("prefill")
    pf.add_argument("--data", type=Path,
                    default=Path("../dataset/dataset_myself"))
    pf.add_argument("--yolo", type=Path, required=True)
    # --lora опциональный: на первом проходе (до обучения LoRA) можно опустить —
    # VLMReader корректно работает без адаптера (см. pipeline.py:131).
    pf.add_argument("--lora", type=Path, required=False,
                    default=Path("runs/lora/v3/final"))
    pf.add_argument("--out", type=Path, default=Path("labels_myself"))
    pf.add_argument("--videos", type=str, default=None,
                    help="comma-list stem'ов видео для прогонки (без расш.), "
                         "напр. '1,2,5'. По умолчанию все .mp4/.mov из --data.")
    pf.set_defaults(func=cmd_prefill)

    fn = sub.add_parser("finalize")
    fn.add_argument("--in-dir", type=Path, default=Path("labels_myself"))
    fn.add_argument("--out", type=Path, default=Path("datasets/ocr_raw"))
    fn.set_defaults(func=cmd_finalize)

    bd = sub.add_parser("build")
    bd.add_argument("--our-videos", type=Path,
                    default=Path("../dataset/dataset_myself"))
    bd.add_argument("--our-csvs", type=Path, default=Path("datasets/ocr_raw"))
    bd.add_argument("--organizer-root", type=Path,
                    default=Path("../dataset/dataset_orig"))
    bd.add_argument("--photos-dir", type=Path, default=None,
                    help="папка с парами image.jpg + image.json (23 поля). "
                         "Без фильтра sharpness/h. Например: datasets/photos_v3")
    bd.add_argument("--organizer-videos", type=str, default=None,
                    help="comma-list имён видео орг. По умолчанию все 5 "
                         "(25_12-20,26_12-20,25_2-10,43_15,49_5).")
    bd.add_argument("--out", type=Path, default=Path("datasets/ocr_v3"))
    bd.add_argument("--val-frac", type=float, default=0.2)
    bd.add_argument("--h-min", type=int, default=120)
    bd.add_argument("--sharp-min", type=float, default=40.0)
    bd.set_defaults(func=cmd_build)

    args = p.parse_args()
    args.func(args)
