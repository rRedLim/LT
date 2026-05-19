"""End-to-end подготовка к разметке в Label Studio.

Использование (2 шага + ручная разметка + экспорт):

  # 1. Нарезаем видео на кадры:
  python3 scripts/prepare_labeling.py sample-frames \\
      --data ../dataset/dataset_myself \\
      --out ../labeling/frames

  # 2. Собираем tasks.json и labels_config.xml:
  python3 scripts/prepare_labeling.py build-tasks \\
      --frames ../labeling/frames \\
      --out ../labeling/tasks.json

  # 3. РУЧНОЙ ШАГ: разметка в Label Studio
  #    a) запусти HTTP-сервер чтобы LS видел картинки:
  #       cd ../labeling/frames && python3 -m http.server 8081
  #    b) в LS: New Project → Labeling Setup → Custom template
  #       → вставь содержимое ../labeling/labels_config.xml
  #    c) Import → выбери ../labeling/tasks.json
  #    d) Размечай ценники (RectangleLabels → pricetag)
  #    e) Когда закончил: Export → JSON → сохрани как ../labeling/ls_export.json

  # 4. Запускай детекшн-пайплайн:
  python3 scripts/prepare_detection_data.py
"""
# sys.path bootstrap
import sys as _sys
from pathlib import Path as _Path
_ML_CORE = str(_Path(__file__).resolve().parents[1])
if _ML_CORE not in _sys.path:
    _sys.path.insert(0, _ML_CORE)

import argparse
import logging
from pathlib import Path

from src.data.sample_frames import sample_all
from src.data.build_ls_tasks import build_tasks


def cmd_sample_frames(args):
    sample_all(
        data_dir=args.data,
        out_dir=args.out,
        fps=args.fps,
        max_side=args.max_side,
        jpg_quality=args.jpg_quality,
        skip_existing=not args.no_skip_existing,
    )


def cmd_build_tasks(args):
    n = build_tasks(args.frames, args.out, args.http_base)
    print(f"\nДальше:")
    print(f"  1) cd {args.frames} && python3 -m http.server 8081")
    print( "  2) В Label Studio: New Project → Labeling Setup → Custom template")
    print(f"     → вставь содержимое {args.out.parent}/labels_config.xml")
    print(f"  3) Import → выбери {args.out}")
    print(f"  4) Размечай ценники")
    print(f"  5) Export → JSON → сохрани как {args.out.parent}/ls_export.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sf = sub.add_parser("sample-frames", help="Нарезка видео на JPG для разметки")
    sf.add_argument("--data", type=Path, default=Path("../dataset/dataset_myself"))
    sf.add_argument("--out", type=Path, default=Path("../labeling/frames"))
    sf.add_argument("--fps", type=float, default=1.0)
    sf.add_argument("--max-side", type=int, default=1920)
    sf.add_argument("--jpg-quality", type=int, default=92)
    sf.add_argument("--no-skip-existing", action="store_true")
    sf.set_defaults(func=cmd_sample_frames)

    bt = sub.add_parser("build-tasks", help="Собрать tasks.json для Label Studio")
    bt.add_argument("--frames", type=Path, default=Path("../labeling/frames"))
    bt.add_argument("--out", type=Path, default=Path("../labeling/tasks.json"))
    bt.add_argument("--http-base", type=str, default="http://localhost:8081")
    bt.set_defaults(func=cmd_build_tasks)

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.func(args)
