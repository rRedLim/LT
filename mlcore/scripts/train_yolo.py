"""CLI-обёртка: py scripts/train_yolo.py --config configs/yolo_v3.yaml"""
# sys.path bootstrap для запуска `py scripts/...` (Python добавляет в path
# папку скрипта, а не родителя). Без этого `from src.X import Y` упадёт.
import sys as _sys
from pathlib import Path as _Path
_ML_CORE = str(_Path(__file__).resolve().parents[1])
if _ML_CORE not in _sys.path:
    _sys.path.insert(0, _ML_CORE)

from pathlib import Path
from src.training.train_yolo import train

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/yolo_v3.yaml"))
    args = p.parse_args()
    r = train(args.config)
    print("Done. save_dir:", r["save_dir"])
    print("Metrics:", r["metrics"])
