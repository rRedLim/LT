"""CLI: py scripts/train_lora.py --config configs/lora_v3.yaml

Запускает прямую тренировку LoRA через src.training.train_lora.train (без legacy).
"""
# sys.path bootstrap для запуска `py scripts/...` (Python добавляет в path
# папку скрипта, а не родителя). Без этого `from src.X import Y` упадёт.
import sys as _sys
from pathlib import Path as _Path
_ML_CORE = str(_Path(__file__).resolve().parents[1])
if _ML_CORE not in _sys.path:
    _sys.path.insert(0, _ML_CORE)

from pathlib import Path
import argparse
import logging

from src.training.train_lora import train


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/lora_v3.yaml"))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    r = train(args.config)
    print(r)
