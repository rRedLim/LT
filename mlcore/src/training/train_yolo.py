from __future__ import annotations
from pathlib import Path
import yaml


def train(config_path: Path) -> dict:
    """Запускает YOLO11s train через ultralytics. Конфиг полностью прокидывается."""
    from ultralytics import YOLO
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = YOLO(cfg["model"])
    keys_for_train = {k: v for k, v in cfg.items() if k != "model"}
    results = model.train(**keys_for_train)
    metrics = getattr(results, "results_dict", {}) or {}
    return {
        "save_dir": str(results.save_dir),
        "metrics": metrics,
    }


if __name__ == "__main__":
    import argparse
    import json
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/yolo_v3.yaml"))
    args = p.parse_args()
    r = train(args.config)
    print(json.dumps(r, indent=2, default=str))
