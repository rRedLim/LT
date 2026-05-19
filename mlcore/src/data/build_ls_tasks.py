"""Собирает tasks.json и labels_config.xml для Label Studio из папки с кадрами.

Запускается ПОСЛЕ sample_frames. На вход — `<frames_root>/<video>/t<ms>.jpg`.

Использование:
    cd ml_core
    python3 scripts/prepare_labeling.py build-tasks \\
        --frames ../labeling/frames \\
        --out ../labeling/tasks.json

Затем:
    1) cd <frames_root> && python3 -m http.server 8081
    2) В Label Studio: New Project → Labeling Setup → Custom template
       → вставить содержимое labels_config.xml
    3) Import → выбрать tasks.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


log = logging.getLogger(__name__)


LABELS_XML = """\
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="bbox" toName="image">
    <Label value="pricetag" background="#FF3333"/>
  </RectangleLabels>
</View>
"""


def build_tasks(
    frames_root: Path,
    out_json: Path,
    http_base: str = "http://localhost:8081",
) -> int:
    """Собирает tasks.json + labels_config.xml.

    Параметры
    ---------
    frames_root: папка вида labeling/frames/ с подпапками <video>/t<ms>.jpg.
    out_json:    путь куда сохранить tasks.json (например, ../labeling/tasks.json).
    http_base:   базовый URL HTTP-сервера, обслуживающего frames_root.

    Возвращает количество созданных задач (одна задача = один кадр).
    """
    if not frames_root.exists():
        raise FileNotFoundError(f"frames_root not found: {frames_root}")

    tasks: list[dict] = []
    for video_dir in sorted(frames_root.iterdir()):
        if not video_dir.is_dir():
            continue
        for img in sorted(video_dir.glob("t*.jpg")):
            rel = f"{video_dir.name}/{img.name}"
            try:
                ts_ms = int(img.stem.lstrip("t"))
            except ValueError:
                ts_ms = 0
            tasks.append({
                "data": {
                    "image": f"{http_base.rstrip('/')}/{rel}",
                    "video": video_dir.name,
                    "frame": img.name,
                    "ts_ms": ts_ms,
                }
            })

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    xml_path = out_json.parent / "labels_config.xml"
    xml_path.write_text(LABELS_XML, encoding="utf-8")

    log.info("Создано задач: %d", len(tasks))
    log.info("  tasks: %s", out_json)
    log.info("  xml:   %s", xml_path)
    return len(tasks)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--http-base", type=str, default="http://localhost:8081")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    n = build_tasks(args.frames, args.out, args.http_base)
    print(f"\nДальше:")
    print(f"  1) cd {args.frames} && python3 -m http.server 8081")
    print( "  2) В Label Studio: New Project → Labeling Setup → Custom template")
    print(f"     → вставить содержимое labels_config.xml")
    print(f"  3) Import → выбрать {args.out.name}")
    print(f"  4) После разметки: Export → JSON → сохранить как {args.out.parent}/ls_export.json")
