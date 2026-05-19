"""
ls_export_to_yolo.py — конвертирует экспорт Label Studio (JSON / JSON-MIN) в YOLO-датасет.

Публичная функция: export_ls_to_yolo(...)

Структура выходной папки (Ultralytics YOLO):
    <out_root>/
        images/train/<video>__t*.jpg
        images/val/<video>__t*.jpg
        labels/train/<video>__t*.txt   # YOLO: class cx cy w h (норм. [0..1])
        labels/val/<video>__t*.txt
        data.yaml

Разделение train/val — video-wise:
    Все кадры одного видео уходят в val (если val_video совпадает с именем видео),
    остальные — в train.
    Спец. значение val_video="__nonexistent__" кладёт всё в train (используется,
    когда val формируется отдельно, напр. из организаторских видео).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict


CLASSES = ["pricetag"]


def parse_ls_export(path: Path) -> list[dict]:
    """LS отдаёт либо JSON-MIN (плоский список), либо JSON (вложенный)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for item in raw:
        # JSON-MIN: ключ "bbox" / "label" — поле RectangleLabels
        # JSON: вложено в "annotations" → "result"
        rects = []
        if "annotations" in item:
            ann = item["annotations"][0] if item["annotations"] else None
            results = ann.get("result", []) if ann else []
            for r in results:
                if r.get("type") != "rectanglelabels":
                    continue
                v = r["value"]
                rects.append({
                    "x_pct": v["x"], "y_pct": v["y"],
                    "w_pct": v["width"], "h_pct": v["height"],
                    "img_w": r["original_width"],
                    "img_h": r["original_height"],
                })
        else:
            # JSON-MIN
            for r in item.get("bbox", []) or item.get("label", []) or []:
                rects.append({
                    "x_pct": r["x"], "y_pct": r["y"],
                    "w_pct": r["width"], "h_pct": r["height"],
                    "img_w": r["original_width"],
                    "img_h": r["original_height"],
                })

        # Имя картинки: data.image может быть http://localhost:8081/<video>/t*.jpg
        image_url = item.get("data", {}).get("image", "") or item.get("image", "")
        # вычленяем '<video>/t*.jpg' с конца URL
        # пример: http://localhost:8081/1.mov/t005000.jpg → 1.mov/t005000.jpg
        parts = image_url.split("/")
        rel = "/".join(parts[-2:]) if len(parts) >= 2 else image_url
        out.append({"image_rel": rel, "rects": rects})
    return out


def export_ls_to_yolo(
    ls_export_path: Path,
    frames_root: Path,
    out_root: Path,
    val_video: str = "__nonexistent__",
) -> Dict[str, int]:
    """LS-export JSON → YOLO структура.

    Параметры
    ---------
    ls_export_path : Path
        JSON, выгруженный из Label Studio (Export → JSON или JSON-MIN).
    frames_root : Path
        Папка с исходными кадрами (labeling/frames).
        Ожидается структура: frames_root/<video>/t*.jpg
    out_root : Path
        Куда положить YOLO-датасет.
    val_video : str
        Имя видео (например "4.mov"), кадры которого уйдут в val.
        Специальное значение "__nonexistent__" (или любое имя, которого нет в
        экспорте) кладёт ВСЁ в train. Это нужно, когда val формируется
        отдельно (например, через import_organizer_videos).

    Возвращает
    ----------
    dict
        {"train": N, "val": M, "skipped": K}
        (никаких sys.exit — ошибки пробрасываются исключениями)
    """
    ls_export_path = Path(ls_export_path)
    frames_root = Path(frames_root)
    out_root = Path(out_root)

    items = parse_ls_export(ls_export_path)

    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    stats: Dict[str, int] = {"train": 0, "val": 0, "skipped": 0}

    from src.utils.progress import progress
    for it in progress(items, "ls_export"):
        rel = it["image_rel"]
        video = rel.split("/")[0] if "/" in rel else rel
        split = "val" if video == val_video else "train"

        src = frames_root / rel
        if not src.exists():
            stats["skipped"] += 1
            continue

        # уникальное имя файла (video__t*.jpg, чтобы не было коллизий)
        flat_name = rel.replace("/", "__")
        dst_img = out_root / f"images/{split}/{flat_name}"
        dst_lbl = out_root / f"labels/{split}/{flat_name.rsplit('.', 1)[0]}.txt"
        shutil.copy2(src, dst_img)

        # YOLO labels: LS даёт % от original_width/height → нормируем в [0..1]
        lines = []
        for r in it["rects"]:
            x_pct = r["x_pct"] / 100.0
            y_pct = r["y_pct"] / 100.0
            w_pct = r["w_pct"] / 100.0
            h_pct = r["h_pct"] / 100.0
            # cx, cy — центр bbox в долях изображения
            cx = x_pct + w_pct / 2.0
            cy = y_pct + h_pct / 2.0
            lines.append(f"0 {cx:.6f} {cy:.6f} {w_pct:.6f} {h_pct:.6f}")
        dst_lbl.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        stats[split] += 1

    # data.yaml — манифест для Ultralytics YOLO
    # path: . означает «папка рядом с data.yaml», независимо от cwd при обучении
    yaml_text = (
        "# Сгенерировано ls_export_to_yolo.py\n"
        "# Если нужно — замени `path:` на абсолютный путь целевой системы.\n"
        f"path: .\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    (out_root / "data.yaml").write_text(yaml_text, encoding="utf-8")

    return stats


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Конвертирует экспорт Label Studio в YOLO-датасет."
    )
    p.add_argument(
        "--ls-export",
        type=Path,
        default=Path("labeling/ls_export.json"),
        help="JSON, выгруженный из Label Studio",
    )
    p.add_argument(
        "--frames",
        type=Path,
        default=Path("labeling/frames"),
        help="папка с исходными кадрами (labeling/frames/<video>/t*.jpg)",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="куда положить YOLO-датасет",
    )
    p.add_argument(
        "--val-video",
        default="__nonexistent__",
        help='имя видео целиком в val; "__nonexistent__" = всё в train',
    )
    args = p.parse_args()

    s = export_ls_to_yolo(args.ls_export, args.frames, args.out, args.val_video)
    print(f"train={s['train']} val={s['val']} skipped={s['skipped']}")
