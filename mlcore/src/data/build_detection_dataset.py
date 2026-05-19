"""Сборка финального YOLO-датасета detection_v3.

Источники:
 1. ORGANIZER ВСЕ видео (5 шт): 25_12-20, 25_2-10, 26_12-20, 49_5, 43_15 —
    fisheye-кадры с робота, уже с fisheye-эффектом.
 2. LS-export наших 13 видео (dataset_myself) — iPhone, БЕЗ fisheye.
 3. Fisheye-аугментация ТОЛЬКО для наших 13 (чтобы они «выглядели» как орг).
    На орг повторно НЕ применяем — у них fisheye уже есть.
 4. Motion-аугментация — тоже только для наших (имитация движения робота).
 5. ВAL = random 20% от train (по картинкам). 43_15.csv размечен крайне
    скудно (29 строк на 2 timestamp'a), поэтому отдельным val он бесполезен —
    добавляем его в train, а split делаем случайным.

Структура output:
    out_root/
        images/train/, images/val/
        labels/train/, labels/val/
        data.yaml
"""
from __future__ import annotations

import logging
import random
import shutil
import time
from pathlib import Path

import yaml

from src.data.import_organizer_videos import import_video
from src.data.ls_export_to_yolo import export_ls_to_yolo
from src.data.augment_fisheye import augment_yolo_dataset as augment_fisheye_dataset
from src.data.augment_motion import augment_motion_dataset


log = logging.getLogger(__name__)


# Все 5 видео орг идут в train. Val формируется random-split'ом (см. ниже).
ORG_ALL = ["25_12-20", "25_2-10", "26_12-20", "49_5", "43_15"]


def _copy_files(src_dir: Path, dst_dir: Path, pattern: str = "*") -> int:
    """Копирует все файлы по pattern из src_dir в dst_dir. Возвращает счётчик."""
    if not src_dir.exists():
        return 0
    n = 0
    for p in src_dir.glob(pattern):
        if p.is_file():
            shutil.copy2(p, dst_dir / p.name)
            n += 1
    return n


def _step_header(n: int, total: int, title: str) -> None:
    """Печатает заголовок этапа в формате `[i/N] title…`."""
    msg = f"[{n}/{total}] {title}"
    print(msg, flush=True)
    log.info(msg)


def build(
    organizer_root: Path,
    ls_export: Path,
    frames_root: Path,
    out_root: Path,
    fisheye_strengths: tuple[str, ...] = (),
    motion_copies: int = 0,
    val_frac: float = 0.2,
    split_seed: int = 42,
) -> dict:
    """Собирает detection_v3.

    Параметры
    ---------
    organizer_root: путь к ../dataset/dataset_orig/ — содержит подпапки <video>/.
    ls_export:      labeling/ls_export.json (наша разметка 13 видео).
    frames_root:    labeling/frames/ (кадры из 13 видео).
    out_root:       куда сохранять detection_v3.
    fisheye_strengths: какие силы fisheye применять к нашим.
    motion_copies:  сколько мотион-копий каждого нашего кадра.
    val_frac:       доля train, которая случайно уйдёт в val (по картинкам).
    split_seed:     seed для воспроизводимости split'а.
    """
    n_org_all = len([v for v in ORG_ALL
                     if (organizer_root / v / f"{v}.mp4").exists()])
    has_ls = ls_export.exists() and frames_root.exists()

    # Этапы: import org (1) + ls_export (1) + fisheye × N + motion (если есть)
    #       + copy originals (1) + train/val split (1) + write data.yaml (1)
    n_steps = 1 + 1 + len(fisheye_strengths) + (1 if motion_copies > 0 else 0) + 1 + 1 + 1
    cur = 0
    t_start = time.perf_counter()
    print(f"\n=== build_detection_dataset → {out_root} ===", flush=True)
    print(f"   organizer_root: {organizer_root}  (all={n_org_all})", flush=True)
    print(f"   ls_export:      {ls_export}  (exists={has_ls})", flush=True)
    print(f"   fisheye:        {list(fisheye_strengths) or 'off'}", flush=True)
    print(f"   motion_copies:  {motion_copies}", flush=True)
    print(f"   val_frac:       {val_frac}  (random split от train, seed={split_seed})",
          flush=True)
    print(f"   total steps:    {n_steps}\n", flush=True)

    stats: dict = {}
    train_img = out_root / "images" / "train"
    train_lbl = out_root / "labels" / "train"
    val_img = out_root / "images" / "val"
    val_lbl = out_root / "labels" / "val"
    for d in [train_img, train_lbl, val_img, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Organizer ВСЕ видео в train (fisheye уже есть, повторно не аугментируем) ──
    cur += 1
    _step_header(cur, n_steps, f"Importing {len(ORG_ALL)} organizer videos to train…")
    t0 = time.perf_counter()
    s_org_train = 0
    for v in ORG_ALL:
        vid_path = organizer_root / v / f"{v}.mp4"
        csv_path = organizer_root / v / f"{v}.csv"
        if not (vid_path.exists() and csv_path.exists()):
            print(f"   SKIP {v}: video or csv missing", flush=True)
            continue
        t_v = time.perf_counter()
        n = import_video(
            video_path=vid_path,
            csv_path=csv_path,
            out_images=train_img, out_labels=train_lbl,
            window_ms=500,
        )
        s_org_train += n
        print(f"   {v}: {n} pairs ({time.perf_counter()-t_v:.1f}s)", flush=True)
    stats["organizer_pairs"] = s_org_train
    print(f"   → {s_org_train} pairs total ({time.perf_counter()-t0:.1f}s)\n", flush=True)

    # ── 3. LS-export наших 13 видео ──────────────────────────────────────────
    cur += 1
    _step_header(cur, n_steps, "Exporting LS labels (our 13 videos)…")
    t0 = time.perf_counter()
    ours_tmp = out_root / "_tmp_ours"
    ours_img = ours_tmp / "images" / "train"
    ours_lbl = ours_tmp / "labels" / "train"
    for d in [ours_img, ours_lbl]:
        d.mkdir(parents=True, exist_ok=True)
    if has_ls:
        s_ls = export_ls_to_yolo(ls_export, frames_root, ours_tmp,
                                 val_video="__nonexistent__")
        stats["ls_pairs"] = s_ls.get("train", 0)
        print(f"   → {stats['ls_pairs']} pairs from LS (skipped={s_ls.get('skipped', 0)}, {time.perf_counter()-t0:.1f}s)\n",
              flush=True)
    else:
        stats["ls_pairs"] = 0
        print(f"   → SKIP: ls_export or frames not found\n", flush=True)

    # ── 4. Fisheye-аугментация — только для наших ────────────────────────────
    s_fish = 0
    for strength in fisheye_strengths:
        cur += 1
        _step_header(cur, n_steps, f"Fisheye augmentation (strength={strength})…")
        t0 = time.perf_counter()
        fish_tmp = out_root / f"_tmp_fish_{strength}"
        n_added = augment_fisheye_dataset(
            ours_img, ours_lbl,
            fish_tmp / "images", fish_tmp / "labels",
            strength_name=strength, suffix=f"_fish_{strength}",
        )
        s_fish += n_added
        _copy_files(fish_tmp / "images", train_img)
        _copy_files(fish_tmp / "labels", train_lbl)
        shutil.rmtree(fish_tmp, ignore_errors=True)
        print(f"   → +{n_added} pairs ({time.perf_counter()-t0:.1f}s)\n", flush=True)
    stats["fisheye_pairs"] = s_fish

    # ── 5. Motion-аугментация — тоже только для наших ────────────────────────
    s_motion = 0
    if motion_copies > 0:
        cur += 1
        _step_header(cur, n_steps, f"Motion-blur augmentation (×{motion_copies})…")
        t0 = time.perf_counter()
        motion_tmp = out_root / "_tmp_motion"
        s_motion = augment_motion_dataset(
            ours_img, ours_lbl,
            motion_tmp / "images", motion_tmp / "labels",
            n_copies=motion_copies,
        )
        _copy_files(motion_tmp / "images", train_img)
        _copy_files(motion_tmp / "labels", train_lbl)
        shutil.rmtree(motion_tmp, ignore_errors=True)
        print(f"   → +{s_motion} pairs ({time.perf_counter()-t0:.1f}s)\n", flush=True)
    stats["motion_pairs"] = s_motion

    # ── 6. Наконец, копируем ОРИГИНАЛЫ наших 13 в финальный train ───────────
    cur += 1
    _step_header(cur, n_steps, "Copying originals of our 13 videos to final train…")
    t0 = time.perf_counter()
    n_ours = _copy_files(ours_img, train_img)
    _copy_files(ours_lbl, train_lbl)
    stats["ours_orig_pairs"] = n_ours
    shutil.rmtree(ours_tmp, ignore_errors=True)
    print(f"   → +{n_ours} originals ({time.perf_counter()-t0:.1f}s)\n", flush=True)

    # ── 7. Random split: val_frac% от train → val ────────────────────────────
    cur += 1
    _step_header(cur, n_steps,
                 f"Random split train→val (val_frac={val_frac}, seed={split_seed})…")
    t0 = time.perf_counter()
    all_train_imgs = sorted(train_img.glob("*.jpg"))
    rng = random.Random(split_seed)
    rng.shuffle(all_train_imgs)
    n_val = int(round(len(all_train_imgs) * val_frac))
    moved = 0
    for img_path in all_train_imgs[:n_val]:
        lbl_path = train_lbl / f"{img_path.stem}.txt"
        shutil.move(str(img_path), str(val_img / img_path.name))
        if lbl_path.exists():
            shutil.move(str(lbl_path), str(val_lbl / lbl_path.name))
        moved += 1
    stats["val_split_pairs"] = moved
    print(f"   → moved {moved} pairs to val ({time.perf_counter()-t0:.1f}s)\n",
          flush=True)

    # ── 8. data.yaml + финальный count ───────────────────────────────────────
    cur += 1
    _step_header(cur, n_steps, "Writing data.yaml + counting final pairs…")
    data_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["pricetag"],
    }
    with open(out_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    stats["final_train"] = len(list(train_img.glob("*.jpg")))
    stats["final_val"] = len(list(val_img.glob("*.jpg")))
    print(f"   → train={stats['final_train']}, val={stats['final_val']}\n", flush=True)

    print(f"=== Done in {time.perf_counter()-t_start:.1f}s ===\n", flush=True)
    return stats


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--organizer-root", type=Path,
                   default=Path("../dataset/dataset_orig"))
    p.add_argument("--ls-export", type=Path,
                   default=Path("../labeling/ls_export.json"))
    p.add_argument("--frames", type=Path,
                   default=Path("../labeling/frames"))
    p.add_argument("--out", type=Path, default=Path("datasets/detection_v3"))
    # ВНИМАНИЕ: fisheye/motion аугментация по умолчанию ВЫКЛЮЧЕНА.
    # Мой synthetic fisheye не похож на реальный 140°-HFOV робота:
    # синтетика сжимает контент к центру, теряет полки по краям, bbox висят
    # в воздухе. Хуже модели чем реальные кадры без аугментации.
    # Если есть желание поэкспериментировать — `--fisheye-strengths weak`.
    p.add_argument("--fisheye-strengths", nargs="*",
                   default=[],
                   choices=["weak", "medium", "strong"])
    p.add_argument("--motion-copies", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="Доля train → val (random split по картинкам)")
    p.add_argument("--split-seed", type=int, default=42,
                   help="Seed для воспроизводимости val-split'а")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = build(
        args.organizer_root, args.ls_export, args.frames, args.out,
        fisheye_strengths=tuple(args.fisheye_strengths),
        motion_copies=args.motion_copies,
        val_frac=args.val_frac,
        split_seed=args.split_seed,
    )
    for k, v in s.items():
        print(f"  {k}: {v}")
