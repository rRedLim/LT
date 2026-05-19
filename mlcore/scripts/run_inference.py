"""CLI: py scripts/run_inference.py video|batch ...

Флаги дебага:
  --debug                  включить DebugRecorder (артефакты в debug/<video_stem>/)
  --debug-dir PATH         корень для debug артефактов (по умолчанию ml_core/debug/)
  --debug-every-n-frames N сохранять каждый N-ый кадр с bbox оверлеями (по умолчанию 30)
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

from src.inference.pipeline import run


def cmd_video(args):
    r = run(
        args.video, args.yolo_weights, args.lora_adapter, args.out,
        smoke=args.smoke,
        debug=args.debug,
        debug_dir=args.debug_dir,
        debug_every_n_frames=args.debug_every_n_frames,
        skip_qr=args.skip_qr,
        qr_max_frames=args.qr_max_frames,
        skip_barcode_vlm_fallback=args.skip_barcode_vlm_fallback,
        max_pixels=args.max_pixels,
    )
    print(r)


def cmd_batch(args):
    import time
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for d in sorted([p for p in args.data.iterdir()
                     if p.is_dir() and p.name != "Unlabeled"]):
        found = None
        for ext in [".mp4", ".mov", ".MP4", ".MOV"]:
            v = d / f"{d.name}{ext}"
            if v.exists():
                found = v
                break
        if found is None:
            for ext in [".mp4", ".mov"]:
                files = sorted(d.glob(f"*{ext}"))
                if files:
                    found = files[0]
                    break
        if found is not None:
            candidates.append(found)
    print(f"\n=== Batch inference: {len(candidates)} videos ===\n", flush=True)
    t_total = time.perf_counter()
    for i, v in enumerate(candidates, 1):
        out = args.out_dir / f"{v.parent.name}.csv"
        print(f"[{i}/{len(candidates)}] {v.name} → {out.name}", flush=True)
        t0 = time.perf_counter()
        r = run(
            v, args.yolo_weights, args.lora_adapter, out,
            smoke=args.smoke,
            debug=args.debug,
            debug_dir=args.debug_dir,
            debug_every_n_frames=args.debug_every_n_frames,
            skip_qr=args.skip_qr,
            qr_max_frames=args.qr_max_frames,
            skip_barcode_vlm_fallback=args.skip_barcode_vlm_fallback,
            max_pixels=args.max_pixels,
        )
        print(f"   → tracks={r.get('tracks')} rows={r.get('rows')} "
              f"({time.perf_counter()-t0:.1f}s)\n", flush=True)
    print(f"=== Total batch time: {time.perf_counter()-t_total:.1f}s ===", flush=True)


def _add_common_args(parser):
    parser.add_argument("--yolo-weights", type=Path,
                        default=Path("runs/yolo/v3/weights/best.pt"))
    # LoRA пишется в runs/lora/v3 (см. lora_v3.yaml::out_dir),
    # после Trainer.save_model(out_dir/"final") итоговый адаптер — runs/lora/v3/final.
    parser.add_argument("--lora-adapter", type=Path,
                        default=Path("runs/lora/v3/final"))
    parser.add_argument("--smoke", action="store_true",
                        help="Только YOLO+ByteTrack, без VLM (smoke I/O).")
    parser.add_argument("--debug", action="store_true",
                        help="Сохранять артефакты дебага в debug/<video_stem>/.")
    parser.add_argument("--debug-dir", type=Path, default=None,
                        help="Корень debug-папки (по умолчанию ml_core/debug/).")
    parser.add_argument("--debug-every-n-frames", type=int, default=30,
                        help="Сохранять каждый N-ый кадр с bbox оверлеями.")
    # ── Speed knobs ─────────────────────────────────────────────────────
    parser.add_argument("--skip-qr", action="store_true",
                        help="Пропустить QR-декодинг (экономит ~5-15с/трек).")
    parser.add_argument("--qr-max-frames", type=int, default=3,
                        help="Сколько кадров трека пробовать для QR (по умолчанию 3).")
    parser.add_argument("--skip-barcode-vlm-fallback", action="store_true",
                        default=True,
                        help="Не делать VLM-fallback для barcode (экономит ~15с/трек "
                             "когда zxing промахнулся; обычно zxing справляется).")
    parser.add_argument("--max-pixels", type=int, default=800_000,
                        help="Макс пикселей картинки для VLM (по умолчанию 800K "
                             "= ~2x быстрее чем default 1.6M; на ценниках качество "
                             "почти не падает).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("video")
    s.add_argument("--video", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    _add_common_args(s)
    s.set_defaults(func=cmd_video)

    b = sub.add_parser("batch")
    b.add_argument("--data", type=Path,
                   default=Path("../dataset/dataset_orig"))
    b.add_argument("--out-dir", type=Path, default=Path("outputs"))
    _add_common_args(b)
    b.set_defaults(func=cmd_batch)

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.func(args)
