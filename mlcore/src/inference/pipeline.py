from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import logging

from src.inference.detector import detect_and_track, Track
from src.inference.frame_picker import pick_A
from src.inference.vlm_reader import VLMReader, FIELDS
from src.inference.barcode import decode_barcode_track, barcode_vlm_fallback
from src.inference.template_router import TemplateRouter
from src.inference.csv_writer import write_csv, COLUMNS
from src.data.qr_decode import (
    decode_qr_from_image, parse_qr_payload, ALL_QR_FIELDS,
)
from src.utils.crop import extract_crop
from src.utils.debug import make_recorder
from src.utils.metrics import ean13_checksum_ok


log = logging.getLogger(__name__)


# Резолвинг configs относительно ml_core/ а НЕ от CWD (фикс ревью #C-2).
_ML_CORE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GALLERY = _ML_CORE_ROOT / "configs" / "gallery.json"
DEFAULT_ARCHETYPE = _ML_CORE_ROOT / "configs" / "archetype_schema.yaml"


def _video_filename_in_organizer_format(video_path: Path) -> str:
    """Возвращает строку filename в формате эталона: '<videodir>/<video>.mp4'.

    Эталон: '25_12-20/2.mp4'. То есть: имя папки-родителя + имя файла.
    Если родительская папка отсутствует — просто имя файла.
    """
    parent = video_path.parent.name
    return f"{parent}/{video_path.name}" if parent else video_path.name


def _row_from_picked(
    video_path: Path,
    picked,
    fields_dict: dict,
) -> dict:
    """Создаёт CSV-строку с заполненными filename/ts/bbox + поля из fields_dict."""
    x1, y1, x2, y2 = picked.bbox
    row = {c: "" for c in COLUMNS}
    row.update({
        "filename": _video_filename_in_organizer_format(video_path),
        "frame_timestamp": picked.ts_ms,
        "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
    })
    for k in fields_dict:
        if k in row:                   # обновляем только колонки эталона
            row[k] = fields_dict[k]
    return row


def run(
    video_path: Path,
    yolo_weights: Path,
    lora_adapter: Path,
    out_csv: Path,
    *,
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    gallery_path: Optional[Path] = None,
    archetype_schema: Optional[Path] = None,
    rotate: int = 270,
    smoke: bool = False,
    debug: bool = False,
    debug_dir: Optional[Path] = None,
    debug_every_n_frames: int = 30,
    skip_barcode_vlm_fallback: bool = False,
    qr_max_frames: int = 3,
    skip_qr: bool = False,
    max_pixels: int = 800_000,
) -> dict:
    """End-to-end: video.mp4 → CSV в формате организаторов.

    Параметры
    ---------
    video_path:   путь к видео.
    yolo_weights: путь к YOLO weights (.pt).
    lora_adapter: путь к LoRA адаптеру. Если не существует — VLM грузится без LoRA.
    out_csv:      куда писать финальный CSV (29 колонок).
    gallery_path: путь к configs/gallery.json. None → берётся ml_core/configs/.
    archetype_schema: путь к configs/archetype_schema.yaml. None → ml_core/configs/.
    rotate:       угол поворота кропа перед VLM (270 = CCW90 для ценников Лента).
    smoke:        True → только YOLO+ByteTrack, без VLM, для проверки I/O.
    debug:        True → создаётся DebugRecorder и сохраняет артефакты в debug_dir/<stem>/.
    debug_dir:    корень debug-папки. По умолчанию ml_core/debug/.
    debug_every_n_frames: как часто сохранять frame_with_bbox (30 ≈ 1/сек при 30fps).
    """
    # Резолвинг configs (фикс ревью #C-2)
    if gallery_path is None:
        gallery_path = DEFAULT_GALLERY
    if archetype_schema is None:
        archetype_schema = DEFAULT_ARCHETYPE
    if debug_dir is None:
        debug_dir = _ML_CORE_ROOT / "debug"

    # DebugRecorder (NoOp если debug=False — нулевая стоимость)
    recorder = make_recorder(debug, debug_root=debug_dir)
    recorder.setup(video_path.stem)
    recorder.log("=== run() %s ===", video_path.name)
    recorder.log("yolo_weights=%s lora_adapter=%s", yolo_weights, lora_adapter)
    recorder.log("gallery_path=%s archetype_schema=%s", gallery_path, archetype_schema)
    recorder.log("smoke=%s rotate=%s", smoke, rotate)

    log.info("YOLO+ByteTrack on %s", video_path.name)
    with recorder.timer("detect_and_track"):
        tracks = detect_and_track(
            video_path, yolo_weights, keep_frames=True,
            recorder=recorder, debug_every_n_frames=debug_every_n_frames,
        )
    log.info("Got %d tracks", len(tracks))
    recorder.log("got %d tracks", len(tracks))

    # Принудительно освобождаем VRAM от YOLO+ByteTrack. На 4090 (24 ГБ) YOLO
    # держит ~17 ГБ; если её не отпустить, Qwen2.5-VL-7B bf16 (~16 ГБ) не
    # влезет, и accelerate начнёт offload-ить слои на CPU → инференс становится
    # медленнее в десятки раз (343s/трек вместо 5s/трек).
    try:
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass

    if smoke:
        rows = []
        with recorder.timer("smoke_rows"):
            for tr in tracks:
                picked = pick_A(tr)
                if picked is None:
                    continue
                rows.append(_row_from_picked(video_path, picked, {}))
        with recorder.timer("write_csv"):
            write_csv(rows, out_csv)
        result = {"tracks": len(tracks), "rows": len(rows),
                  "out": str(out_csv), "smoke": True}
        recorder.save_summary(result)
        recorder.close()
        return result

    # Полный inference
    with recorder.timer("vlm_load"):
        reader = VLMReader(
            base_model=base_model,
            adapter=lora_adapter if lora_adapter.exists() else None,
            gallery_path=gallery_path,
            max_pixels=max_pixels,
        )
    if not lora_adapter.exists():
        recorder.warning("LoRA adapter %s не существует — VLM без LoRA", lora_adapter)

    if not archetype_schema.exists():
        raise FileNotFoundError(
            f"archetype_schema.yaml not found at {archetype_schema}"
        )
    router = TemplateRouter(archetype_schema)

    rows: list[dict] = []
    n_barcode_zxing = 0
    n_barcode_vlm = 0
    n_barcode_empty = 0
    n_qr_decoded = 0

    from src.utils.progress import progress
    for tr in progress(tracks, f"infer {video_path.stem}"):
        picked = pick_A(tr)
        if picked is None or picked.frame is None:
            recorder.debug("track %d: no picked frame, skipping", tr.track_id)
            continue
        # picked.frame теперь содержит УЖЕ-кроп (см. detector.py).
        # Применяем только rotate.
        import cv2 as _cv2
        crop = picked.frame
        if rotate == 90:
            crop = _cv2.rotate(crop, _cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 180:
            crop = _cv2.rotate(crop, _cv2.ROTATE_180)
        elif rotate == 270:
            crop = _cv2.rotate(crop, _cv2.ROTATE_90_COUNTERCLOCKWISE)
        if crop is None or crop.size == 0:
            recorder.debug("track %d: empty crop, skipping", tr.track_id)
            continue

        # VLM с raw для дебага
        with recorder.timer("vlm_read"):
            vlm_out, vlm_raw = reader.read_with_raw(crop)

        # Barcode: приоритет 1 = zxing на всех кадрах
        with recorder.timer("barcode_zxing"):
            bc = decode_barcode_track(tr, recorder=recorder)
        if bc:
            n_barcode_zxing += 1
            bc_source = "zxing"
        else:
            bc_vlm = (vlm_out.get("barcode") or "").strip()
            if len(bc_vlm) == 13 and bc_vlm.isdigit() and ean13_checksum_ok(bc_vlm):
                bc = bc_vlm
                bc_source = "vlm_inline"
            elif skip_barcode_vlm_fallback:
                # prefill-режим: fallback не нужен, штрихкод правится в Excel
                bc = ""
                n_barcode_empty += 1
                bc_source = "skipped_fallback"
            else:
                with recorder.timer("barcode_vlm_fallback"):
                    bc = barcode_vlm_fallback(tr, reader, recorder=recorder) or ""
                if bc:
                    n_barcode_vlm += 1
                    bc_source = "vlm_fallback"
                else:
                    n_barcode_empty += 1
                    bc_source = "none"
        vlm_out["barcode"] = bc

        # QR-декодинг: пытаемся декодить QR ТОЛЬКО на picked-кадре и
        # `qr_max_frames`-1 соседних (по убыванию sharpness). На все кадры
        # трека брать слишком долго (50 кадров × 28 preproc per zxing = 1400
        # zxing-вызовов на трек). QR — источник правды для 11 полей.
        qr_payload = ""
        qr_source = "none"
        if not skip_qr:
            with recorder.timer("qr_decode"):
                candidate_frames = []
                if picked.frame is not None:
                    candidate_frames.append(picked.frame)
                # Добавляем ещё до qr_max_frames-1 кадров трека равномерно
                if qr_max_frames > 1:
                    step = max(1, len(tr.frames) // max(1, qr_max_frames - 1))
                    for fd in tr.frames[::step][:qr_max_frames - 1]:
                        if fd.frame is not None and fd.frame.size > 0:
                            candidate_frames.append(fd.frame)
                for img in candidate_frames[:qr_max_frames]:
                    payload, source = decode_qr_from_image(img)
                    if payload:
                        qr_payload = payload
                        qr_source = source
                        break
        if qr_payload:
            n_qr_decoded += 1
            qr_fields = parse_qr_payload(qr_payload)
            for f in ALL_QR_FIELDS:
                v = qr_fields.get(f)
                if v:
                    vlm_out[f] = v
            # Если из QR пришёл qr_code_barcode, дублируем в barcode (когда оно
            # пустое или совпадает — это один и тот же EAN-13).
            qr_bc = qr_fields.get("qr_code_barcode", "")
            if qr_bc and not vlm_out.get("barcode"):
                vlm_out["barcode"] = qr_bc
            recorder.debug("track %d: QR decoded via %s, %d fields filled",
                           tr.track_id, qr_source, sum(1 for f in ALL_QR_FIELDS
                                                       if qr_fields.get(f)))

        with recorder.timer("template_route"):
            routed = router.route(vlm_out)

        row = _row_from_picked(video_path, picked, routed)
        rows.append(row)

        recorder.debug(
            "track %d: ts=%dms bbox=%s sharpness_picked, barcode=%s(%s)",
            tr.track_id, picked.ts_ms, picked.bbox, bc, bc_source,
        )
        recorder.save_track_artifacts(
            tr.track_id,
            pick_a_frame=picked.frame,
            bbox=picked.bbox,
            crop_for_vlm=crop,
            vlm_raw=vlm_raw,
            vlm_parsed=vlm_out,
            final_row=row,
        )

    with recorder.timer("write_csv"):
        write_csv(rows, out_csv)

    result = {
        "tracks": len(tracks),
        "rows": len(rows),
        "out": str(out_csv),
        "barcode": {
            "zxing": n_barcode_zxing,
            "vlm_fallback": n_barcode_vlm,
            "empty": n_barcode_empty,
        },
        "qr_decoded": n_qr_decoded,
    }
    recorder.save_summary(result)
    recorder.close()
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--yolo-weights", type=Path,
                   default=Path("runs/yolo/v3/weights/best.pt"))
    # LoRA сохраняется через Trainer.save_model в runs/lora/v3/final.
    p.add_argument("--lora-adapter", type=Path,
                   default=Path("runs/lora/v3/final"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--debug", action="store_true",
                   help="Сохранять артефакты дебага в debug/<video_stem>/.")
    p.add_argument("--debug-dir", type=Path, default=None)
    p.add_argument("--debug-every-n-frames", type=int, default=30,
                   help="Каждый N-ый кадр с bbox оверлеями. 1 = все кадры.")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    print(run(
        args.video, args.yolo_weights, args.lora_adapter, args.out,
        smoke=args.smoke, debug=args.debug,
        debug_dir=args.debug_dir,
        debug_every_n_frames=args.debug_every_n_frames,
    ))
