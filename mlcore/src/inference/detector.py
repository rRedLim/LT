from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np


# Кастомный ByteTrack-конфиг: длинный track_buffer + низкий track_low_thresh,
# чтобы трек не рвался на motion blur (см. configs/bytetrack_v3.yaml).
_BYTETRACK_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "bytetrack_v3.yaml"
)


@dataclass
class FrameDet:
    """Single detection inside one frame of a track.

    `frame` хранит ТОЛЬКО область bbox (кроп), а не весь кадр, — экономия RAM
    ~50× (1920×1080 → ~200×400). Если нужен полный кадр (для debug-frame), его
    легко перечитать по `ts_ms`/`frame_idx`. Для pick_A (Laplacian sharpness)
    кроп достаточен.

    `touches_edge` = True, если bbox упёрся в край исходного кадра (то есть
    ценник физически обрезан рамкой видео). Такие детекции pick_A
    деприоритезирует: лучше взять полный читаемый ценник из соседнего кадра,
    чем половинку с большим количеством текста.
    """

    ts_ms: int
    bbox: tuple                          # (x_min, y_min, x_max, y_max) pixels in original frame
    conf: float
    frame: Optional[np.ndarray] = None  # bbox-кроп (НЕ весь кадр), если keep_frames=True
    frame_idx: int = 0                  # позиция кадра в видео (для повторного чтения)
    touches_edge: bool = False          # bbox прижат к краю кадра (обрезан)


@dataclass
class Track:
    """All detections belonging to one track_id across the full video."""

    track_id: int
    frames: List[FrameDet] = field(default_factory=list)


def detect_and_track(
    video_path: Path,
    yolo_weights: Path,
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 1280,
    keep_frames: bool = True,
    recorder: Any = None,
    debug_every_n_frames: int = 30,
) -> List[Track]:
    """Run YOLO + ByteTrack via a manual cv2 frame loop.

    Parameters
    ----------
    video_path:   Path to input video file.
    yolo_weights: Path to YOLO .pt weights file.
    conf:         Detection confidence threshold.
    iou:          NMS IoU threshold.
    imgsz:        Inference image size (longest side).
    keep_frames:  When True, stores the raw frame in FrameDet.frame (higher RAM
                  usage but convenient for downstream crops). When False,
                  frame=None and the caller must re-read frames if needed.
    recorder:     Optional DebugRecorder (or NoOp). When debug-enabled, every
                  Nth frame is saved with bbox+track_id overlays to
                  debug/<video>/frames_with_bbox/.
    debug_every_n_frames: How often to dump a frame (1 = every frame, 30 ≈ 1/sec at 30 fps).

    Returns
    -------
    List of Track objects, one per unique track_id observed in the video.
    """
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(yolo_weights))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    import time
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
    # Print progress every 5% (или каждые 100 кадров если total неизвестен)
    if total_frames:
        progress_step = max(1, total_frames // 20)
    else:
        progress_step = 100
    desc = f"YOLO+ByteTrack {Path(video_path).stem}"
    t_start = time.perf_counter()
    print(f"[{desc}] start, total_frames={total_frames}", flush=True)

    tracks: dict[int, Track] = {}
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if frame_idx % progress_step == 0:
                elapsed = time.perf_counter() - t_start
                if total_frames:
                    pct = int(frame_idx * 100 / total_frames)
                    rate = frame_idx / elapsed if elapsed > 0 else 0
                    eta = (total_frames - frame_idx) / rate if rate > 0 else 0
                    print(f"[{desc}] {frame_idx}/{total_frames} ({pct}%) "
                          f"{elapsed:.1f}s, ETA {eta:.1f}s, "
                          f"tracks={len(tracks)}", flush=True)
                else:
                    print(f"[{desc}] {frame_idx} frames, {elapsed:.1f}s, "
                          f"tracks={len(tracks)}", flush=True)

            results = model.track(
                frame,
                persist=True,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False,
                tracker=str(_BYTETRACK_CONFIG),
            )

            frame_detections: list[dict] = []
            for r in results:
                if r.boxes is None or r.boxes.id is None:
                    continue

                ids  = r.boxes.id.cpu().int().tolist()
                xyxy = r.boxes.xyxy.cpu().tolist()
                confs = r.boxes.conf.cpu().tolist()

                for tid, box, cf in zip(ids, xyxy, confs):
                    tr = tracks.setdefault(int(tid), Track(track_id=int(tid)))
                    # Сохраняем ТОЛЬКО bbox-кроп, не весь кадр. Экономия ~50× RAM.
                    # Если keep_frames=False — не сохраняем ничего.
                    # Padding с каждой стороны. YOLO часто отдаёт bbox только
                    # на цветную часть ценника (красная цена), а белая часть
                    # с названием/штрихкодом отрезается. Это последствие
                    # неконсистентной LS-разметки. Большой padding x по горизонтали
                    # (где соседний кусок ценника) и умеренный y по вертикали.
                    crop = None
                    touches_edge = False
                    if keep_frames:
                        x1, y1, x2, y2 = box
                        bw = x2 - x1; bh = y2 - y1
                        pad_x = bw * 0.25
                        pad_y = bh * 0.10
                        H, W = frame.shape[:2]
                        # Bbox считаем "обрезанным рамкой кадра", если ДО padding
                        # его сторона лежит ближе 3px к краю исходного кадра.
                        # Эти 3px нужны как минимальный зазор: даже если YOLO
                        # отдала "ровно у края", это всё равно обрезан.
                        edge_tol = 3
                        touches_edge = (
                            x1 < edge_tol or y1 < edge_tol
                            or x2 > W - edge_tol or y2 > H - edge_tol
                        )
                        x1i = max(0, int(x1 - pad_x)); y1i = max(0, int(y1 - pad_y))
                        x2i = min(W, int(x2 + pad_x)); y2i = min(H, int(y2 + pad_y))
                        if x2i > x1i and y2i > y1i:
                            crop = frame[y1i:y2i, x1i:x2i].copy()
                    tr.frames.append(
                        FrameDet(
                            ts_ms=ts_ms,
                            bbox=tuple(box),
                            conf=float(cf),
                            frame=crop,
                            frame_idx=frame_idx,
                            touches_edge=touches_edge,
                        )
                    )
                    frame_detections.append({
                        "track_id": int(tid),
                        "bbox": tuple(box),
                        "conf": float(cf),
                    })

            # Debug: save frame snapshot with overlays
            if recorder is not None and getattr(recorder, "enabled", False):
                recorder.save_frame_with_bboxes(
                    frame, ts_ms, frame_detections,
                    every_n_frames=debug_every_n_frames,
                )
    finally:
        cap.release()
        elapsed = time.perf_counter() - t_start
        print(f"[{desc}] DONE {frame_idx} frames in {elapsed:.1f}s, "
              f"tracks={len(tracks)}", flush=True)

    if recorder is not None and getattr(recorder, "enabled", False):
        recorder.log("detect_and_track: %d frames processed, %d tracks",
                     frame_idx, len(tracks))

    # Явно отпускаем YOLO+ByteTrack из VRAM. Без этого на 24 ГБ-карте Qwen2.5-VL
    # не влезает рядом, и accelerate offload-ит её слои на CPU.
    try:
        import torch
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass

    return list(tracks.values())
