from __future__ import annotations
from typing import Any, Iterator, List, Optional, Tuple
import numpy as np
import cv2

from src.inference.detector import Track
from src.utils.metrics import ean13_checksum_ok


# Названия 7 базовых препроцессингов — порядок ВАЖЕН (тесты на счёт 7×4=28
# и debug-лог опираются на этот порядок).
PREPROC_NAMES: list[str] = [
    "gray", "equalize_hist", "adaptive_thresh", "adaptive_thresh_inv",
    "gaussian_blur", "upscale2x", "sharpen",
]
ROT_NAMES: list[str] = ["rot0", "rot90cw", "rot180", "rot90ccw"]


def preprocess_variants(img: np.ndarray) -> Iterator[np.ndarray]:
    """7 препроцессингов × 4 поворота = 28 вариантов (анонимный iterator)."""
    for _name, variant in preprocess_variants_named(img):
        yield variant


def preprocess_variants_named(img: np.ndarray) -> Iterator[Tuple[str, np.ndarray]]:
    """То же что preprocess_variants, но дополнительно отдаёт имя варианта.

    Имя формата `{preproc}_{rot}`, например `equalize_hist_rot90cw`.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    variants_base: List[np.ndarray] = [
        gray,
        cv2.equalizeHist(gray),
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 31, 5),
        cv2.bitwise_not(cv2.adaptiveThreshold(gray, 255,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 31, 5)),
        cv2.GaussianBlur(gray, (3, 3), 0),
        cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2),
                   interpolation=cv2.INTER_CUBIC),
        cv2.filter2D(gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])),
    ]
    rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180,
                 cv2.ROTATE_90_COUNTERCLOCKWISE]
    for preproc_name, v in zip(PREPROC_NAMES, variants_base):
        for rot_name, rot in zip(ROT_NAMES, rotations):
            full_name = f"{preproc_name}_{rot_name}"
            yield full_name, (cv2.rotate(v, rot) if rot is not None else v)


def decode_with_zxing(img: np.ndarray) -> Optional[str]:
    """Прогоняет ансамбль препроцессингов через zxing-cpp.

    Возвращает первый EAN-13 checksum-pass или None.
    """
    code, _ = decode_with_zxing_meta(img)
    return code


def decode_with_zxing_meta(img: np.ndarray) -> Tuple[Optional[str], dict]:
    """Как decode_with_zxing, но дополнительно отдаёт meta: какой препроцесс сработал.

    Возвращает (code | None, {"preproc": str, "tried": int}).
    """
    try:
        import zxingcpp
    except ImportError:
        return None, {"preproc": "", "tried": 0, "error": "zxingcpp_not_installed"}
    tried = 0
    for preproc_name, variant in preprocess_variants_named(img):
        tried += 1
        try:
            results = zxingcpp.read_barcodes(variant)
        except Exception:
            continue
        for r in results:
            text = (r.text or "").strip()
            if len(text) == 13 and text.isdigit() and ean13_checksum_ok(text):
                return text, {"preproc": preproc_name, "tried": tried}
    return None, {"preproc": "", "tried": tried}


def decode_barcode_track(
    track: Track,
    recorder: Any = None,
) -> Optional[str]:
    """Идёт по всем кадрам трека (по ORGANIZER §1) и пробует zxing.

    Останавливается на первом EAN-13 checksum-pass.

    Если передан `recorder` (debug-режим), пишет результат попыток в barcode.log.
    """
    if not track.frames:
        if recorder is not None and getattr(recorder, "enabled", False):
            recorder.log_barcode_attempt(track.track_id, 0, None, source="zxing",
                                         note="empty track")
        return None

    n_tried = 0
    last_preproc = ""
    for fd in track.frames:
        if fd.frame is None:
            continue
        # fd.frame теперь уже-кроп (rotate=0), используем напрямую.
        crop = fd.frame
        n_tried += 1
        code, meta = decode_with_zxing_meta(crop)
        last_preproc = meta.get("preproc", "")
        if code is not None:
            if recorder is not None and getattr(recorder, "enabled", False):
                recorder.log_barcode_attempt(
                    track.track_id, n_tried, code, source="zxing",
                    note=f"preproc={last_preproc}",
                )
            return code
    if recorder is not None and getattr(recorder, "enabled", False):
        recorder.log_barcode_attempt(
            track.track_id, n_tried, None, source="zxing",
            note="no_decode_in_any_frame",
        )
    return None


def barcode_vlm_fallback(
    track: Track,
    vlm_reader,
    top_k: int = 3,
    recorder: Any = None,
) -> Optional[str]:
    """Берём топ-K самых резких кадров → VLM → majority vote среди EAN-13-валидных."""
    if not track.frames:
        return None
    from collections import Counter

    def sharp(crop_arr):
        g = cv2.cvtColor(crop_arr, cv2.COLOR_BGR2GRAY) if crop_arr.ndim == 3 else crop_arr
        return cv2.Laplacian(g, cv2.CV_64F).var()

    scored = []
    for fd in track.frames:
        if fd.frame is None:
            continue
        # fd.frame уже-кроп. Поворачиваем под VLM (270° CCW).
        c = cv2.rotate(fd.frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        scored.append((sharp(c), c))
    scored.sort(key=lambda x: x[0], reverse=True)
    candidates: list[str] = []
    for _, c in scored[:top_k]:
        out = vlm_reader.read(c)
        bc = (out.get("barcode") or "").strip()
        if len(bc) == 13 and bc.isdigit() and ean13_checksum_ok(bc):
            candidates.append(bc)
    if not candidates:
        if recorder is not None and getattr(recorder, "enabled", False):
            recorder.log_barcode_attempt(
                track.track_id, min(top_k, len(scored)), None,
                source="vlm_fallback", note="no_valid_ean13_candidates",
            )
        return None
    winner = Counter(candidates).most_common(1)[0][0]
    if recorder is not None and getattr(recorder, "enabled", False):
        recorder.log_barcode_attempt(
            track.track_id, len(candidates), winner,
            source="vlm_fallback",
            note=f"votes={dict(Counter(candidates))}",
        )
    return winner
