from __future__ import annotations
from typing import Optional
import cv2
import numpy as np

from src.inference.detector import Track, FrameDet
from src.utils.crop import extract_crop


# Минимальная короткая сторона кропа в пикселях.
_MIN_SIDE_PX = 60

# Порог |grad| для пикселя, чтобы считать его "текстовым".
# Эмпирически: на резком тексте Sobel-magnitude 80-300, на blur < 50.
_TEXT_GRAD_THRESHOLD = 80.0


def _sharpness(crop: np.ndarray) -> float:
    """Оценка количества читаемого текста в кропе.

    Метрика: count(|grad| > 80) — абсолютное число "текстовых" пикселей
    во всём кропе (НЕ доля, НЕ percentile, НЕ только центр).

    Почему так:
      - Резкий полный ценник 800×400 с текстом на 15% площади даёт ~48k
        текстовых пикселей.
      - Резкий обрезанный кусок цены 300×200 с цифрами на 30% площади —
        ~18k пикселей. То есть полный кадр (даже при меньшей плотности
        текста) выигрывает за счёт большей площади.
      - Размытый кадр любого размера даёт <5k.
      - Это автоматически предпочитает полный читаемый кроп над частичным,
        и любой читаемый — над любым blur.

    Если короткая сторона < 60px → 0 (ценник слишком далеко).
    """
    if crop is None or crop.size == 0:
        return 0.0
    h, w = crop.shape[:2]
    if min(h, w) < _MIN_SIDE_PX:
        return 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float((mag > _TEXT_GRAD_THRESHOLD).sum())


def pick_A(track: Track) -> Optional[FrameDet]:
    """Возвращает кадр трека с максимальным количеством читаемого текста.

    Многоступенчатый отбор:
      1. Только bbox шириной ≥ 80% от max-ширины bbox в треке. Это отсекает
         кадры, где ценник частично обрезан рамкой видео или YOLO зацепил
         только цветную половину. Такие "узкие" bbox в одном и том же треке —
         надёжный признак неполноты, потому что физический ценник один и тот же.
      2. Среди этих — только НЕ упирающиеся в край кадра (touches_edge=False).
      3. Среди этих — максимум по sharpness.
      4. Если ни один фильтр ничего не оставил — fallback на любые.

    Если трек пуст — возвращает None.
    """
    if not track.frames:
        return None

    # Для отсечения "узких" кадров (где YOLO зацепил половину ценника или
    # ценник наполовину уехал за рамку видео) используем PERCENTILE-75 площади
    # bbox в треке.
    #   - max было плохо: один кадр-выброс с раздутым bbox (YOLO накинула
    #     рамку на пустое место) поднимает max и пропускает обрезанные.
    #   - percentile-75 более устойчив. Если в треке 40 кадров и 10 из них
    #     полные ~720k, а 30 обрезанные ~400k, то p75 ≈ 720k. Фильтр
    #     "площадь ≥ 0.75 * p75" пропустит только полные.
    areas = []
    for fd in track.frames:
        if fd.frame is None:
            continue
        x1, y1, x2, y2 = fd.bbox
        areas.append((x2 - x1) * (y2 - y1))
    if not areas:
        return None
    p75_area = float(np.percentile(areas, 75))
    area_threshold = 0.75 * p75_area

    def _best_of(predicate) -> Optional[FrameDet]:
        best: Optional[FrameDet] = None
        best_score = -1.0
        for fd in track.frames:
            if fd.frame is None:
                continue
            if not predicate(fd):
                continue
            s = _sharpness(fd.frame)
            if s > best_score:
                best_score = s
                best = fd
        return best

    def _is_full_size(fd: FrameDet) -> bool:
        x1, y1, x2, y2 = fd.bbox
        return (x2 - x1) * (y2 - y1) >= area_threshold

    # 1. Полный размер + не у края
    picked = _best_of(lambda fd: _is_full_size(fd) and not fd.touches_edge)
    if picked is not None:
        return picked
    # 2. Полный размер (даже если у края)
    picked = _best_of(_is_full_size)
    if picked is not None:
        return picked
    # 3. Не у края
    picked = _best_of(lambda fd: not fd.touches_edge)
    if picked is not None:
        return picked
    # 4. Fallback — любые
    picked = _best_of(lambda fd: True)
    return picked or track.frames[0]
