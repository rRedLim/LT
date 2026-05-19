from typing import Optional, Tuple
import numpy as np
import cv2


def extract_crop(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    rotate: int = 270,
    padding: int = 0,
) -> Optional[np.ndarray]:
    """Извлекает кроп из кадра по bbox (x_min, y_min, x_max, y_max) и поворачивает.

    Параметры
    ----------
    frame : np.ndarray
        Исходный кадр HxWxC.
    bbox : tuple
        (x_min, y_min, x_max, y_max) в пикселях исходного кадра.
    rotate : int
        Угол поворота по часовой стрелке после кропа: 0, 90, 180, 270.
        Для ценников Lenta — 270 (CCW 90°).
    padding : int
        Расширение bbox в пикселях (НЕ использовать >0 — bench показал, что
        VLM начинает галлюцинировать соседние товары).
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(round(x1 - padding)))
    y1 = max(0, int(round(y1 - padding)))
    x2 = min(W, int(round(x2 + padding)))
    y2 = min(H, int(round(y2 + padding)))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2].copy()
    if rotate == 90:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        crop = cv2.rotate(crop, cv2.ROTATE_180)
    elif rotate == 270:
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return crop
