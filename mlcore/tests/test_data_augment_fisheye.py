import numpy as np
from src.data.augment_fisheye import apply_fisheye_image, distort_bbox, STRENGTH_PRESETS

def test_apply_fisheye_returns_same_shape():
    img = np.full((1000, 1000, 3), 128, dtype=np.uint8)
    out = apply_fisheye_image(img, strength=STRENGTH_PRESETS["medium"])
    assert out.shape == img.shape

def test_distort_bbox_center_stays_close_to_center():
    """Центр кадра при дисторсии почти не двигается."""
    H, W = 1000, 1000
    bbox = (450, 450, 550, 550)
    new = distort_bbox(bbox, W, H, strength=STRENGTH_PRESETS["medium"])
    cx_old = (bbox[0] + bbox[2]) / 2
    cy_old = (bbox[1] + bbox[3]) / 2
    cx_new = (new[0] + new[2]) / 2
    cy_new = (new[1] + new[3]) / 2
    assert abs(cx_new - cx_old) < 20
    assert abs(cy_new - cy_old) < 20

def test_distort_bbox_edge_moves_more():
    """bbox у края при сильной дисторсии смещается заметно."""
    H, W = 1000, 1000
    bbox = (10, 10, 110, 110)
    new = distort_bbox(bbox, W, H, strength=STRENGTH_PRESETS["strong"])
    cx_new = (new[0] + new[2]) / 2
    # центр должен сместиться внутрь (растягивание у краёв)
    assert cx_new > 20
