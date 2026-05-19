import numpy as np
from src.utils.crop import extract_crop

def test_extract_crop_no_padding():
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    frame[100:200, 300:400] = 255
    crop = extract_crop(frame, bbox=(300, 100, 400, 200), rotate=0)
    assert crop.shape == (100, 100, 3)
    assert crop[50, 50, 0] == 255

def test_extract_crop_rotate_270():
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    frame[100:200, 300:500] = 255  # 100 высота × 200 ширина
    crop = extract_crop(frame, bbox=(300, 100, 500, 200), rotate=270)
    # после поворота 270° CCW: было (h=100, w=200), стало (h=200, w=100)
    assert crop.shape == (200, 100, 3)

def test_extract_crop_clamps_to_frame():
    """bbox выходит за границы кадра — clamping без падения"""
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    crop = extract_crop(frame, bbox=(450, 450, 600, 600), rotate=0)
    assert crop.shape == (50, 50, 3)

def test_extract_crop_invalid_bbox_returns_none():
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    assert extract_crop(frame, bbox=(100, 100, 100, 100), rotate=0) is None
    assert extract_crop(frame, bbox=(100, 100, 50, 50), rotate=0) is None
