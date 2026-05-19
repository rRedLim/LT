import numpy as np
import cv2
import pytest
from src.inference.barcode import preprocess_variants, decode_with_zxing
from src.utils.metrics import ean13_checksum_ok


def test_preprocess_variants_count():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    variants = list(preprocess_variants(img))
    assert len(variants) == 28  # 7 × 4


def test_ean13_checksum_valid():
    assert ean13_checksum_ok("4670025474665")


def test_decode_with_zxing_synthetic_doesnt_crash():
    """zxing на бессмысленной картинке должен вернуть None, не упасть."""
    try:
        import zxingcpp  # noqa
    except ImportError:
        pytest.skip("zxing-cpp not installed")
    img = np.full((200, 400, 3), 255, dtype=np.uint8)
    for i, x in enumerate(range(50, 350, 8)):
        if i % 2 == 0:
            cv2.rectangle(img, (x, 50), (x+4, 150), (0, 0, 0), -1)
    result = decode_with_zxing(img)
    assert result is None or (isinstance(result, str) and len(result) == 13)
