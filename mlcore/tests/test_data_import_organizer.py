from src.data.import_organizer_videos import csv_to_yolo_bbox, OrganizerCsvRow


def test_csv_to_yolo_bbox_normal():
    """3840x2160 frame, bbox 100..300 x 200..400"""
    row = OrganizerCsvRow(ts_ms=0, x_min=100, y_min=200, x_max=300, y_max=400)
    cx, cy, w, h = csv_to_yolo_bbox(row, frame_w=3840, frame_h=2160)
    assert abs(cx - 200/3840) < 1e-4
    assert abs(cy - 300/2160) < 1e-4
    assert abs(w - 200/3840) < 1e-4
    assert abs(h - 200/2160) < 1e-4


def test_csv_to_yolo_bbox_clamps():
    """bbox goes outside frame — clamping"""
    row = OrganizerCsvRow(ts_ms=0, x_min=-10, y_min=-10, x_max=3900, y_max=2200)
    cx, cy, w, h = csv_to_yolo_bbox(row, frame_w=3840, frame_h=2160)
    assert 0 < cx < 1 and 0 < cy < 1
    assert w <= 1.0 and h <= 1.0
