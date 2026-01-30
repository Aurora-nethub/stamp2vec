"""Unit tests for detection helpers."""

from seal_embedding_api.core.detection_service import _extract_seal_boxes


def test_extract_seal_boxes_empty():
    """Empty input yields no seal boxes."""
    assert _extract_seal_boxes({}) == []


def test_extract_seal_boxes_filters():
    """Only seal boxes (label=seal or cls_id=16) are kept."""
    boxes = [
        {"label": "text", "coordinate": [0, 0, 1, 1]},
        {"label": "seal", "coordinate": [0, 0, 1, 1]},
        {"cls_id": 16, "coordinate": [0, 0, 1, 1]},
    ]
    result = _extract_seal_boxes({"boxes": boxes})
    assert len(result) == 2
