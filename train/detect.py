import os
import sys
import contextlib
import logging
from pathlib import Path
from typing import List, Dict, Any

os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ.setdefault("GLOG_minloglevel", "2")  # suppress glog INFO/NOTICE
os.environ.setdefault("FLAGS_minloglevel", "2")

from PIL import Image  # noqa: E402


def _extract_seal_boxes(res_obj: Any) -> List[Dict[str, Any]]:
    """Pull out seal boxes from a LayoutDetection result object."""
    boxes = None
    if hasattr(res_obj, "boxes"):
        boxes = getattr(res_obj, "boxes")
    elif isinstance(res_obj, dict) and "boxes" in res_obj:
        boxes = res_obj["boxes"]

    seals: List[Dict[str, Any]] = []
    if boxes:
        for item in boxes:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            if label == "seal" or item.get("cls_id") == 16:
                seals.append(item)
    return seals


@contextlib.contextmanager
def _suppress_host_check_logs():
    """Silence model host connectivity noise by redirecting stdout/stderr (fd-level)."""
    logging.disable(logging.CRITICAL)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(devnull_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)
        logging.disable(logging.NOTSET)


def crop_seals(image_path: str, out_dir: str = "output", model_name: str = "PP-DocLayout-L") -> int:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with _suppress_host_check_logs():
        from paddleocr import LayoutDetection  # local import to keep logs silenced

        model = LayoutDetection(model_name=model_name)
        results = model.predict(image_path, batch_size=1)

    img = Image.open(image_path).convert("RGB")
    stem = Path(image_path).stem

    crop_count = 0
    for res in results:
        for box in _extract_seal_boxes(res):
            coord = box.get("coordinate") or box.get("bbox") or box.get("box")
            if not coord or len(coord) != 4:
                continue
            x0, y0, x1, y1 = map(int, map(round, coord))
            crop = img.crop((x0, y0, x1, y1))
            crop_path = out_path / f"{stem}_seal_{crop_count}.png"
            crop.save(crop_path)
            crop_count += 1

    return crop_count


if __name__ == "__main__":
    with _suppress_host_check_logs():
        cnt = crop_seals("data/seal_imgs/huishan_ano.png", out_dir="output")
    print(f"Saved {cnt} seal crops to output")
