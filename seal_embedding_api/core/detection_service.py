import asyncio
from typing import List, Optional, Dict, Any
import numpy as np
from PIL import Image
from paddleocr import LayoutDetection

from seal_embedding_api.logger_config import get_logger
from seal_embedding_api.config_loader import DetectionModelConfig

logger = get_logger(__name__)


def _extract_seal_boxes(res_obj: Any) -> List[Dict[str, Any]]:
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


class DetectionService:
    def __init__(self, config: DetectionModelConfig):
        self.config = config
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            logger.info(f"初始化 LayoutDetection 模型: {self.config.model_name}")

            self.model = LayoutDetection(model_name=self.config.model_name)
            
            logger.info(
                f"LayoutDetection 模型加载成功: {self.config.model_name}, "
                f"batch_size={self.config.batch_size}"
            )
        except Exception as e:
            logger.error(f"LayoutDetection 模型加载失败: {e}", exc_info=True)
            raise

    async def detect_seals(self, images: List[Image.Image]) -> List[Image.Image]:
        if self.model is None:
            logger.error("检测模型未加载")
            raise RuntimeError("Detection model not initialized")

        if not images:
            logger.warning("输入图像列表为空")
            return []

        try:
            seals = await asyncio.to_thread(self._detect_seals_sync, images)
            
            logger.debug(f"检测完成: 从 {len(images)} 张图片检测到 {sum(1 for s in seals if s is not None)} 个印章")
            return seals
        except Exception as e:
            logger.error(f"检测过程出错: {e}", exc_info=True)
            raise

    def _detect_seals_sync(self, images: List[Image.Image]) -> List[Optional[Image.Image]]:
        try:
            batch_size = len(images)
            logger.debug(f"开始印章检测，处理 {batch_size} 个图像...")
            
            rgb_images = []
            np_images = []
            for img in images:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                rgb_images.append(img)
                np_images.append(np.array(img))
            
            results = self.model.predict(np_images, batch_size=batch_size)
            
            seals = []
            detected_count = 0
            
            for img_idx, res in enumerate(results):
                seal_boxes = _extract_seal_boxes(res)
                
                if not seal_boxes:
                    logger.debug(f"图像 {img_idx}: 未检测到印章")
                    seals.append(None)
                    continue
                
                best_box = None
                best_score = -1
                best_area = 0
                
                for box in seal_boxes:
                    coord = box.get("coordinate") or box.get("bbox") or box.get("box")
                    if not coord or len(coord) != 4:
                        continue
                    
                    score = box.get("score") or 0
                    
                    x0, y0, x1, y1 = coord
                    area = (x1 - x0) * (y1 - y0)
                    
                    if score > best_score or (score == best_score and area > best_area):
                        best_box = box
                        best_score = score
                        best_area = area
                
                if best_box:
                    coord = best_box.get("coordinate") or best_box.get("bbox") or best_box.get("box")
                    x0, y0, x1, y1 = map(int, map(round, coord))
                    seal_crop = rgb_images[img_idx].crop((x0, y0, x1, y1))
                    seals.append(seal_crop)
                    detected_count += 1
                    logger.debug(
                        f"图像 {img_idx}: 选择最佳印章 (score={best_score:.3f}, area={best_area:.0f})"
                    )
                else:
                    seals.append(None)
            
            logger.info(
                f"印章检测成功: 从 {batch_size} 个图像检测到 {detected_count} 个印章 "
                f"(每张图片最多1个)"
            )
            return seals
        except Exception as e:
            logger.error(f"同步检测失败: {e}", exc_info=True)
            raise
