"""
检测服务模块
- 封装 LayoutDetection 模型（印章检测）
- 提供异步检测接口
- 模型在启动时一次性加载，永久驻留内存
"""

import asyncio
from typing import List
from PIL import Image
from train.detect import _suppress_host_check_logs, _extract_seal_boxes
from paddleocr import LayoutDetection

from seal_embedding_api.logger_config import get_logger
from seal_embedding_api.config_loader import DetectionModelConfig

logger = get_logger(__name__)


class DetectionService:
    """检测服务类 - 负责印章检测"""

    def __init__(self, config: DetectionModelConfig):
        """
        初始化检测服务

        参数:
            config: DetectionModelConfig 配置对象，包含 model_name 和 batch_size
        """
        self.config = config
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """初始化 LayoutDetection 模型（一次性加载，驻留内存）"""
        try:
            logger.info(f"初始化 LayoutDetection 模型: {self.config.model_name}")
            
            # 抑制 PaddleOCR 的日志噪声
            with _suppress_host_check_logs():
                self.model = LayoutDetection(model_name=self.config.model_name)
            
            logger.info(
                f"LayoutDetection 模型加载成功: {self.config.model_name}, "
                f"batch_size={self.config.batch_size}"
            )
        except Exception as e:
            logger.error(f"LayoutDetection 模型加载失败: {e}", exc_info=True)
            raise

    async def detect_seals(self, images: List[Image.Image]) -> List[Image.Image]:
        """
        异步检测印章（接受图像列表）

        参数:
            images: List[PIL.Image] - 图像列表（由 pipeline 分片后传入）

        返回:
            List[PIL.Image] - 所有检测到的印章裁剪图像列表（无阈值，一定要检测出来）

        异常:
            RuntimeError: 模型加载失败或检测出错
        """
        if self.model is None:
            logger.error("检测模型未加载")
            raise RuntimeError("Detection model not initialized")

        if not images:
            logger.warning("输入图像列表为空")
            return []

        try:
            # 在线程池中运行 CPU 密集操作，不阻塞事件循环
            seals = await asyncio.to_thread(self._detect_seals_sync, images)
            
            logger.debug(f"检测完成: 检测到 {len(seals)} 个印章")
            return seals
        except Exception as e:
            logger.error(f"检测过程出错: {e}", exc_info=True)
            raise

    def _detect_seals_sync(self, images: List[Image.Image]) -> List[Image.Image]:
        """
        同步的印章检测逻辑（运行在线程池中）
        
        不使用置信度阈值，一定要检测出来（第一版本要求）

        参数:
            images: PIL.Image 列表（长度就是 batch_size）

        返回:
            List[PIL.Image] - 所有检测到的印章裁剪图像列表
        """
        try:
            batch_size = len(images)
            logger.debug(f"开始印章检测，处理 {batch_size} 个图像...")
            
            # 确保所有图像都是 RGB
            rgb_images = []
            for img in images:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                rgb_images.append(img)
            
            # 使用 LayoutDetection 模型进行检测
            # batch_size 就是列表长度
            with _suppress_host_check_logs():
                results = self.model.predict(rgb_images, batch_size=batch_size)
            
            # 提取 seal box 并裁剪
            seals = []
            for img_idx, res in enumerate(results):
                for box in _extract_seal_boxes(res):
                    # 获取坐标（无阈值要求，只要检测到就裁剪）
                    coord = box.get("coordinate") or box.get("bbox") or box.get("box")
                    if not coord or len(coord) != 4:
                        logger.debug(f"图像 {img_idx}: 跳过无效坐标的 box")
                        continue
                    
                    x0, y0, x1, y1 = map(int, map(round, coord))
                    seal_crop = rgb_images[img_idx].crop((x0, y0, x1, y1))
                    seals.append(seal_crop)
            
            logger.info(f"印章检测成功: 从 {batch_size} 个图像检测到 {len(seals)} 个印章")
            return seals
        except Exception as e:
            logger.error(f"同步检测失败: {e}", exc_info=True)
            raise

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.config.model_name,
            "batch_size": self.config.batch_size,
            "status": "loaded" if self.model is not None else "not_loaded",
        }
