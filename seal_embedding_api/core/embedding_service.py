"""
嵌入提取服务模块
- 封装 SealEmbeddingNet 模型（嵌入提取）
- 提供异步嵌入提取接口
- 支持批量处理（batch_size 可配）
"""

import asyncio
import torch
from PIL import Image
from typing import List
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF

from seal_embedding_api.logger_config import get_logger

logger = get_logger(__name__)


class SquarePad:
    def __init__(self, fill: int = 255, mode: str = "constant"):
        self.fill = int(fill)
        self.mode = mode

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return TF.pad(image, padding, fill=self.fill, padding_mode=self.mode)


class ExtraPad:
    def __init__(self, pad: int, fill: int = 255, mode: str = "constant"):
        self.pad = int(pad)
        self.fill = int(fill)
        self.mode = mode

    def __call__(self, image):
        if self.pad <= 0:
            return image
        padding = (self.pad, self.pad, self.pad, self.pad)
        return TF.pad(image, padding, fill=self.fill, padding_mode=self.mode)


class EmbeddingService:
    """嵌入提取服务类 - 负责提取 768 维向量"""

    def __init__(self, model, config_dict: dict, device: str = "cpu", batch_size: int = 32):
        """
        初始化嵌入服务

        参数:
            model: SealEmbeddingNet 模型实例
            config_dict: 模型配置字典（来自 manifest.json）
            device: 设备名称 ("cpu" 或 "cuda")
            batch_size: 批处理大小（≤32，来自配置）
        """
        self.model = model
        self.config_dict = config_dict
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.transform = self._build_transform()
        
        logger.info(f"EmbeddingService 初始化: device={device}, batch_size={batch_size}")

    def _build_transform(self):
        """从配置构建图像变换"""
        pp = self.config_dict.get("preprocess", {})
        img_size = int(pp.get("img_size", 518))
        mean = pp.get("normalize", {}).get("mean", [0.485, 0.456, 0.406])
        std = pp.get("normalize", {}).get("std", [0.229, 0.224, 0.225])

        sp = pp.get("square_pad", {})
        fill = int(sp.get("fill", 255))
        mode = sp.get("mode", "constant")

        interp_name = pp.get("resize", {}).get("interpolation", "bilinear").lower()
        if interp_name == "bilinear":
            interp = transforms.InterpolationMode.BILINEAR
        elif interp_name == "bicubic":
            interp = transforms.InterpolationMode.BICUBIC
        elif interp_name == "nearest":
            interp = transforms.InterpolationMode.NEAREST
        else:
            interp = transforms.InterpolationMode.BILINEAR

        logger.debug(f"构建变换: img_size={img_size}, interp={interp_name}")

        return transforms.Compose([
            SquarePad(fill=fill, mode=mode),
            ExtraPad(0, fill=fill, mode=mode),
            transforms.Resize((img_size, img_size), interpolation=interp),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    async def extract_embedding(self, image: Image.Image) -> np.ndarray:
        """
        异步提取单个图像的嵌入

        参数:
            image: PIL.Image 对象

        返回:
            numpy.ndarray (768,) - L2 归一化的嵌入向量

        异常:
            TypeError: 输入不是 PIL.Image
            RuntimeError: 提取过程出错
        """
        if not isinstance(image, Image.Image):
            logger.error("输入必须是 PIL.Image")
            raise TypeError("Input must be PIL Image")

        try:
            embeddings = await self.extract_embeddings_batch([image])
            if len(embeddings) == 0:
                raise RuntimeError("Failed to extract embedding")
            logger.debug(f"单个图像嵌入提取成功: shape={embeddings[0].shape}")
            return embeddings[0]
        except Exception as e:
            logger.error(f"嵌入提取失败: {e}", exc_info=True)
            raise

    async def extract_embeddings_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        异步提取多个图像的嵌入（不做内部分批，由 pipeline 控制分片）

        参数:
            images: PIL.Image 列表（由 pipeline 分片后传入，长度 <= batch_size）

        返回:
            numpy.ndarray (N, 768) - L2 归一化的嵌入向量

        异常:
            RuntimeError: 提取过程出错
        """
        if not images:
            logger.warning("输入图像列表为空")
            return np.array([]).reshape(0, 768)

        try:
            batch_size = len(images)
            logger.info(f"嵌入提取: 处理 {batch_size} 个图像")

            # 在线程池中运行同步提取（不再内部分批）
            result = await asyncio.to_thread(
                self._extract_embeddings_batch_sync, images
            )

            logger.info(f"嵌入提取完成: shape={result.shape}")
            return result
        except Exception as e:
            logger.error(f"嵌入提取失败: {e}", exc_info=True)
            raise

    @torch.no_grad()
    def _extract_embeddings_batch_sync(
        self, images: List[Image.Image]
    ) -> np.ndarray:
        """
        同步提取多个图像的嵌入（在线程池中运行）

        参数:
            images: PIL.Image 列表

        返回:
            numpy.ndarray (N, 768) - L2 归一化的嵌入向量
        """
        xs = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            xs.append(self.transform(img))

        # 堆叠成 batch
        x = torch.stack(xs, dim=0).to(self.device)  # [N, 3, H, W]

        # 前向传播
        emb = self.model.extract_feat(x)  # [N, 768]

        return emb.cpu().numpy()

    def get_service_info(self) -> dict:
        """获取服务信息"""
        return {
            "batch_size": self.batch_size,
            "device": str(self.device),
            "embedding_dim": 768,
            "model_status": "loaded" if self.model is not None else "not_loaded",
        }
