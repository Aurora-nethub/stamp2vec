"""
Embedding extraction service
"""

import os
import torch
from PIL import Image
from typing import List
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF


class SquarePad:
    """Square pad transform"""
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
    """Extra pad transform"""
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
    """Service for extracting embeddings from images"""
    
    def __init__(self, model, config: dict, device: torch.device = None):
        """
        Initialize embedding service
        
        Args:
            model: SealEmbeddingNet model instance
            config: Model config dict (from config.json)
            device: torch device (defaults to CPU)
        """
        self.model = model
        self.config = config
        self.device = device or torch.device("cpu")
        self.transform = self._build_transform()
    
    def _build_transform(self):
        """Build image transform from config"""
        pp = self.config.get("preprocess", {})
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
        
        return transforms.Compose([
            SquarePad(fill=fill, mode=mode),
            ExtraPad(0, fill=fill, mode=mode),
            transforms.Resize((img_size, img_size), interpolation=interp),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    
    @torch.no_grad()
    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Extract embedding from a single PIL image
        
        Args:
            image: PIL Image (RGB)
            
        Returns:
            Embedding vector (768,) as numpy array
        """
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be PIL Image")
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transform
        x = self.transform(image).unsqueeze(0).to(self.device)  # [1, 3, H, W]
        
        # Forward pass
        emb = self.model.extract_feat(x)  # [1, D]
        
        return emb.squeeze(0).cpu().numpy()
    
    @torch.no_grad()
    def extract_embeddings_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Extract embeddings from multiple PIL images
        
        Args:
            images: List of PIL Images (RGB)
            
        Returns:
            Embeddings (N, 768) as numpy array
        """
        if not images:
            return np.array([])
        
        # Process all images
        xs = []
        for img in images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            xs.append(self.transform(img))
        
        x = torch.stack(xs, dim=0).to(self.device)  # [N, 3, H, W]
        emb = self.model.extract_feat(x)  # [N, D]
        
        return emb.cpu().numpy()
