"""
Detection service for seal detection and cropping
"""

import os
import contextlib
import logging
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import tempfile


class DetectionService:
    """Service for detecting and cropping seals from documents"""
    
    def __init__(self):
        """Initialize detection service"""
        pass
    
    def detect_and_crop_seals(
        self, 
        image: Image.Image,
        output_dir: str = None
    ) -> List[Dict[str, Any]]:
        """
        Detect seals in image and return cropped regions
        
        Args:
            image: PIL Image (RGB)
            output_dir: Directory to save cropped images (temp dir if None)
            
        Returns:
            List of dicts with keys:
              - 'crop': PIL Image of the cropped seal
              - 'bbox': (x0, y0, x1, y1) coordinates
              - 'index': Index of the seal
        """
        from train.detect import crop_seals, _extract_seal_boxes, _suppress_host_check_logs
        
        # Use temp directory if not specified
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save image to temp file
        temp_file = os.path.join(output_dir, "temp_image.png")
        image.save(temp_file)
        
        try:
            # Run detection
            with _suppress_host_check_logs():
                from paddleocr import LayoutDetection
                model = LayoutDetection(model_name="PP-DocLayout-L")
                results = model.predict(temp_file, batch_size=1)
            
            # Extract crops
            crops = []
            crop_index = 0
            
            for res in results:
                for box in _extract_seal_boxes(res):
                    coord = box.get("coordinate") or box.get("bbox") or box.get("box")
                    if not coord or len(coord) != 4:
                        continue
                    
                    x0, y0, x1, y1 = map(int, map(round, coord))
                    crop = image.crop((x0, y0, x1, y1))
                    
                    crops.append({
                        'crop': crop,
                        'bbox': (x0, y0, x1, y1),
                        'index': crop_index
                    })
                    crop_index += 1
            
            return crops
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
