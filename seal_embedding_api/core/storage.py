"""
Storage service for embeddings and metadata
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np


class Storage:
    """
    Local storage service for embeddings and metadata.
    
    Directory structure:
    database/embeddings/
      ├── embeddings.pkl    # Dict[seal_id, embedding]
      ├── metadata.json     # Dict[seal_id, metadata]
      └── crops/           # Cropped seal images
    """
    
    def __init__(self, base_dir: str = "database/embeddings"):
        self.base_dir = Path(base_dir)
        self.embeddings_file = self.base_dir / "embeddings.pkl"
        self.metadata_file = self.base_dir / "metadata.json"
        self.crops_dir = self.base_dir / "crops"
        
        # Create directories if not exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._embeddings = self._load_embeddings()
        self._metadata = self._load_metadata()
    
    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        """Load embeddings from pickle file"""
        if self.embeddings_file.exists():
            with open(self.embeddings_file, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def _load_metadata(self) -> Dict[str, dict]:
        """Load metadata from JSON file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_embeddings(self):
        """Save embeddings to pickle file"""
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self._embeddings, f)
    
    def _save_metadata(self):
        """Save metadata to JSON file"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self._metadata, f, indent=2, ensure_ascii=False)
    
    def save_embedding(
        self, 
        seal_id: str, 
        embedding: np.ndarray, 
        doc_id: str, 
        crop_path: Optional[str] = None
    ):
        """
        Save a single embedding with metadata
        
        Args:
            seal_id: Unique ID for the seal
            embedding: Embedding vector (768,)
            doc_id: Document ID for association
            crop_path: Path to cropped image (relative to base_dir)
        """
        # Store embedding
        self._embeddings[seal_id] = embedding.copy()
        
        # Store metadata
        self._metadata[seal_id] = {
            "doc_id": doc_id,
            "crop_path": crop_path or "",
            "timestamp": self._get_timestamp()
        }
        
        # Persist to disk
        self._save_embeddings()
        self._save_metadata()
    
    def get_embedding(self, seal_id: str) -> Optional[np.ndarray]:
        """Get embedding by seal_id"""
        return self._embeddings.get(seal_id)
    
    def list_all_embeddings(self) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        """Return all embeddings and metadata"""
        return self._embeddings.copy(), self._metadata.copy()
    
    def get_metadata(self, seal_id: str) -> Optional[dict]:
        """Get metadata for a seal"""
        return self._metadata.get(seal_id)
    
    def count_embeddings(self) -> int:
        """Get total number of stored embeddings"""
        return len(self._embeddings)
    
    def seal_exists(self, seal_id: str) -> bool:
        """Check if a seal exists in database"""
        return seal_id in self._embeddings
    
    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()
