"""
Similarity search service
"""

import numpy as np
from typing import List, Dict, Tuple


class SimilarityService:
    """Service for computing similarity between embeddings"""
    
    @staticmethod
    def compute_similarity(
        query_embedding: np.ndarray,
        candidate_embeddings: Dict[str, np.ndarray],
        top_k: int = 3
    ) -> List[Dict[str, any]]:
        """
        Compute cosine similarity between query and candidates
        
        Args:
            query_embedding: Query embedding (768,)
            candidate_embeddings: Dict[seal_id, embedding (768,)]
            top_k: Number of top results to return
            
        Returns:
            List of dicts with keys:
              - 'seal_id': Seal ID
              - 'similarity': Cosine similarity score [0, 1]
              Sorted by similarity descending
        """
        if not candidate_embeddings:
            return []
        
        # Normalize query embedding (should already be L2-normalized from model)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        results = []
        for seal_id, candidate_emb in candidate_embeddings.items():
            # Normalize candidate (should already be L2-normalized)
            cand_norm = np.linalg.norm(candidate_emb)
            if cand_norm > 0:
                candidate_emb = candidate_emb / cand_norm
            
            # Cosine similarity
            similarity = np.dot(query_embedding, candidate_emb)
            results.append({
                'seal_id': seal_id,
                'similarity': float(similarity)
            })
        
        # Sort by similarity descending
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top-k (or all if fewer than k)
        return results[:min(top_k, len(results))]
    
    @staticmethod
    def batch_similarity(
        query_embeddings: np.ndarray,  # [N, D]
        candidate_embeddings: np.ndarray,  # [M, D]
        top_k: int = 3
    ) -> List[List[Dict[str, any]]]:
        """
        Compute similarity for multiple queries
        
        Args:
            query_embeddings: Query embeddings (N, 768)
            candidate_embeddings: Candidate embeddings (M, 768)
            top_k: Number of top results per query
            
        Returns:
            List of results for each query
        """
        # Assume embeddings are already L2-normalized
        # Use matrix multiplication for efficiency
        similarities = query_embeddings @ candidate_embeddings.T  # [N, M]
        
        results = []
        for i in range(similarities.shape[0]):
            top_indices = np.argsort(similarities[i])[::-1][:top_k]
            results.append([
                {
                    'index': int(idx),
                    'similarity': float(similarities[i, idx])
                }
                for idx in top_indices
            ])
        
        return results
