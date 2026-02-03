import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from pymilvus import MilvusClient
from seal_embedding_api.logger_config import get_logger

logger = get_logger(__name__)


class MilvusService:
    def __init__(self, db_path: str, collection_name: str = "seals"):
        db_file = Path(db_path)
        if not db_file.parent.exists():
            raise FileNotFoundError(f"Milvus db directory not found: {db_file.parent}")
        if not db_file.exists():
            raise FileNotFoundError(f"Milvus db file not found: {db_file}")

        self.client = MilvusClient(uri=str(db_file))
        self.collection_name = collection_name
        logger.info(f"Milvus service initialized: {db_file}, collection: {collection_name}")
    
    def insert_embedding(
        self,
        seal_id: str,
        embedding: np.ndarray,
    ) -> bool:
        try:
            data = [{
                "id": seal_id,
                "vector": embedding.tolist(),
            }]
            
            self.client.insert(
                collection_name=self.collection_name,
                data=data,
            )
            
            logger.debug(f"Inserted embedding: {seal_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert embedding {seal_id}: {e}")
            return False
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        exclude_id: Optional[str] = None,
    ) -> List[Dict]:
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding.tolist()],
                limit=top_k * 2,
                output_fields=["id"],
            )
            
            if not results or len(results) == 0:
                logger.info("No search results found")
                return []
            
            matches = []
            for result in results[0]:
                match_id = result.get("id")
                
                if exclude_id and match_id == exclude_id:
                    continue
                
                distance = result.get("distance", 0)
                similarity = 1 - distance
                
                match_data = {
                    "id": match_id,
                    "similarity": similarity,
                }
                matches.append(match_data)
            
            logger.info(f"Search returned {len(matches)} results")
            return matches[:top_k]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_embedding(self, seal_id: str) -> Optional[np.ndarray]:
        try:
            results = self.client.get(
                collection_name=self.collection_name,
                ids=[seal_id],
                output_fields=["vector"],
            )
            
            if results and len(results) > 0:
                vector = results[0].get("vector")
                if vector:
                    logger.debug(f"Retrieved embedding: {seal_id}")
                    return np.array(vector, dtype=np.float32)
            
            logger.warning(f"Embedding not found: {seal_id}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get embedding {seal_id}: {e}")
            return None
    
    def delete(self, seal_ids: List[str]) -> Tuple[List[str], List[str]]:
        succeeded = []
        failed = []
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                ids=seal_ids,
            )
            succeeded = seal_ids.copy()
            logger.info(f"Deleted {len(succeeded)} embeddings: {succeeded}")
        except Exception as e:
            logger.warning(f"Batch delete failed, trying individually: {e}")
            for seal_id in seal_ids:
                try:
                    self.client.delete(
                        collection_name=self.collection_name,
                        ids=[seal_id],
                    )
                    succeeded.append(seal_id)
                    logger.debug(f"Deleted embedding: {seal_id}")
                except Exception as delete_error:
                    failed.append(seal_id)
                    logger.error(f"Failed to delete embedding {seal_id}: {delete_error}")
        
        return succeeded, failed
    
    def count(self) -> int:
        try:
            stats = self.client.get_collection_stats(collection_name=self.collection_name)
            count = int(stats.get("row_count", 0))
            logger.info(f"Collection count: {count}")
            return count
        except Exception as e:
            logger.error(f"Failed to get collection count: {e}")
            return 0
