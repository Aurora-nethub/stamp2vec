#!/usr/bin/env python
"""
Milvus 数据库初始化脚本
- 检查 Milvus 数据库是否存在
- 如果不存在，创建数据库和集合
- 配置从 config/api_config.json 读取

使用 Milvus Lite（SQLite 后端）
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from seal_embedding_api.logger_config import get_logger
from pymilvus import MilvusClient, DataType
from pymilvus.exceptions import MilvusException

logger = get_logger(__name__)


def load_milvus_config(config_path: str = "config/api_config.json") -> dict:
    """Load Milvus configuration from JSON"""
    config_file = ROOT_DIR / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    if 'milvus' not in config:
        raise KeyError("Milvus configuration not found in config file")
    
    return config['milvus']


def init_milvus_database():
    """Initialize Milvus database and collection"""
    try:
        # Load configuration
        milvus_config = load_milvus_config()
        db_path = milvus_config.get('db_path', 'database/milvus')
        collection_name = milvus_config.get('collection_name', 'seals')
        dimension = milvus_config.get('dimension', 768)
        
        # Resolve absolute path
        if not os.path.isabs(db_path):
            db_path = ROOT_DIR / db_path
        else:
            db_path = Path(db_path)

        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_file = db_path
        logger.info(f"Milvus database file: {db_file}")
        
        # Initialize Milvus Lite client (SQLite backend)
        client = MilvusClient(uri=str(db_file))
        logger.info("Connected to Milvus Lite")
        
        # Check if collection already exists
        collections = client.list_collections()
        if collection_name in collections:
            logger.info(f"Collection '{collection_name}' already exists")
            return True
        
        # Create explicit schema
        logger.info(f"Creating schema for collection '{collection_name}'...")
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        
        # Add id field (VARCHAR primary key)
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=64,  # UUID standard is 36 chars, 64 is safe
        )
        
        # Add vector field (FLOAT_VECTOR)
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dimension,
        )
        
        logger.info(f"Schema created: id (VARCHAR, primary), vector (FLOAT_VECTOR, dim={dimension})")
        
        # Prepare index parameters
        logger.info(f"Preparing index parameters for collection '{collection_name}'...")
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        
        logger.info("Index parameters prepared: HNSW, COSINE metric")
        
        # Create collection with schema and index
        logger.info(f"Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        
        logger.info(f"Collection '{collection_name}' created successfully with schema and index")
        
        return True
        
    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        return False
    except MilvusException as e:
        logger.error(f"Milvus error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during Milvus initialization: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    logger.info("Starting Milvus Lite database initialization...")
    success = init_milvus_database()
    
    if success:
        logger.info("Milvus Lite database initialization completed successfully")
        sys.exit(0)
    else:
        logger.error("Milvus Lite database initialization failed")
        sys.exit(1)
