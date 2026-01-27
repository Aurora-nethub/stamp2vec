"""
Ingest and search endpoints for detection, embedding, and similarity search
"""

import base64
import io
import uuid
from typing import List
import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form

from seal_embedding_api.logger_config import get_logger
from seal_embedding_api.models import (
    IngestRequest,
    IngestResponse,
    EmbeddingMetadata,
    SimilaritySearchRequest,
    SimilaritySearchResponse,
    SimilarityMatch,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/seals", tags=["seals"])


async def _ingest_images(
    images,
    doc_id: str,
    save_crops: bool,
    request_id: str,
    app_state,
):
    if not images:
        logger.error(f"[{request_id}] 未提供图像")
        raise HTTPException(status_code=400, detail="No images provided")
    detection_service = app_state.detection_service
    embedding_service = app_state.embedding_service
    storage = app_state.storage
    config = app_state.config

    if not all([detection_service, embedding_service, storage, config]):
        logger.error(f"[{request_id}] 服务未初始化")
        raise HTTPException(status_code=500, detail="Services not initialized")

    logger.info(f"[{request_id}] 开始检测印章...")
    detection_batch_size = config.detection_model.batch_size
    all_seals = []
    for batch_start in range(0, len(images), detection_batch_size):
        batch_end = min(batch_start + detection_batch_size, len(images))
        batch_images = images[batch_start:batch_end]

        logger.debug(
            f"[{request_id}] 检测 batch [{batch_start}:{batch_end}]，大小={len(batch_images)}"
        )
        batch_seals = await detection_service.detect_seals(batch_images)
        all_seals.extend(batch_seals)

    logger.info(f"[{request_id}] 检测完成: 检测到 {len(all_seals)} 个印章")

    if len(all_seals) == 0:
        logger.warning(f"[{request_id}] 未检测到印章")
        return IngestResponse(
            doc_id=doc_id,
            seals_detected=0,
            embeddings_stored=[],
            status="success"
        )

    logger.info(f"[{request_id}] 开始提取嵌入，共 {len(all_seals)} 个印章...")
    embedding_batch_size = config.batch_processing.embedding_batch_size

    all_embeddings = []
    for batch_start in range(0, len(all_seals), embedding_batch_size):
        batch_end = min(batch_start + embedding_batch_size, len(all_seals))
        batch_seals = all_seals[batch_start:batch_end]

        logger.debug(f"[{request_id}] 提取 batch [{batch_start}:{batch_end}]，大小={len(batch_seals)}")

        batch_embeddings = await embedding_service.extract_embeddings_batch(batch_seals)
        all_embeddings.append(batch_embeddings)

    embeddings = np.vstack(all_embeddings)
    logger.info(f"[{request_id}] 嵌入提取完成: shape={embeddings.shape}")

    logger.info(f"[{request_id}] 开始存储嵌入...")
    embeddings_stored = []
    for seal_idx, (seal, emb) in enumerate(zip(all_seals, embeddings)):
        seal_id = f"{doc_id}_seal_{seal_idx}"
        try:
            storage.save_embedding(seal_id, emb, doc_id, crop_path=crop_path)

            crop_path = None
            if save_crops:
                crop_path = f"database/embeddings/crops/{seal_id}.png"
                seal.save(crop_path)

            embeddings_stored.append(
                EmbeddingMetadata(
                    seal_id=seal_id,
                    doc_id=doc_id,
                    embedding_dim=embeddings.shape[1],
                    image_crop_saved=save_crops,
                )
            )
            logger.debug(f"[{request_id}] 保存嵌入: {seal_id}")
        except Exception as e:
            logger.error(f"[{request_id}] 保存嵌入 {seal_id} 失败: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save embedding: {e}")

    logger.info(
        f"[{request_id}] ingest 完成: 检测={len(all_seals)}, 存储={len(embeddings_stored)}"
    )

    return IngestResponse(
        doc_id=doc_id,
        seals_detected=len(all_seals),
        embeddings_stored=embeddings_stored,
        status="success"
    )


@router.post("/ingest_base64", response_model=IngestResponse)
async def ingest_base64(request: IngestRequest, fastapi_request: Request):
    """
    Detect seals in image and extract embeddings
    
    Steps:
    1. Decode base64 images
    2. Detect seals using LayoutDetection (batched)
    3. Extract embeddings for each seal (with batch_size control from ingest config)
    4. Store embeddings and metadata
    5. Return results
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] 开始 ingest_base64: doc_id={request.doc_id}")
    
    try:
        app_state = fastapi_request.app.state

        logger.debug(f"[{request_id}] 解码 base64 图像...")
        try:
            images = []
            for idx, image_b64 in enumerate(request.images_b64):
                image_data = base64.b64decode(image_b64)
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                images.append(image)
                logger.debug(f"[{request_id}] 图像解码成功: idx={idx}, size={image.size}")
        except Exception as e:
            logger.error(f"[{request_id}] 图像解码失败: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

        return await _ingest_images(
            images=images,
            doc_id=request.doc_id,
            save_crops=request.save_crops,
            request_id=request_id,
            app_state=app_state,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] ingest 发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")


@router.post("/ingest_upload", response_model=IngestResponse)
async def ingest_upload(
    fastapi_request: Request,
    doc_id: str = Form(...),
    save_crops: bool = Form(True),
    images: List[UploadFile] = File(...),
):
    """
    Detect seals in uploaded images and extract embeddings
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] 开始 ingest_upload: doc_id={doc_id}")

    try:
        app_state = fastapi_request.app.state

        logger.debug(f"[{request_id}] 读取上传文件...")
        try:
            pil_images = []
            for idx, upload in enumerate(images):
                content = await upload.read()
                image = Image.open(io.BytesIO(content)).convert("RGB")
                pil_images.append(image)
                logger.debug(f"[{request_id}] 文件读取成功: idx={idx}, filename={upload.filename}")
        except Exception as e:
            logger.error(f"[{request_id}] 文件读取失败: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to read uploaded files: {e}")

        return await _ingest_images(
            images=pil_images,
            doc_id=doc_id,
            save_crops=save_crops,
            request_id=request_id,
            app_state=app_state,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] ingest 发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")


@router.post("/search", response_model=SimilaritySearchResponse)
async def search_similar(request: SimilaritySearchRequest, fastapi_request: Request):
    """
    Search for similar seals in database
    
    Input options:
    - image_b64: New seal image (will extract embedding)
    - query_seal_id: Existing seal ID from database
    
    Returns:
    - Top-k similar seals from database
    - If database has < k items, return all
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] 开始 search")

    try:
        app_state = fastapi_request.app.state
        detection_service = app_state.detection_service
        embedding_service = app_state.embedding_service
        storage = app_state.storage
        similarity_service = app_state.similarity_service

        if not all([detection_service, embedding_service, storage, similarity_service]):
            logger.error(f"[{request_id}] 服务未初始化")
            raise HTTPException(status_code=500, detail="Services not initialized")

        query_seal_id = request.query_seal_id
        query_embedding = None

        if request.image_b64:
            logger.debug(f"[{request_id}] 解码 base64 查询图像...")
            try:
                image_data = base64.b64decode(request.image_b64)
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
            except Exception as e:
                logger.error(f"[{request_id}] 图像解码失败: {e}")
                raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

            logger.info(f"[{request_id}] 开始检测查询印章...")
            seals = await detection_service.detect_seals([image])
            if len(seals) == 0:
                logger.warning(f"[{request_id}] 查询图像未检测到印章")
                raise HTTPException(status_code=400, detail="No seals detected in query image")

            query_seal = max(seals, key=lambda s: s.size[0] * s.size[1])
            query_embedding = await embedding_service.extract_embedding(query_seal)
            query_seal_id = query_seal_id or "query_image"
        else:
            query_embedding = storage.get_embedding(query_seal_id)
            if query_embedding is None:
                logger.warning(f"[{request_id}] 未找到查询 seal_id: {query_seal_id}")
                raise HTTPException(status_code=404, detail="Query seal_id not found")

        embeddings, metadata = storage.list_all_embeddings()
        total_in_database = len(embeddings)
        if total_in_database == 0:
            return SimilaritySearchResponse(
                query_seal_id=query_seal_id,
                top_1=None,
                top_3=[],
                total_in_database=0,
                returned_count=0,
                status="success",
            )

        top_k = max(request.top_k, 3)
        results = similarity_service.compute_similarity(
            query_embedding=query_embedding,
            candidate_embeddings=embeddings,
            top_k=top_k,
        )

        matches = []
        for idx, item in enumerate(results):
            seal_id = item["seal_id"]
            doc_id = ""
            crop_path = None
            meta = metadata.get(seal_id)
            if meta:
                doc_id = meta.get("doc_id", "")
                crop_path = meta.get("crop_path") or None
            matches.append(
                SimilarityMatch(
                    seal_id=seal_id,
                    doc_id=doc_id,
                    similarity_score=item["similarity"],
                    rank=idx + 1,
                    crop_path=crop_path,
                )
            )

        top_1 = matches[0] if matches else None
        top_3 = matches[:3]

        return SimilaritySearchResponse(
            query_seal_id=query_seal_id,
            top_1=top_1,
            top_3=top_3,
            total_in_database=total_in_database,
            returned_count=len(matches),
            status="success",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] search 发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

