import base64
import io
import uuid
from typing import List
import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException, Request

from seal_embedding_api.logger_config import get_logger
from seal_embedding_api.models import (
    IngestRequest,
    IngestResponse,
    EmbeddingMetadata,
    SimilaritySearchRequest,
    SimilaritySearchResponse,
    SimilarityMatch,
    DeleteRequest,
    DeleteResponse,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/seals", tags=["seals"])


async def _ingest_images(
    images,
    seal_ids: List[str],
    request_id: str,
    app_state,
):
    if not images:
        logger.error(f"[{request_id}] 未提供图像")
        raise HTTPException(status_code=400, detail="No images provided")
    
    if not seal_ids:
        logger.error(f"[{request_id}] seal_ids 不能为空")
        raise HTTPException(status_code=400, detail="seal_ids is required")
    
    if len(seal_ids) != len(images):
        logger.error(f"[{request_id}] seal_ids 数量不匹配: {len(seal_ids)} vs {len(images)}")
        raise HTTPException(status_code=400, detail="seal_ids count must match images count")
    
    logger.info(f"[{request_id}] 使用外部传入的 seal_ids: {seal_ids}")
    
    detection_service = app_state.detection_service
    embedding_service = app_state.embedding_service
    milvus_service = app_state.milvus_service
    config = app_state.config

    if not all([detection_service, embedding_service, milvus_service, config]):
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

    detected_count = sum(1 for seal in all_seals if seal is not None)
    logger.info(f"[{request_id}] 检测完成: 从 {len(images)} 张图片检测到 {detected_count} 个印章")

    if detected_count == 0:
        logger.warning(f"[{request_id}] 所有图片均未检测到印章")
        return IngestResponse(
            seals_detected=0,
            embeddings_stored=[],
            status="success"
        )

    valid_seals = [seal for seal in all_seals if seal is not None]
    
    logger.info(f"[{request_id}] 开始提取嵌入，共 {len(valid_seals)} 个印章...")
    embedding_batch_size = config.batch_processing.embedding_batch_size

    all_embeddings = []
    for batch_start in range(0, len(valid_seals), embedding_batch_size):
        batch_end = min(batch_start + embedding_batch_size, len(valid_seals))
        batch_seals = valid_seals[batch_start:batch_end]

        logger.debug(f"[{request_id}] 提取 batch [{batch_start}:{batch_end}]，大小={len(batch_seals)}")

        batch_embeddings = await embedding_service.extract_embeddings_batch(batch_seals)
        all_embeddings.append(batch_embeddings)

    embeddings = np.vstack(all_embeddings)
    logger.info(f"[{request_id}] 嵌入提取完成: shape={embeddings.shape}")

    logger.info(f"[{request_id}] 开始存储嵌入到 Milvus...")
    embeddings_stored = []
    embedding_idx = 0
    
    for img_idx, (seal_id, seal) in enumerate(zip(seal_ids, all_seals)):
        if seal is None:
            logger.warning(f"[{request_id}] 图片 {img_idx} (seal_id={seal_id}) 未检测到印章，跳过")
            continue
        
        try:
            emb = embeddings[embedding_idx]
            embedding_idx += 1
            
            success = milvus_service.insert_embedding(
                seal_id=seal_id,
                embedding=emb,
            )

            if success:
                crop_buffer = io.BytesIO()
                seal.save(crop_buffer, format="PNG")
                crop_b64 = base64.b64encode(crop_buffer.getvalue()).decode('utf-8')
                
                embeddings_stored.append(
                    EmbeddingMetadata(
                        seal_id=seal_id,
                        embedding_dim=embeddings.shape[1],
                        crop_image_b64=crop_b64,
                    )
                )
                logger.debug(f"[{request_id}] 保存嵌入到 Milvus: {seal_id}")
            else:
                raise Exception(f"Failed to insert into Milvus")
        except Exception as e:
            logger.error(f"[{request_id}] 保存嵌入 {seal_id} 失败: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save embedding: {e}")

    logger.info(
        f"[{request_id}] ingest 完成: 输入图片={len(images)}, 检测成功={detected_count}, 存储={len(embeddings_stored)}"
    )

    return IngestResponse(
        seals_detected=detected_count,
        embeddings_stored=embeddings_stored,
        status="success"
    )


@router.post("/ingest_base64", response_model=IngestResponse)
async def ingest_base64(request: IngestRequest, fastapi_request: Request):
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] 开始 ingest_base64，数量: {len(request.items)}")
    
    try:
        app_state = fastapi_request.app.state

        logger.debug(f"[{request_id}] 解码 base64 图像...")
        try:
            images = []
            seal_ids = []
            for idx, item in enumerate(request.items):
                image_data = base64.b64decode(item.image_b64)
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                images.append(image)
                seal_ids.append(item.seal_id)
                logger.debug(f"[{request_id}] 图像解码成功: idx={idx}, seal_id={item.seal_id}, size={image.size}")
        except Exception as e:
            logger.error(f"[{request_id}] 图像解码失败: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

        return await _ingest_images(
            images=images,
            seal_ids=seal_ids,
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
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] 开始 search")

    try:
        app_state = fastapi_request.app.state
        detection_service = app_state.detection_service
        embedding_service = app_state.embedding_service
        milvus_service = app_state.milvus_service

        if not all([detection_service, embedding_service, milvus_service]):
            logger.error(f"[{request_id}] 服务未初始化")
            raise HTTPException(status_code=500, detail="Services not initialized")

        query_embedding = None
        query_image_b64 = None

        logger.debug(f"[{request_id}] 解码 base64 查询图像...")
        try:
            image_data = base64.b64decode(request.image_b64)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            query_image_b64 = request.image_b64
        except Exception as e:
            logger.error(f"[{request_id}] 图像解码失败: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

        logger.info(f"[{request_id}] 开始检测查询印章...")
        seals = await detection_service.detect_seals([image])
        valid_seals = [seal for seal in seals if seal is not None]
        if not valid_seals:
            logger.warning(f"[{request_id}] 查询图像未检测到印章")
            raise HTTPException(status_code=400, detail="No seals detected in query image")

        query_seal = max(valid_seals, key=lambda s: s.size[0] * s.size[1])
        query_embedding = await embedding_service.extract_embedding(query_seal)
        query_seal_id = "query_image"

        logger.info(f"[{request_id}] 搜索相似印章...")
        top_k = request.top_k
        
        search_results = milvus_service.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        total_in_database = milvus_service.count()

        if not search_results:
            return SimilaritySearchResponse(
                query_seal_id=query_seal_id,
                top_1=None,
                top_3=[],
                total_in_database=total_in_database,
                returned_count=0,
                image_base64=query_image_b64,
                status="success",
            )

        matches = []
        for idx, item in enumerate(search_results):
            matches.append(
                SimilarityMatch(
                    seal_id=item["id"],
                    similarity_score=item["similarity"],
                    rank=idx + 1,
                    image_base64=query_image_b64,
                )
            )

        top_1 = matches[0] if matches else None
        top_3 = matches[:3]

        logger.info(f"[{request_id}] 搜索完成: 返回 {len(matches)} 个结果")

        return SimilaritySearchResponse(
            query_seal_id=query_seal_id,
            top_1=top_1,
            top_3=top_3,
            total_in_database=total_in_database,
            returned_count=len(matches),
            image_base64=query_image_b64,
            status="success",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] search 发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@router.post("/delete", response_model=DeleteResponse)
async def delete_embeddings(request: DeleteRequest, fastapi_request: Request):
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] 开始 delete，待删除数量: {len(request.seal_ids)}")
    
    try:
        app_state = fastapi_request.app.state
        milvus_service = app_state.milvus_service
        
        if not milvus_service:
            logger.error(f"[{request_id}] Milvus 服务未初始化")
            raise HTTPException(status_code=500, detail="Milvus service not initialized")
        
        if not request.seal_ids:
            logger.error(f"[{request_id}] seal_ids 为空")
            raise HTTPException(status_code=400, detail="seal_ids cannot be empty")
        
        succeeded, failed = milvus_service.delete(request.seal_ids)
        
        logger.info(
            f"[{request_id}] 删除完成: 成功={len(succeeded)}, 失败={len(failed)}"
        )
        
        return DeleteResponse(
            deleted_count=len(succeeded),
            failed_count=len(failed),
            deleted_ids=succeeded,
            failed_ids=failed,
            status="success" if len(failed) == 0 else "partial_success",
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] delete 发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")

