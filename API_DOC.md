# Seal Embedding API - Interface Doc

Base URL: http://localhost:8000

All responses are JSON. Thresholds/confidence filtering is not used.

**Current input format: Only accepts base64-encoded images. File upload not currently supported.**

## Common error responses
- `400 Bad Request`: invalid input, missing fields, or no seals detected
- `404 Not Found`: query_seal_id does not exist
- `500 Internal Server Error`: service not initialized or unexpected errors

Error response body (FastAPI default):
```json
{
  "detail": "Error message"
}
```

## 1) Health
GET `/health`

Query params:
- `verify` (bool, default: false): run verification using detection+embedding
- `max_images` (int, default: 3): max samples per directory
- `random_sample` (bool, default: true): random sample when `max_images` is set
- `force` (bool, default: false): bypass verify cache
- `verify_ttl_sec` (int, default: 300): cache ttl for verify

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "embedding_dim": 768,
  "milvus_loaded": true,
  "milvus_collection": "seals",
  "milvus_count": 125,
  "message": "Model is ready",
  "verify": {
    "query_count": 3,
    "candidate_count": 3,
    "query_detected": 5,
    "candidate_detected": 7,
    "top1_avg": 0.91,
    "top1_min": 0.85,
    "top1_max": 0.97,
    "duration_ms": 142,
    "cached": false
  }
}
```

Notes:
- `verify=true` requires data in `data/query` and `data/candidate`.
- Verification runs detection first, then embedding, then similarity top-1 stats.
- `milvus_loaded`: Whether Milvus vector database is properly initialized.
- `milvus_collection`: Name of the Milvus collection (e.g., "seals").
- `milvus_count`: Total number of embeddings stored in the database.
- If health fails, the service attempts one in-process reinit and retries once.
- If `verify=true` and no images exist, returns `400` with `detail="Verification data not found"`.
- If `verify=true` and no seals are detected, returns `400` with `detail="No seals detected for verification"`.

## 2) Ingest (base64)
POST `/seals/ingest_base64`

Request body:
```json
{
  "items": [
    {
      "image_b64": "<base64-1>",
      "seal_id": "uid_001"
    },
    {
      "image_b64": "<base64-2>",
      "seal_id": "uid_002"
    }
  ]
}
```

Response:
```json
{
  "seals_detected": 2,
  "embeddings_stored": [
    {
      "seal_id": "uid_001",
      "embedding_dim": 768,
      "crop_image_b64": "<base64-encoded-crop-image-1>"
    },
    {
      "seal_id": "uid_002",
      "embedding_dim": 768,
      "crop_image_b64": "<base64-encoded-crop-image-2>"
    }
  ],
  "status": "success"
}
```

Notes:
- Detection runs first, then embedding extraction in batches.
- Each item contains `image_b64` and `seal_id` paired together, avoiding order mismatch.
- **Each image returns only ONE seal**: if multiple seals detected, selects the one with highest confidence (or largest area if no confidence).
- If an image has no seal detected, it will be skipped (not stored, no error).
- `seals_detected` may be less than input image count if some images have no seals.
- `embeddings_stored` only includes successfully detected and stored seals.
- Stores only `id` (seal_id) and `vector` (embedding) in Milvus.
- Returns `crop_image_b64` for each detected seal (cropped from original image).
- If `items` is empty or missing, returns `400` with `detail="seal_ids is required"` or `detail="No images provided"`.
- If any `image_b64` fails to decode, returns `400` with `detail="Failed to decode image: ..."` 
- If no seals are detected for all images, returns `200` with `seals_detected=0` and empty `embeddings_stored`.
- If Milvus insert fails, returns `500` with `detail="Failed to save embedding: ..."`

## 3) Search
POST `/seals/search`

Request body (image):
```json
{
  "image_b64": "<base64-encoded-image>",
  "top_k": 3
}
```

Response:
```json
{
  "query_seal_id": "query_image",
  "top_1": {
    "seal_id": "uid_005",
    "similarity_score": 0.95,
    "rank": 1,
    "image_base64": "<query-image-base64>"
  },
  "top_3": [
    {
      "seal_id": "uid_005",
      "similarity_score": 0.95,
      "rank": 1,
      "image_base64": "<query-image-base64>"
    }
  ],
  "total_in_database": 25,
  "returned_count": 3,
  "image_base64": "<query-image-base64>",
  "status": "success"
}
```

Notes:
- If `image_b64` is provided, search runs detection and uses the largest detected seal.
- `image_base64` in response is the query image (not the matched crops).
- If `image_b64` is missing, request validation fails (`422`).
- If `top_k < 1`, request validation fails (`422`).
- If `image_b64` is provided but no seals are detected, returns `400` with `detail="No seals detected in query image"`.

## 4) Delete (batch)
POST `/seals/delete`

Request body:
```json
{
  "seal_ids": ["uid_001", "uid_002", "uid_003"]
}
```

Response:
```json
{
  "deleted_count": 2,
  "failed_count": 1,
  "deleted_ids": ["uid_001", "uid_002"],
  "failed_ids": ["uid_003"],
  "status": "partial_success"
}
```

Notes:
- Accepts a list of seal IDs for batch deletion.
- Returns separate lists of succeeded and failed deletions.
- `status` is "success" if all deleted, "partial_success" if some failed.
- If the database is empty, returns `top_1 = null` and `top_3 = []`.
- If `seal_ids` is empty, returns `400` with `detail="seal_ids cannot be empty"`.
