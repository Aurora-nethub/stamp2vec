# Seal Embedding API - Interface Doc

Base URL: http://localhost:8000

All responses are JSON. Thresholds/confidence filtering is not used.

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
- If health fails, the service attempts one in-process reinit and retries once.

## 2) Ingest (base64)
POST `/seals/ingest_base64`

Request body:
```json
{
  "doc_id": "doc_001",
  "images_b64": ["<base64-1>", "<base64-2>"],
  "save_crops": true
}
```

Response:
```json
{
  "doc_id": "doc_001",
  "seals_detected": 2,
  "embeddings_stored": [
    {
      "seal_id": "doc_001_seal_0",
      "doc_id": "doc_001",
      "embedding_dim": 768,
      "image_crop_saved": true
    }
  ],
  "status": "success"
}
```

Notes:
- Detection runs first, then embedding extraction in batches.
- `save_crops=true` saves crops under `database/embeddings/crops/`.

## 3) Ingest (upload)
POST `/seals/ingest_upload`

Form-Data:
- `images`: one or more files
- `doc_id`: string
- `save_crops`: bool (optional, default true)

Response: same shape as `/seals/ingest_base64`.

## 4) Search
POST `/seals/search`

Request body (image):
```json
{
  "image_b64": "<base64-encoded-image>",
  "top_k": 3
}
```

Request body (seal id):
```json
{
  "query_seal_id": "doc_001_seal_0",
  "top_k": 3
}
```

Response:
```json
{
  "query_seal_id": "query_image",
  "top_1": {
    "seal_id": "doc_002_seal_1",
    "doc_id": "doc_002",
    "similarity_score": 0.95,
    "rank": 1,
    "crop_path": "database/embeddings/crops/doc_002_seal_1.png"
  },
  "top_3": [
    {
      "seal_id": "doc_002_seal_1",
      "doc_id": "doc_002",
      "similarity_score": 0.95,
      "rank": 1,
      "crop_path": "database/embeddings/crops/doc_002_seal_1.png"
    }
  ],
  "total_in_database": 25,
  "returned_count": 3,
  "status": "success"
}
```

Notes:
- If `image_b64` is provided, search runs detection and uses the largest detected seal.
- If the database is empty, returns `top_1 = null` and `top_3 = []`.
