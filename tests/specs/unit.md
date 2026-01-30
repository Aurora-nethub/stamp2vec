# Unit Tests

## models.py
- SimilaritySearchRequest
  - no image_b64 and no query_seal_id => validation error
  - top_k < 1 => validation error
- IngestRequest
  - empty items => validation error (optional if enforced by API)

## detection_service.py
- _extract_seal_boxes
  - no boxes => empty list
  - boxes with non-seal labels => empty list
  - label="seal" or cls_id=16 => included

## embedding_service.py
- extract_embeddings_batch([])
  - Expect empty array shape (0, 768)
