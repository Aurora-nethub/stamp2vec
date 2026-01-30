# API Tests

## /health
- healthy, verify=false
  - Expect status=200, response.status in {"healthy","unhealthy"}
  - Expect embedding_dim, milvus_* fields present
- verify=true with missing data
  - Setup: remove or empty `data/query` or `data/candidate`
  - Expect status=400 with detail="Verification data not found"
- verify=true with no seals detected
  - Use images with no seals
  - Expect status=400 with detail="No seals detected for verification"

## /seals/ingest_base64
- valid single image
  - Expect status=200, seals_detected>=0
  - If seals_detected>0, embeddings_stored length matches
- invalid base64
  - Expect status=400, detail includes "Failed to decode image"
- mismatched seal_ids
  - items missing or empty => 400

## /seals/search
- by image_b64 with no seal
  - Expect status=400, detail="No seals detected in query image"
- by query_seal_id not found
  - Expect status=404, detail="Query seal_id not found"
- top_k < 1
  - Expect 422 from pydantic validation

## /seals/delete
- empty seal_ids
  - Expect status=400, detail="seal_ids cannot be empty"
- delete mix of valid/invalid ids
  - Expect status=200, status in {"success","partial_success"}
