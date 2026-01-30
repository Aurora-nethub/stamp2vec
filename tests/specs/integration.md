# Integration Tests (Milvus Lite)

## Prereqs
- `config/api_config.json` has a writable `milvus.db_path`
- Run `scripts/init_milvus.py` before tests
- Have at least one seal image for ingest

## Flow: ingest -> search -> delete
1) Ingest 1-2 images with known seal_ids
2) Search by image_b64, expect top_1 similarity >= 0
3) Search by query_seal_id, expect exclude_id behavior (no self-match if exclude enabled)
4) Delete by seal_ids, expect deleted_count > 0
5) Search deleted id, expect 404

## Health verify
- With valid data in `data/query` and `data/candidate`
- Expect verify summary present and cached behavior when re-run within ttl
