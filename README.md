Stamp2Vec API
=============

FastAPI service for seal detection, embedding ingestion, similarity search, and deletion.

Quick Start
-----------
1) Create and activate virtual environment in `.venv`.
2) Run the server:

Windows:
```
run.bat
```

macOS/Linux:
```
./run.sh
```

Docs
----
API interface details are in `API_DOC.md`.

Endpoints
---------
- `GET /health`
- `POST /seals/ingest_base64`
- `POST /seals/search`
- `POST /seals/delete`

Notes
-----
Current input format is base64 images only.

Repository Layout
-----------------
- `seal_embedding_api/`: FastAPI app
- `config/`: runtime config (`api_config.json`)
- `models/`: exported model package
- `database/`: local embeddings + metadata + crops
- `data/`: health verification samples (`query/`, `candidate/`)
- `train/`: training and offline scripts

Offline Verification
--------------------
Use `verify.py` to compare query/candidate sets offline (independent of the API).

Example
```
python verify.py \
	--query-dir data/query \
	--cand-dir  data/candidate \
	--pkg models/seal_pkg_v1 \
```
