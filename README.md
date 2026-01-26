Stamp CMP
=========

Lightweight utilities for training and verifying stamp matching models. The core entry points are:

- `train.py`: train a model and save checkpoints.
- `verify.py`: load an exported package and compute similarities between query and candidate images.

Ignored Artifacts
-----------------
The `.gitignore` excludes generated assets to keep the repo clean:

- `checkpoints*/` and `local_models/`: training outputs and downloaded models.
- `stamps/`: raw stamp images.
- `__pycache__/`, `*.py[oc]`, build/dist wheels, and `.venv`.

Verification (verify.py)
------------------------
`verify.py` loads a packaged model (defaults to `exported/seal_pkg_v1`) and computes cosine similarities between query and candidate sets.

Quick start
```
python verify.py \
	--query-dir ~/dinov3/query \
	--cand-dir  ~/dinov3/candidate \
	--pkg exported/seal_pkg_v1 \
	--batch-size 32
```

Arguments
- `--pkg`: directory containing `config.json` and `model.pt` produced by `export.py`.
- `--query-dir` / `--cand-dir`: folders of images (`.png/.jpg/.jpeg/.webp/.bmp/.tif/.tiff`). Files are sorted lexicographically (see `list_images` in `verify.py`).
- `--batch-size`: inference batch size.
- `--out-csv` (optional): path to save the similarity matrix as CSV.

Save similarity matrix
```
python verify.py \
	--query-dir ~/dinov3/query \
	--cand-dir  ~/dinov3/candidate \
	--pkg exported/seal_pkg_v1 \
	--out-csv outputs/sim_matrix.csv
```

Outputs
- Logs the shapes of query/candidate embeddings and the cosine similarity matrix.
- If `--out-csv` is provided, writes the full matrix to the given path.

Training
--------
Use `train.py` to train a model and write checkpoints under `checkpoints*/`. Export a lightweight package with `export.py` before running `verify.py`.
