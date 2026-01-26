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

Dataset Generation
------------------
Use [generate_dataset.py](generate_dataset.py) to synthesize clean/defect ROI pairs per seal/background and emit manifests.

Example
```
python generate_dataset.py \
	--train_stamp_dir stamps/train_png \
	--val_stamp_dir   stamps/val_png \
	--bg_scan_dir     data/backgrounds/scan \
	--bg_photo_dir    data/backgrounds/photo \
	--out_dir         data/generated \
	--train_pairs_per_bg 2 \
	--val_pairs_per_bg   0 \
	--min_cover 0.10 --max_cover 0.40
```

Notes
- For each seal/background pair it creates `N` clean + `N` defect ROIs (same compose) where `N=train_pairs_per_bg` or `val_pairs_per_bg`.
- Outputs under `out_dir/train/<seal_id>/` and `out_dir/val/<seal_id>/` plus `manifest_train.csv` and `manifest_val.csv` with paths and metadata.
- Optional debug masks land in `out_dir/_debug/...` when `--save_debug_masks true`.
