# AI Shelf Product Detection

Detect products on beverage-shelf photos and return a structured listing (SKU, packaging type, bounding box, confidence, facings).

## Goal

**Input:** a high-resolution photo of a beverage shelf (variable lighting, angle, distance, partial occlusions).

**Output:** JSON listing of detected products, including at least:

- product id / label (SKU from a reference catalog)
- packaging type: `bottle`, `can`, or `carton` (from the matched gallery folder)
- bounding box per instance
- confidence score
- recognition status: identified or unknown
- facing count per product (`sku_id` + packaging)

## Architecture (two stages)

```
Shelf photo
    |
    v
[Stage 1] Detector  -->  boxes only (class: object)
    |
    | crop each box
    v
[Stage 2] Embedding + Vector DB  -->  SKU match or "unknown"
    |                                  packaging from gallery folder
    v
JSON result + facings
```

| Stage | Job | Does it know the brand? |
|-------|-----|-------------------------|
| 1 – Localization | Find items | No |
| 2 – Recognition | Match crop to SKU gallery | Yes (via embeddings). Packaging comes from `data/single_products/{packaging}/{sku_id}/`. |

Product identity is **`(sku_id, packaging_type)`**. The same folder name under `bottle/` and `carton/` is two catalog products.

New products are enrolled by inserting embeddings into the gallery — **no full-model retrain**.

## Current status

| Step | Name | Status |
|------|------|--------|
| **1** | Data pipeline (labels) | SKU-110K 1-class boxes for Stage 1; gallery photos under `data/single_products/` |
| **2** | Stage 1 detector | **Done** — 1-class YOLO (`object`); web app uses `runs/finetune_shelf/weights/best.pt` |
| **3** | Stage 2 SKU recognition | **Implemented** — OpenCLIP + FAISS gallery, full-index search |
| **4** | Unknowns & dynamic enrollment | Not started |

## Project layout

```
ai-marchup-v2/
  auto_label.py          # OWLv2 sliced pseudo-labeling -> YOLO labels (legacy 3-class path)
  plot_labels.py         # Draw YOLO labels for visual QA
  train_yolo.py          # Train 1-class detector on data/SKU-110K
  embed_gallery.py       # Build FAISS index from single_products/{packaging}/{sku}/
  recognize.py           # CLI Stage 2 matching
  pipeline.py            # Web inference: detect -> crop -> embed -> match
  app.py                 # FastAPI server
  requirements.txt
  data/
    shelf_images/        # Raw unlabeled shelf photos
    single_products/     # SKU reference images: {bottle|can|carton}/{sku_id}/
    SKU-110K/            # Dense-shelf detector training set
  gallery/               # faiss.index + metadata.json
  yolo_data/             # Ultralytics lists + data.yaml (nc=1, object)
  runs/                  # Training & predict outputs
  docs/                  # Detailed step guides
```

## Local setup

The app needs these paths, and they **are tracked in git**: `data/single_products/`, `gallery/`, `runs/sku110k_s/weights/best.pt`, `runs/dinov3_ft/`. Training-only folders (`data/SKU-110K`, `data/shelf_images`, other `runs/`) stay gitignored.

`runs/dinov3_ft/model.safetensors` is ~327 MB — use Git LFS before you push to GitHub (see `DEPLOY.md`).

1. **Clone and create a virtualenv** (Python 3.12):

```powershell
git clone <repo-url>
cd ai-marchup-v2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. **Install PyTorch, then the rest of the deps.** The VPS demo is **CPU-only**. Locally you may install a CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/); otherwise:

```powershell
pip install torch
pip install -r requirements.txt
```

If you have SKU photos but no gallery index yet:

```powershell
python embed_gallery.py --source data/single_products --out gallery --device cpu
```

3. **Run the web app** and open http://127.0.0.1:8000

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The pipeline falls back to CPU if CUDA is not available. First startup downloads OpenCLIP weights and can take a minute.

## Web app

A minimal FastAPI + HTML/JS app wraps the full pipeline (detect -> crop -> embed -> SKU match). All models are loaded once at startup.

```powershell
.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000 — upload a shelf photo to get the annotated image, per-SKU facing counts, and all product crops.

On each crop card you can change the SKU (or create a new one). Use the search box to find a SKU, then press **Save corrections** once at the bottom. Only crops you reassigned are copied into `data/single_products/{bottle|can|carton}/{sku_id}/`. Catalog facing counts are on the **Catalog matches** tab. Re-run `embed_gallery.py` afterward so the next analyze uses the new images. The live FAISS index is not updated on Save.

- `app.py` — FastAPI server (`POST /api/analyze`, `GET /api/skus`, `POST /api/save-crop`, static frontend, result serving)
- `enroll.py` — sanitize SKU names and copy a crop into the gallery folder
- `pipeline.py` — model loading + single `analyze()` inference call
- `web/` — frontend (HTML/CSS/JS, no build step)
- Results are written under `tmp/web/<request_id>/` (last 10 kept)

## Quick start

```powershell
# Train 1-class detector on SKU-110K (optional; weights may already exist)
python train_yolo.py --epochs 20 --imgsz 940 --batch 3 --device 0

# Predict boxes only
yolo predict model=runs/sku110k/weights/best.pt source=data/shelf_images/17.jpg imgsz=940 conf=0.45 iou=0.5 max_det=700 save=True save_crop=True

# Debug Stage 2 on one crop (full gallery; packaging comes from the matched folder)
python test_similarity.py runs/detect/predict-10/crops/object/17-42.jpg
```

## Roadmap

1. [Step 1 – Data pipeline](docs/step1_data_pipeline.md) — synthetic data and/or pseudo-labels  
2. [Step 2 – Stage 1 detector](docs/step2_stage1_detector.md) — localize products (`object`)  
3. [Step 3 – Stage 2 SKU engine](docs/step3_sku_recognition.md) — embeddings + vector gallery  
4. [Step 4 – Unknowns & enrollment](docs/step4_unknown_enrollment.md) — threshold, cluster, insert  

## Notes

- Stage 1 uses a 1-class Ultralytics YOLO trained on SKU-110K. Packaging is **not** a detector class. For production, prefer a permissively licensed model (e.g. **RT-DETR**, Apache-2.0). Ultralytics YOLO is AGPL unless you have a commercial license.
- Dense shelves need `max_det=700` (Ultralytics default 300 drops boxes).
