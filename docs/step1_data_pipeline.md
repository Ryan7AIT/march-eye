# Step 1 — Data Pipeline (Solve "No Labels")

**Status: IN PROGRESS (Approach B)**

You do not start with hand-annotated shelf photos. This step creates training labels for Stage 1 (bottle / can / carton boxes).

## Why this step matters

Object detectors need boxes. Missing labels teach the model that products are "background", so **recall of your labels** matters a lot before serious training.

## Two approaches

### Approach A — Synthetic data (recommended when you have clean SKU photos)

1. Take single-product images from `data/single_products/`.
2. Remove backgrounds (e.g. Rembg).
3. Paste cutouts onto empty shelves / textures with Albumentations (or similar).
4. Boxes and packaging labels are known exactly because you placed the products.

**Pros:** perfect boxes, unlimited volume, controlled variety.  
**Cons:** domain gap vs real fridge photos — still need some real images later.

### Approach B — Pseudo-labeling (what we use now)

1. Run a zero-shot detector (OWLv2) on unlabeled shelf photos.
2. Prompt for packaging types: bottle, can, juice carton.
3. Export YOLO-format labels.
4. Manually review/correct a subset for a **gold validation set**.

**Pros:** uses real shelves immediately.  
**Cons:** noise (misses, duplicates, ads) — needs cleanup.

## What is already implemented

| Piece | Location | Notes |
|-------|----------|-------|
| OWLv2 sliced labeling | `auto_label.py` | Tiles + NMS + EXIF-safe image save |
| Visual QA | `plot_labels.py` | Overlay boxes on dataset images |
| Raw photos | `data/shelf_images/` | Unlabeled inputs |
| Labeled set | `dataset/images/` + `dataset/labels/` | Current: ~7 images |
| Verification samples | `data/verification_samples/` | First N auto-labeled previews |

Key defaults in `auto_label.py` (kept conservative to avoid box spam):

- score threshold ~0.30
- 640px tiles, 20% overlap
- no full-image pass by default
- reject oversized / strip-shaped boxes (`--max-box-area-ratio`)

```powershell
python auto_label.py --input-dir data/shelf_images --device cpu
python plot_labels.py --image dataset/images/20260507_104718.jpg
```

## What to do next in Step 1

1. Scale Approach B over more of the ~10k shelf images (GPU strongly preferred).
2. Manually correct **40–60** diverse frames in CVAT / X-AnyLabeling → frozen **val** set.
3. Optionally start Approach A once `data/single_products/` is filled.
4. Optional self-training: train a detector, re-label confident predictions, merge, retrain.

## Done criteria for Step 1

- [ ] Hundreds+ of usable labeled shelf images for train
- [ ] Small corrected hold-out val set (never used for training)
- [ ] Visual checks show mostly facing products (bottle / can / carton), few ads / giant boxes

## Related files

- [README](../README.md)
- [Step 2 – Detector](step2_stage1_detector.md)
