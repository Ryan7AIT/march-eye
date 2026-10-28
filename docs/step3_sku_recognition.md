# Step 3 — Stage 2 Recognition Engine (SKU Identification)

**Status: NOT STARTED**

After Stage 1 finds a box and packaging type, Stage 2 answers: **which SKU is this?** New products must be enrollable without retraining the whole system.

## Idea

1. Crop each detection from the shelf photo.
2. Encode the crop into an embedding vector (OpenCLIP, DINOv3, or a metric-learning ConvNeXt from `timm`).
3. Compare to a **reference gallery** of single-product images with cosine similarity.
4. Return the nearest SKU (or defer to Step 4 if the match is weak).

```
crop → embedding model → vector
                              \
                               → nearest neighbor in gallery → SKU + score
                              /
              gallery vectors (from data/single_products/)
```

## Components to build

### Embedding model

- Prefer open, reusable backbones: **OpenCLIP** or **DINOv3**.
- Fine-tune later on your shelf crops vs studio shots if needed.
- Augment gallery images (blur, glare, slight crop) so embeddings match in-shelf conditions.

### Reference gallery

- Input: clean images in `data/single_products/` (one or more views per SKU).
- Store: vector + `sku_id` + packaging type.
- Start simple with **FAISS / numpy** cosine search; move to **Qdrant** or **Milvus** (Apache-2.0) when the catalog grows.

### Inference logic

1. Stage 1 detects boxes.
2. Crop each box (optional padding).
3. Embed crop.
4. Query gallery for top-1 (or top-k) neighbor.
5. Attach SKU, similarity score, packaging type, bbox, detector confidence.

### Facings

Count detections grouped by SKU → `facings` in the JSON output.

## Suggested JSON shape (draft)

```json
{
  "image": "20260507_104718.jpg",
  "detections": [
    {
      "sku": "RAMY_ORANGE_1L",
      "label": "Ramy Orange 1L",
      "packaging_type": "bottle",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.91,
      "similarity": 0.84,
      "status": "identified"
    }
  ],
  "facings": {
    "RAMY_ORANGE_1L": 4
  }
}
```

## Prerequisites before coding Step 3

- [ ] Stage 1 produces reasonable facing crops (Step 2)
- [ ] `data/single_products/` populated with SKU reference images and an id mapping
- [ ] Decide gallery store (numpy/FAISS first is fine)

## What to implement (order)

1. `embed_gallery.py` — build embeddings from `single_products/`, save index + metadata.
2. `recognize.py` — detect → crop → embed → search → JSON.
3. Calibrate similarity threshold on real shelf crops (do not hardcode 0.70 blindly).

## Done criteria for Step 3

- [ ] Known SKUs matched on real shelf crops with acceptable accuracy
- [ ] JSON includes sku, packaging, bbox, confidence, facings
- [ ] Adding a new gallery image improves recognition without detector retrain

## Related files

- [README](../README.md)
- [Step 2 – Detector](step2_stage1_detector.md)
- [Step 4 – Unknowns](step4_unknown_enrollment.md)
