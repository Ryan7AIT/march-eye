# Step 4 — Unknown Products & Dynamic Enrollment

**Status: NOT STARTED**

Handle products that are not in the reference gallery, and enroll them after human QA — **without retraining** Stage 1 or Stage 2.

## Detecting unknowns

After Stage 2 nearest-neighbor search:

- If best cosine similarity **≥ threshold** → `status: "identified"`, return SKU.
- If best similarity **< threshold** → `status: "unknown"`, `sku: null`.

Example threshold in the roadmap: **0.70**. Treat that as a starting point only — calibrate on real shelf crops (precision vs recall for “unknown”).

```json
{
  "sku": null,
  "packaging_type": "bottle",
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.88,
  "similarity": 0.52,
  "status": "unknown"
}
```

## Clustering for QA (validation file)

Unknown crops arrive one by one. Group them offline so a human can approve a new drink in batch:

1. Collect embeddings of unknown crops over a period / batch of images.
2. Run **DBSCAN** (or similar) in embedding space.
3. Present one cluster as “likely the same new SKU” (e.g. 50 facings).
4. Human confirms label / SKU id / packaging.

## Instant enrollment

When QA confirms a new SKU:

1. Take representative embeddings (or re-embed approved crops / studio shots).
2. **Insert** into the vector gallery with the new `sku_id`.
3. Next inference run recognizes it immediately — no detector or embedder retrain required.

Optional: also add a clean studio photo to `data/single_products/` for a stronger gallery entry.

## Suggested workflow

```
shelf photos
  → Stage 1 + Stage 2
  → unknowns (low similarity)
  → DBSCAN clusters
  → human QA approves SKU
  → insert embeddings into gallery
  → live recognition
```

## What to implement (order)

1. Similarity threshold + `status` field in recognition JSON.
2. Export unknown crops + embeddings to a review folder / DB.
3. Offline `cluster_unknowns.py` (DBSCAN) + simple review UI or CSV.
4. `enroll_sku.py` — write vectors + metadata into the gallery index.

## Done criteria for Step 4

- [ ] Unknowns flagged reliably (few known SKUs marked unknown, few junk marked identified)
- [ ] Clusters group same product for batch QA
- [ ] Enrollment updates gallery and works on the next image without retrain

## Related files

- [README](../README.md)
- [Step 3 – SKU engine](step3_sku_recognition.md)
