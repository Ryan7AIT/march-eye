# Step 2 — Stage 1 Detector (Localization)

**Status: STARTED (smoke test only)**

Train a fast detector that finds products and classifies **packaging only**: bottle, can, carton. It does **not** identify brands or SKUs.

## Why separate Stage 1 from SKU ID

Localization is a small, stable task (3 classes). Brand recognition changes whenever a new product is enrolled — that belongs in Stage 2 embeddings, so you avoid retraining the detector for every SKU.

## Model choice

| Option | Notes |
|--------|--------|
| **YOLOv8n** (current smoke test) | Small and easy via Ultralytics — AGPL unless commercial license |
| **RT-DETR / RF-DETR / D-FINE** | Prefer for production if you need Apache/MIT-style licensing |
| YOLOv10 | Fast, but check license (often AGPL via Ultralytics lineage) |

Roadmap originally suggested YOLOv10 / RT-DETR for permissive licenses. Use RT-DETR (or similar) when you leave the smoke-test phase.

## What is already implemented

| Piece | Location |
|-------|----------|
| Dataset prep + train script | `train_yolo.py` |
| Ultralytics layout | `yolo_data/images/{train,val}`, `yolo_data/labels/{train,val}` |
| Config | `yolo_data/data.yaml` (nc=3: bottle, can, carton) |
| Smoke weights | `runs/.../shelf_nano/weights/best.pt` |

For the tiny set, **train = val** (same 7 images) so the loop can be tested end-to-end. Metrics look optimistic / still weak — not a real eval.

```powershell
# Prepare folders + train nano model
python train_yolo.py --epochs 30 --imgsz 640 --batch 2 --device cpu

# Predict
yolo predict model=runs/detect/runs/shelf_nano/weights/best.pt source=data/shelf_images/20260507_104718.jpg imgsz=640 conf=0.25 save=True
```

If predict shows `(no detections)`, the model is undertrained (few epochs / few images). Try lower `conf` (e.g. 0.01) to diagnose, then retrain longer with more labels.

## Training tips for dense shelves

- Prefer **higher imgsz** (e.g. 1280) once you have a GPU — products are small in the frame.
- Keep mosaic moderate; avoid heavy mixup early on tiny datasets.
- Measure on the **frozen gold val** from Step 1, not on training images.

## What to do next in Step 2

1. Grow labeled data from Step 1; use a real train/val split.
2. Retrain (more epochs, larger imgsz, GPU).
3. Optional self-training: run detector on unlabeled shelves → merge confident boxes → retrain.
4. Swap to a production backbone/license when ready.

## Done criteria for Step 2

- [ ] Solid recall on gold val for bottle / can / carton facings
- [ ] Fast enough for your target hardware
- [ ] Predict outputs usable crops for Stage 2

## Related files

- [README](../README.md)
- [Step 1 – Data](step1_data_pipeline.md)
- [Step 3 – SKU](step3_sku_recognition.md)
