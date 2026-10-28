# Improvement plan (2026-10-20)

This file builds on `ROADMAP.md` (2026-09-25). Since then the catalog has grown (1.25 L SKUs added, 438 images / 112 products), `train_embedder.py` exists and produced `runs/dinov3_ft`, and the YOLO detector was fine-tuned on `dataset/`. Every number below was measured today on the current files; the scripts are in `tmp/analysis_stage2.py` and the commands in the appendix.

---

## TL;DR - top 10 actions, in order


| #   | Action                                                                                                | Stage | Effort   | Why (measured)                                                                                                                                                                      |
| --- | ----------------------------------------------------------------------------------------------------- | ----- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Fix / delete the 4 broken YOLO label files and the 2 duplicate images                                 | 1     | 1 h      | `1.jpg`, `19.jpg`, `20.jpg`, `21.jpg` labels agree with <15% of real products; `20.jpg == 22.jpg` and `21.jpg == 24.jpg` (byte-identical) with different labels; `22.jpg` is in val |
| 2   | Clean `data/eval/` (move enrolled products out of `unknown/`, drop self-duplicates, add missing SKUs) | 2     | 2 h      | at least 6 of the top-16 "unknown" crops are now catalog products; 9 known crops are byte-identical to gallery images                                                               |
| 3   | Merge duplicate SKU folders and fix names                                                             | 2     | 1 h      | `carton/ramy_1l_annans` vs `ramy_annans_1l_carton`, `ramy_1l_orange` vs `ramy_orange_1l` appear as "errors" and are being *pushed apart* by fine-tuning                             |
| 4   | Switch product scoring from max-over-images to a per-SKU prototype                                    | 2     | 30 min   | top-1 82.9% -> 85.7% (fine-tuned), 75.2% -> 81.9% (base); top-5 -> 100%                                                                                                             |
| 5   | Raise `PipelineConfig.threshold` from 0.70 to ~0.85 and margin from 0.0005 to ~0.02                   | 2     | 5 min    | at 0.70, 44% of unknown crops are accepted; at 0.85 about 10% (and that set is contaminated, so real is lower)                                                                      |
| 6   | Add a "facing / partial" 2-class detector instead of treating half products as background             | 1     | 1 day    | keeps SKU-110K recall while still filtering partials                                                                                                                                |
| 7   | Detect at higher resolution (imgsz 1280-1536 or tiled/SAHI)                                           | 1     | 0.5 day  | recall is 0.72; almost all misses are small, far-aisle products                                                                                                                     |
| 8   | Size-aware hard negatives + held-out SKUs in `train_embedder.py`                                      | 2     | 1-2 days | 11 of the 15 remaining errors are 1.25 L vs 2 L of the same flavor                                                                                                                  |
| 9   | Add OCR / box-size re-ranking only between size siblings                                              | 2     | 2-3 days | artwork is identical across sizes; the embedding cannot fix it                                                                                                                      |
| 10  | Label the other ~28 shelf photos with model-assisted labeling                                         | 1     | 1-2 days | 28 unique labeled photos is the main Stage 1 limit                                                                                                                                  |


---



## 1. What I measured today



### Stage 1 - detector

Same 5 validation images (`yolo_data/finetune/val.txt`), imgsz 940, `conf=0.001`:


| Weights                                                  | Precision | Recall | mAP50     | mAP50-95  |
| -------------------------------------------------------- | --------- | ------ | --------- | --------- |
| `runs/sku110k/weights/best.pt` (SKU-110K only)           | 0.634     | 0.720  | 0.667     | 0.379     |
| `runs/finetune_shelf/weights/best.pt` (+ your 25 images) | **0.790** | 0.724  | **0.768** | **0.462** |


- Fine-tuning helps, and it helps the way you intended: precision goes up (fewer half products / non-facings), recall stays the same.
- **Recall is the weak point (0.72).** The overlay for `3.jpg` shows the misses are almost all small products at the far end of the aisle, seen at a steep angle. At `imgsz=1024` a 4080 px photo is shrunk 4x, so those products are ~30-40 px wide.
- **The validation number is not trustworthy yet.**
  - `22.jpg` (val) is the same file as `20.jpg` (train).
  - `19.jpg` (val) has shifted / wrong boxes (only 7% of its labels overlap any detection; many boxes sit on empty shelf or floor).
- **Label audit of all 30 images** (fraction of labeled boxes that a generous SKU-110K detector also finds at IoU >= 0.5):
  - Broken: `21.jpg` 0.05, `19.jpg` 0.07, `1.jpg` 0.08, `20.jpg` 0.14.
  - Check by eye: `29.png` 0.42, `7.jpg` 0.53, `4.jpg` 0.62.
  - The rest are 0.66-0.95, which is normal given the facing-only policy.
  - `20.jpg` and `21.jpg` look like the *old* labels of photos that you re-labeled as `22.jpg` and `24.jpg`. Right now the model is trained on the same photo with two conflicting label sets.
- **Training detail:** after epoch 1 of the fine-tune, mAP50 fell to 0.02 and needed ~5 epochs to recover. That is the Ultralytics warmup (`warmup_bias_lr=0.1`, 3 warmup epochs) with a tiny batch. It erases part of what SKU-110K taught.
- The SKU-110K stage itself was trained on a 750-image subset, at `imgsz=640`, for 13 of 20 epochs. Inference runs at 1024.



### Stage 2 - embedding + matching

Eval: 105 known crops whose SKU is in the gallery (4 more have no gallery folder), 61 "unknown" crops; byte-identical gallery images removed from each query's search.


| Embedder                                 | Scoring                         | Top-1     | Top-5    |
| ---------------------------------------- | ------------------------------- | --------- | -------- |
| DINOv3 base                              | max over images (current)       | 75.2%     | 92.4%    |
| DINOv3 base                              | per-SKU prototype (mean vector) | 81.9%     | 99.0%    |
| **DINOv3 fine-tuned** (`runs/dinov3_ft`) | max over images (current)       | 82.9%     | 99.0%    |
| **DINOv3 fine-tuned**                    | **per-SKU prototype**           | **85.7%** | **100%** |


- **Your fine-tune works.** It adds ~8 points top-1, and it lowers the score of genuinely new products:

  | New product (not in catalog)                    | Nearest SKU           | Base  | Fine-tuned |
  | ----------------------------------------------- | --------------------- | ----- | ---------- |
  | Ramy Citron 1.25 L (`unknown/234567.png`)       | a Ramy 1.25 L sibling | 0.917 | 0.840      |
  | Ramy Light Orange 1.25 L (`unknown/789542.png`) | a Ramy orange sibling | 0.926 | 0.786      |

- **What is still wrong (fine-tuned + prototype, 15 errors):**
  - 11 are **same flavor, other size** (1.25 L <-> 2 L). Examples: `ramy_2l_orange_banane -> ramy_125l_orange_bannane` (x4), `ramy_2l_orange_peche -> ramy_125l_orange_peche` (x3).
  - 4 are same size, other flavor (e.g. `ramy_300ml_abricot -> ramy_300ml_peche`).
  - No cross-brand errors remain.
- **Similarity distributions (fine-tuned, max scoring):**

  | Group                | p10   | median | p90   |
  | -------------------- | ----- | ------ | ----- |
  | known, correct top-1 | 0.888 | 0.929  | 0.976 |
  | known, wrong top-1   | 0.860 | 0.916  | 0.948 |
  | unknown              | 0.581 | 0.681  | 0.847 |

  - Threshold 0.85: 82% of known accepted correctly, 16% accepted wrong, 10% of unknowns accepted.
  - Production uses **0.70** (44% of unknowns accepted) and margin **0.0005** (effectively off).
  - Margin (best - second product): median **0.052** when correct vs **0.018** when wrong, so a ~0.02 margin rejects about half the wrong answers.
- **Eval-set problems** (they make the numbers above pessimistic for unknowns and slightly optimistic for knowns):
  - `data/eval/unknown/` contains products that are now in the gallery: `image.png` (Toudja Pêche Mangue, sim 1.00), `hgsdfsh.png` (Ramy Milky Fraise 1 L), `10-4.jpg` (Frutty Ananas), `19-53.jpg` (Tchina Cocktail), `19-5.jpg` (Rouiba ananas carton), plus several Toudja / Frutty / Rouiba crops scoring 0.78-0.87.
  - Known SKUs with no gallery folder (counted as skipped by `eval_similarity.py`): `ramy_125l_menth`, `carton/ramy_20cl_fraise`, `carton/ramy_20cl_orange`, `ramy_2l_peche` (folder exists but is empty).
  - 9 known eval crops are byte-identical to gallery images.
  - 148 gallery images were saved from the web app and carry no source-photo id, so a "never test on a photo you trained on" split cannot be enforced.

---



## 2. Answers to your questions



### Q1. COCO YOLO -> SKU-110K -> my ~30 images. Right approach?

**Yes, the order is right; keep it.** Measured: SKU-110K alone 0.667 mAP50, after your fine-tune 0.768. Using plain COCO YOLO would be worse. COCO's `bottle` class is trained on sparse scenes, it has no carton class, and it is not used to 100+ touching objects per image. SKU-110K gives the "dense shelf" prior and your images teach the "facing only" policy. That is standard transfer learning.

What to change:

1. **Fix the data before anything else.** With 28 unique photos, 4 broken label files are ~15% of the dataset and one of 5 val images. Delete `20.jpg` / `21.jpg` (keep `22` / `24`), re-label `1.jpg` and `19.jpg`, re-check `29.png`, `7.jpg`, `4.jpg`.
2. **Label "partial" as a second class instead of background.** This is the most important modeling change. SKU-110K taught the model that every product, including half-visible ones, is an object. Your fine-tune tries to un-teach that with 25 images, which conflicts with the prior and costs recall. Instead:
  - class 0 `facing`, class 1 `partial` (cut by the image border, mostly hidden behind another product, top-row products seen from below, etc.);
  - drop class 1 at inference (or show it greyed out in the UI).
  - The model keeps finding all products (recall), and the facing/partial decision becomes a classification it can learn from few examples.
  - Alternative if you do not want to re-label: keep 1 class, and add a tiny crop classifier (`facing` vs `partial`) after the detector, trained on crops you already have.
3. **Write a one-page labeling policy** (what counts as a facing: % visible, far-aisle products yes/no, minimum size in px, back-row products). Inconsistent policy is invisible noise. If far-aisle products are too small for Stage 2 to recognize anyway, it is fine to define them as "not counted", but then do it consistently in labels *and* filter by box size at inference.
4. **Fix the resolution mismatch.**
  - Retrain the SKU-110K stage on more of the dataset (all 8k train images, or at least 3-4k) at `imgsz` 1024-1280. Your MX350 (2 GB) cannot do this; use Colab / Kaggle (free T4 / P100) for this one run.
  - Fine-tune and run inference at the same size.
  - For far-aisle products, try tiled inference (SAHI, 2x2 tiles with 20% overlap + merged NMS) and compare recall.
5. **Fine-tune settings** in `fine_tune_yolo.py`:
  - `warmup_epochs=0` (or `warmup_bias_lr=0.0`) to avoid the epoch-1 collapse;
  - try `freeze=0` with `lr0=2e-4` vs the current `freeze=10`;
  - light mosaic (`mosaic=0.5`, `close_mosaic=10`) is fine for dense shelves; keep `fliplr` (flips do not matter for boxes, unlike for the embedder).
6. **Grow the labeled set with model-assisted labeling.** You have 61 shelf photos and only 28 unique labeled ones.
  - Run the current detector on the unlabeled photos, import predictions into CVAT / Label Studio, and correct (~10 min per photo instead of 40).
  - Target 80-100 photos from at least 5 different stores; val = 10-15 photos from stores *not* in training.
7. **Evaluate the detector the way Stage 2 uses it:** recall of facings at IoU >= 0.5, plus a box-tightness check (truncated caps and neighbor bleed hurt the embedding).
8. **License:** Ultralytics is AGPL. If this becomes a product, plan RT-DETR / RF-DETR (Apache-2.0) or buy a license. With the same data, RT-DETR-L typically matches YOLO-s/m on dense scenes.



### Q2a. Dataset advice for a good embedding model

**Catalog / labels**

- **One SKU = one folder, one name.**
  - Merge `carton/ramy_1l_annans` + `carton/ramy_annans_1l_carton`, and `carton/ramy_1l_orange` + `carton/ramy_orange_1l` (check by eye first).
  - Rename `carton/ramy_1l_fraise` to the Milky name it really is, and fix `bottle/ramy_1ml_milky_fraise` (1 ml -> 1 l). Check `ramy_125_coctail` vs `ramy_125l_coctail_tropical`.
  - Split `bottle/tchina` (16 images) into flavors: it is a brand, not a SKU.
  - **Duplicates are actively harmful with metric learning**: SupCon pushes the two folders of the *same* product apart, which damages the space around them.
- **Use** `data/catalog.csv` **as the source of truth** (it exists but is UTF-16 and incomplete). Columns: `sku_id, brand, line, flavor, volume_ml, packaging`. Use a fixed naming pattern `brand_line_flavor_volume` and have `embed_gallery.py` warn on folders that are not in the catalog.

**Images per SKU**

- 26 products have a single image and one (`ramy_2l_peche`) has none. Aim for **at least 5-10 crops per SKU from at least 3 different photos** (different stores, lights, angles, distances). Variety of *photos* matters more than count.
- Full-resolution crops only; avoid screenshots and tiny crops (< 70 px wide).
- Keep a small curated **reference** set per SKU for the gallery (3-5 clean, full, front-facing crops), and a larger, messier **training** set. They can overlap; the eval set must not.

**Splits and provenance**

- **Record the source photo for every crop.** Change `enroll.py` to name files `{photo_hash}_{request_id}_{crop}` or write a sidecar JSON. Without it you cannot split by photo, and same-photo crops look nearly identical (lighting + angle), which inflates eval.
- Eval rule: known crops come only from photos (better: stores) that contributed nothing to the gallery or the training set.
- `unknown/` should contain mostly the costly case: **new variants of known brands** (new Ramy flavor or size, Ramy Light, etc.), plus some other brands. Right now it is mostly other brands, which is the easy case.

**Negatives**

- Build a **negatives pool** of products you will never enroll (Coca-Cola, Pepsi, Fanta, water, Hamoud, Ifruit flavors you do not track...). You already have ~1,900 unlabeled detector crops under `runs/detect/*/crops` and `tmp/web/*/crops`. Use them in training as negatives only (never as positives, never in the gallery). This directly lowers unknown similarity (see Q2b).



### Q2b. "New products score 80% vs an existing SKU; can I get them to 60-70%?"

Short answer: **for other-brand products, yes, and you are almost there. For a new flavor of a known line, partly. For a new size of a known flavor, no, not with an image embedding alone.** And the absolute value is the wrong target: what matters is the *gap* between "same product" scores and "different product" scores.

Why:

1. **Cosine values are model-specific.** A different loss or temperature can shift all scores up or down. What you need is that known-correct scores (today p10 = 0.89) sit clearly above unknown scores (today median 0.68, p90 0.85). Then any threshold in the gap works, whether it is 0.65 or 0.88.
2. **Three kinds of "unknown" behave differently:**

  | Kind                           | Example                      | Today (fine-tuned) | Reachable target           | How                                                     |
  | ------------------------------ | ---------------------------- | ------------------ | -------------------------- | ------------------------------------------------------- |
  | Other brand / different design | Fanta, Rouiba not in catalog | median ~0.68       | < 0.6                      | negatives pool + margin loss                            |
  | New flavor of a known line     | Ramy Citron 1.25 L           | 0.84               | ~0.7-0.75                  | sibling hard negatives, margin loss, local verification |
  | New size of a known flavor     | Ramy Orange Banane 1.5 L     | ~0.9+              | not reachable by embedding | OCR on volume text + box size                           |

   A new flavor of Ramy shares the bottle, cap, logo and label layout, which is 70-90% of the pixels. A global vector will always find it close. A new *size* has the same artwork, so the embedding *should* say "same"; the difference is in the volume text and the physical size.

What to do, in order:

1. **Make the decision rule smarter than one threshold** (no retraining):
  - Prototype scoring (+3 points top-1, measured).
  - Threshold ~0.85 **and** margin ~0.02 vs the second-best product.
  - **Family-aware unknown:** if the top-k candidates are siblings (same brand + line) and the margin is small, return `unknown_variant_of: ramy_125l` instead of guessing. This is exactly the "new product of a known family" case and it is what Step 4 enrollment needs.
  - **Per-SKU thresholds:** SKUs with tight clusters (consistent photos) can use a higher threshold than SKUs with spread-out images. Set threshold_sku = mean - 2*std of that SKU's leave-one-out scores, clipped to a global floor.
  - **Score normalization against the negatives pool** (like "s-norm" in face / speaker verification): `score' = sim(q, sku) - mean(top-10 sim(q, negatives))`. Products that resemble *many* things (hubs, like `tchina` was) get penalized automatically.
  - **Calibrate**: fit a logistic regression on `[sim1, margin, sim1 - sim_negatives, family_match]` -> P(correct) using the eval set; threshold the probability. It is easier to reason about than raw cosine.
2. **Train for a margin, not only for ranking** (`train_embedder.py`):
  - Add an **ArcFace / CosFace** head (sub-center ArcFace, `s=30`, `m=0.3-0.4`), alone or together with SupCon. Additive-margin losses explicitly force a gap between classes; they are what product-matching competition winners (Shopee, Google Landmarks, Products-10K) used.
  - Add the **negatives pool** as extra "background" samples in every batch: they appear only as negatives in SupCon (or as one or more `other` classes for ArcFace that are never put in the gallery).
  - **Fix the family sampler.** `family_of()` uses only the brand token, so "ramy" is one huge family and size siblings rarely meet in a batch. Use catalog attributes and alternate between "same brand + size, different flavor" and "same flavor, different size" batches.
  - **Hold out 15-20% of SKUs** (whole families, e.g. all `ramy_300ml_milky_`*) during training and evaluate on them as "enrolled after training". That measures the real scenario: new products added without retraining.
  - **Stop memorizing.** 600 steps x 72 images on 430 images is ~100 epochs; final in-batch recall 0.996. Add a periodic eval on `data/eval` and keep the best checkpoint, not the last.
  - **WiSE-FT:** average the fine-tuned weights with the base weights (`w = a*ft + (1-a)*base`, try a = 0.5-0.8). It usually keeps most of the fine-tune gain and generalizes better to unseen products. It is a 10-line script and needs no training.
  - Optional: a small projection head (768 -> 256) trained on top; lower temperature (0.05).
3. **Second-stage verification for close calls only** (margin < 0.05 and top-2 are siblings):
  - **OCR** on the label band of the full-resolution crop (PaddleOCR or Florence-2) for volume tokens (`1,25 L`, `2L`, `25cl`, `300ml`) and flavor words. Only a *contradicting* read should penalize a candidate.
  - **Box-size prior** within the same photo / shelf row (normalized by median bottle width): separates <= 330 ml from >= 1 L reliably; 1.25 L vs 2 L only sometimes.
  - **Local feature matching** (DINOv3 patch tokens or SuperPoint + LightGlue) between the crop and the top-2 references: catches flavor-illustration differences that a global vector blurs.

---



## 3. Stage 1 checklist (detector)

- [ ] Delete `dataset/images/20.jpg`, `21.jpg` (+ labels). Re-label `1.jpg`, `19.jpg`. Re-check `29.png`, `7.jpg`, `4.jpg`.
- [ ] Remove the leftover class ids and polygons in `dataset/labels` (classes 0/1/2 and 13-59-column polygon lines are mixed today; the script converts them, but a clean source avoids surprises). Or, better, convert to `0=facing, 1=partial`.
- [ ] Add a hash-dedupe check to `fine_tune_yolo.py` (`write_split` should refuse identical images in train and val).
- [ ] Val: >= 10 photos from stores not in training; report recall at the pipeline `conf`.
- [ ] `warmup_epochs=0`; try `freeze=0, lr0=2e-4`; keep mosaic light.
- [ ] Retrain SKU-110K stage at imgsz >= 1024 on (much) more of the dataset (cloud GPU).
- [ ] Try SAHI / tiled inference and imgsz 1280-1536 for far-aisle recall; pick by recall vs time.
- [ ] Sweep `det_conf` (0.25-0.5) on the val set; 0.45 was not tuned on your data.
- [ ] Model-assisted labeling of the remaining shelf photos.
- [ ] Longer term: RT-DETR / RF-DETR for licensing; shelf-row detection for context (Stage 2 benefits too).



## 4. Stage 2 checklist (recognition)

**Data and eval (do first)**

- [ ] Clean `data/eval/unknown/` (move enrolled products to `Known/`), remove byte-identical gallery copies, create gallery folders for the 4 missing known SKUs or drop those crops.
- [ ] Merge duplicate SKU folders; fix names; split `tchina`; fill `ramy_2l_peche`.
- [ ] `eval_similarity.py`: count missing-from-gallery as a miss (do not skip), save top-5 per row, report errors by type (size / flavor / brand) using `catalog.csv`, and skip gallery images identical to the query (what `tmp/analysis_stage2.py` does).
- [ ] `enroll.py`: record source photo id; never save eval crops into the gallery.
- [ ] Rebuild `gallery/` with `runs/dinov3_ft` (the pipeline already points at it; make sure the index was built with the same model; `require_gallery_match` checks this).

**Decision rule (no training)**

- [ ] Prototype scoring in `rank_skus_from_hits` (for SKUs with many varied images, 2-3 k-means sub-prototypes).
- [ ] Threshold ~0.85, margin ~0.02 as a starting point; re-sweep after cleaning eval. Use the same defaults in `recognize.py`, `test_similarity.py`, `pipeline.py`.
- [ ] Family-aware `unknown_variant` status.
- [ ] Negatives-pool score normalization.

**Training (**`train_embedder.py`**)**

- [ ] Catalog-based family sampler (size siblings + flavor siblings).
- [ ] Negatives pool in batches.
- [ ] ArcFace / CosFace option.
- [ ] Held-out SKUs + periodic eval + best checkpoint.
- [ ] WiSE-FT weight averaging experiment.
- [ ] Optional: canvas 288x832 (more pixels for the volume text) if a bigger GPU is available.

**Re-ranking**

- [ ] OCR probe on the size-sibling errors (11 crops listed above) before building anything.
- [ ] Box-size prior per photo/row.
- [ ] Patch-level verification for low-margin sibling pairs.

---



## 5. Suggested experiment order and success criteria

1. **Data hygiene day** (actions 1-3). Re-run both evals. Expect the detector val mAP to move (it was measuring label noise) and unknown false-accept to drop.
2. **Free wins** (actions 4-5): prototype + threshold/margin. *Success:* known accepted-correct >= 80% with unknown accepted <= 5% on the cleaned eval.
3. **Detector v2:** facing/partial classes, clean labels, warmup fix, higher res or tiles. *Success:* recall >= 0.85 at precision >= 0.85 on a store-held-out val.
4. **Embedder v2:** sampler + negatives + held-out SKUs + ArcFace, then WiSE-FT. *Success:* on held-out SKUs, sibling unknowns (new flavor of a known line) score below the known-correct p10; top-1 on known >= 90%.
5. **Size re-ranking** (OCR + geometry) once 1-4 are stable. *Success:* size-sibling errors cut by at least half.

---

