"""Shelf-analysis pipeline for the web app.

Loads all models once (YOLO detector + DINOv3 encoder + FAISS gallery)
and exposes a single `analyze()` call used by the FastAPI server:

    shelf photo -> detect boxes -> crop -> embed -> SKU match -> counts
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from embed_gallery import DinoV3Embedder, write_outputs
from recognize import (
    DEFAULT_K_SEARCH,
    decide_match,
    load_gallery,
    rank_skus_from_hits,
    read_gallery_metadata,
    require_gallery_match,
    search_index_hits,
)

RECOGNIZED_COLOR = (76, 175, 80)   # green  (BGR-ish, converted below)
UNKNOWN_COLOR = (0, 152, 255)      # orange


@dataclass
class PipelineConfig:
    # Testing: Ultralytics pretrained COCO weights.
    # Switch back to the shelf detector with:
    yolo_weights: Path = Path("runs/finetune_shelf/weights/best.pt")
    yolo_weights: Path = Path("runs/sku110k_s/weights/best.pt")

    # yolo_weights: Path = Path("yolo26s.pt")
    gallery_dir: Path = Path("gallery")
    embedder_model: str = "runs/dinov3_ft"
    device: str = "cuda"
    imgsz: int = 1024
    det_conf: float = 0.45
    iou: float = 0.5
    max_det: int = 700
    pad: float = 0.05
    threshold: float = 0.70
    margin: float = 0.0005  # inter-product gap: best vs second (sku_id, packaging)
    k_search: int = DEFAULT_K_SEARCH
    results_root: Path = Path("tmp/web")
    keep_last_results: int = 10
    batch_size: int = 32


@dataclass
class GalleryItem:
    sku_id: str
    packaging_type: str
    image_path: str
    count: int = 0


def _catalog_key(sku_id: str, packaging_type: str) -> tuple[str, str]:
    return (sku_id, packaging_type)


class ShelfPipeline:
    """Holds all loaded models; thread-safe single-request inference."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._lock = threading.Lock()

        cfg = self.config
        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            cfg.device = "cpu"
        if cfg.device == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)

        self.detector = YOLO(str(cfg.yolo_weights))
        self.embedder = DinoV3Embedder(cfg.embedder_model, cfg.device)

        self.index, self.entries = load_gallery(cfg.gallery_dir)
        require_gallery_match(
            read_gallery_metadata(cfg.gallery_dir),
            self.embedder.model_id,
            self.embedder.canvas_width,
            self.embedder.canvas_height,
        )

        # One representative image per (sku_id, packaging) for the UI cards.
        self.sku_catalog: dict[tuple[str, str], GalleryItem] = {}
        for entry in self.entries:
            key = _catalog_key(entry["sku_id"], entry["packaging_type"])
            self.sku_catalog.setdefault(
                key,
                GalleryItem(
                    sku_id=entry["sku_id"],
                    packaging_type=entry["packaging_type"],
                    image_path=entry["image_path"],
                ),
            )

        self._warmup()

    def _rebuild_sku_catalog(self) -> None:
        catalog: dict[tuple[str, str], GalleryItem] = {}
        for entry in self.entries:
            key = _catalog_key(entry["sku_id"], entry["packaging_type"])
            catalog.setdefault(
                key,
                GalleryItem(
                    sku_id=entry["sku_id"],
                    packaging_type=entry["packaging_type"],
                    image_path=entry["image_path"],
                ),
            )
        self.sku_catalog = catalog

    def _persist_index(self) -> None:
        """Write the live FAISS index + metadata so the next process sees disk changes."""
        n_total = self.index.ntotal
        dim = self.index.d
        vectors = (
            self.index.reconstruct_n(0, n_total)
            if n_total
            else np.zeros((0, dim), dtype=np.float32)
        )
        records = [
            {
                "sku_id": entry["sku_id"],
                "packaging_type": entry["packaging_type"],
                "image_path": entry["image_path"],
            }
            for entry in self.entries
        ]
        write_outputs(
            self.config.gallery_dir,
            self.index,
            records,
            vectors,
            self.embedder.model_id,
            self.embedder.canvas_width,
            self.embedder.canvas_height,
        )
        self.entries = [
            {
                "id": i,
                "sku_id": rec["sku_id"],
                "packaging_type": rec["packaging_type"],
                "image_path": rec["image_path"],
            }
            for i, rec in enumerate(records)
        ]

    def search_similar(self, image: Image.Image, k_search: int | None = None) -> dict:
        """Embed one photo and return SKU ranks plus matching gallery images."""
        with self._lock:
            vector = self.embedder.embed_pils([image])[0]
            k = k_search or max(self.config.k_search, 48)
            hits = search_index_hits(vector, self.index, None, k)
            sku_ranks = rank_skus_from_hits(hits, self.entries)
            images = []
            for entry_id, score in hits:
                entry = self.entries[entry_id]
                images.append(
                    {
                        "sku_id": entry["sku_id"],
                        "packaging_type": entry["packaging_type"],
                        "image_path": entry["image_path"],
                        "similarity": round(float(score), 4),
                    }
                )
            return {
                "skus": [
                    {
                        "sku_id": row["sku_id"],
                        "packaging_type": row["packaging_type"],
                        "match_path": row["match_path"],
                        "similarity": round(float(row["similarity"]), 4),
                    }
                    for row in sku_ranks
                ],
                "images": images,
            }

    def neighbors_of(self, sku_id: str, packaging_type: str, k: int = 8) -> list[dict]:
        """Nearest other SKUs to this product, using a mean gallery prototype."""
        with self._lock:
            ids = [
                i
                for i, entry in enumerate(self.entries)
                if entry["sku_id"] == sku_id and entry["packaging_type"] == packaging_type
            ]
            if not ids or self.index.ntotal == 0:
                return []
            vectors = self.index.reconstruct_n(0, self.index.ntotal)
            proto = vectors[ids].mean(axis=0)
            norm = float(np.linalg.norm(proto))
            if norm > 0:
                proto = proto / norm
            hits = search_index_hits(
                proto.astype(np.float32),
                self.index,
                None,
                min(self.index.ntotal, max(self.config.k_search, len(ids) + k * 8)),
            )
            ranks = rank_skus_from_hits(hits, self.entries)
            out = []
            for row in ranks:
                if row["sku_id"] == sku_id and row["packaging_type"] == packaging_type:
                    continue
                out.append(
                    {
                        "sku_id": row["sku_id"],
                        "packaging_type": row["packaging_type"],
                        "match_path": row["match_path"],
                        "similarity": round(float(row["similarity"]), 4),
                    }
                )
                if len(out) >= k:
                    break
            return out

    def add_gallery_images(self, records: list[dict], images: list[Image.Image]) -> None:
        if not records:
            return
        if len(records) != len(images):
            raise ValueError("records and images must be the same length.")
        with self._lock:
            vectors = self.embedder.embed_pils(images)
            self.index.add(vectors)
            for rec in records:
                self.entries.append(
                    {
                        "id": len(self.entries),
                        "sku_id": rec["sku_id"],
                        "packaging_type": rec["packaging_type"],
                        "image_path": rec["image_path"],
                    }
                )
                key = _catalog_key(rec["sku_id"], rec["packaging_type"])
                self.sku_catalog.setdefault(
                    key,
                    GalleryItem(
                        sku_id=rec["sku_id"],
                        packaging_type=rec["packaging_type"],
                        image_path=rec["image_path"],
                    ),
                )
            self._persist_index()

    def remove_gallery_paths(self, image_paths: list[str]) -> None:
        drop = set(image_paths)
        if not drop:
            return
        with self._lock:
            keep = [i for i, entry in enumerate(self.entries) if entry["image_path"] not in drop]
            if len(keep) == len(self.entries):
                return
            import faiss

            dim = self.index.d
            if keep:
                vectors = self.index.reconstruct_n(0, self.index.ntotal)[keep]
                new_index = faiss.IndexFlatIP(dim)
                new_index.add(np.ascontiguousarray(vectors, dtype=np.float32))
            else:
                new_index = faiss.IndexFlatIP(dim)
            self.index = new_index
            self.entries = [self.entries[i] for i in keep]
            self._rebuild_sku_catalog()
            self._persist_index()

    def rename_gallery_sku(
        self,
        sku_id: str,
        packaging_type: str,
        new_sku_id: str,
        new_packaging_type: str,
        path_map: dict[str, str],
    ) -> None:
        with self._lock:
            for entry in self.entries:
                if entry["sku_id"] == sku_id and entry["packaging_type"] == packaging_type:
                    entry["sku_id"] = new_sku_id
                    entry["packaging_type"] = new_packaging_type
                    mapped = path_map.get(entry["image_path"])
                    if mapped:
                        entry["image_path"] = mapped
            self._rebuild_sku_catalog()
            self._persist_index()

    def _predict_kwargs(self) -> dict:
        cfg = self.config
        return {
            "imgsz": cfg.imgsz,
            "conf": cfg.det_conf,
            "iou": cfg.iou,
            "max_det": cfg.max_det,
            "device": cfg.device,
            "verbose": False,
        }

    def _warmup(self) -> None:
        """Run a dummy pass so the first user request is fast."""
        dummy = np.zeros((self.config.imgsz, self.config.imgsz, 3), dtype=np.uint8)
        self.detector.predict(source=dummy, **self._predict_kwargs())
        with torch.inference_mode():
            pil = Image.fromarray(cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB))
            self.embedder.embed_pils([pil])

    # ------------------------------------------------------------------ #

    def analyze(self, image_bytes: bytes) -> dict:
        with self._lock:
            return self._analyze(image_bytes)

    def _analyze(self, image_bytes: bytes) -> dict:
        cfg = self.config
        t_start = time.perf_counter()

        data = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode the uploaded file as an image.")
        height, width = image.shape[:2]

        request_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        out_dir = cfg.results_root / request_id
        crops_dir = out_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        # --- Stage 1: detect (boxes only) --------------------------------
        t0 = time.perf_counter()
        results = self.detector.predict(source=image, **self._predict_kwargs())
        detect_ms = (time.perf_counter() - t0) * 1000

        detections: list[dict] = []
        crop_paths: list[Path] = []
        boxes = results[0].boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                x1f, y1f, x2f, y2f = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())

                x1, y1, x2, y2 = _expand_box(
                    x1f, y1f, x2f, y2f, cfg.pad, width, height
                )
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_name = f"{i:04d}_{conf:.2f}.jpg"
                crop_path = crops_dir / crop_name
                cv2.imwrite(str(crop_path), crop)
                # Embed the written JPEG, the same pixels test_similarity.py sees.
                if not crop_path.is_file():
                    continue
                crop_paths.append(crop_path)
                detections.append(
                    {
                        "id": i,
                        "packaging": None,
                        "det_conf": round(conf, 3),
                        "box": [x1, y1, x2, y2],
                        "crop_url": f"/results/{request_id}/crops/{crop_name}",
                    }
                )

        # --- Stage 2: embed + match --------------------------------------
        t0 = time.perf_counter()
        if crop_paths:
            vectors = self.embedder.embed_paths(crop_paths, cfg.batch_size)
            self._match(detections, vectors)
        embed_ms = (time.perf_counter() - t0) * 1000

        # --- Aggregate ----------------------------------------------------
        gallery = {
            key: GalleryItem(g.sku_id, g.packaging_type, g.image_path)
            for key, g in self.sku_catalog.items()
        }
        recognized_total = 0
        unknown_total = 0
        for det in detections:
            if det["status"] == "recognized":
                key = _catalog_key(det["sku_id"], det["packaging"])
                item = gallery.get(key)
                if item is not None:
                    item.count += 1
                recognized_total += 1
            else:
                unknown_total += 1

        plain_name = "plain.jpg"
        cv2.imwrite(str(out_dir / plain_name), image)
        annotated = _draw_annotated(image, detections)
        annotated_name = "annotated.jpg"
        cv2.imwrite(str(out_dir / annotated_name), annotated)

        self._prune_old_results(keep=request_id)
        total_ms = (time.perf_counter() - t_start) * 1000

        return {
            "request_id": request_id,
            "image": {
                "width": width,
                "height": height,
                "plain_url": f"/results/{request_id}/{plain_name}",
                "annotated_url": f"/results/{request_id}/{annotated_name}",
            },
            "summary": {
                "total_detections": len(detections),
                "recognized": recognized_total,
                "unknown": unknown_total,
            },
            "gallery": [
                {
                    "sku_id": g.sku_id,
                    "packaging_type": g.packaging_type,
                    "image_url": "/gallery-media/" + g.image_path,
                    "count": g.count,
                }
                for g in sorted(
                    gallery.values(),
                    key=lambda g: (-g.count, g.sku_id, g.packaging_type),
                )
            ],
            "unknown_counts": {"unknown": unknown_total},
            "detections": detections,
            "timings_ms": {
                "detect": round(detect_ms, 1),
                "recognize": round(embed_ms, 1),
                "total": round(total_ms, 1),
            },
        }

    # ------------------------------------------------------------------ #

    def _match(self, detections: list[dict], vectors: np.ndarray) -> None:
        """Full-gallery inter-product search; fills status/sku fields in place.

        Retrieves top-k gallery *images*, keeps the best score per
        (sku_id, packaging_type), then applies threshold + inter-product margin.
        Packaging is copied from the matched gallery folder, not from YOLO.
        """
        cfg = self.config
        for det, vector in zip(detections, vectors):
            hits = search_index_hits(vector, self.index, None, cfg.k_search)
            sku_ranks = rank_skus_from_hits(hits, self.entries)
            decision = decide_match(sku_ranks, cfg.threshold, cfg.margin)

            det["sku_id"] = decision["sku_id"]
            det["similarity"] = decision["similarity"]
            det["margin"] = decision["margin"]
            if decision["runner_up"] is not None:
                det["runner_up"] = decision["runner_up"]
            det["status"] = decision["status"]
            det["reason"] = decision["reason"]
            if decision["status"] == "recognized":
                det["packaging"] = decision.get("packaging_type")
                det["match_path"] = decision.get("match_path")
            else:
                det["packaging"] = None

    def _prune_old_results(self, keep: str) -> None:
        root = self.config.results_root
        if not root.is_dir():
            return
        dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)
        excess = len(dirs) - self.config.keep_last_results
        for old in dirs[:max(excess, 0)]:
            if old.name == keep:
                continue
            for f in old.rglob("*"):
                if f.is_file():
                    f.unlink(missing_ok=True)
            for d in sorted(old.rglob("*"), reverse=True):
                if d.is_dir():
                    d.rmdir()
            old.rmdir()


def _expand_box(
    x1: float, y1: float, x2: float, y2: float, pad: float, w: int, h: int
) -> tuple[int, int, int, int]:
    dx = (x2 - x1) * pad
    dy = (y2 - y1) * pad
    x1 = max(0, min(int(x1 - dx), w - 1))
    y1 = max(0, min(int(y1 - dy), h - 1))
    x2 = max(x1 + 1, min(int(x2 + dx), w))
    y2 = max(y1 + 1, min(int(y2 + dy), h))
    return x1, y1, x2, y2


def _draw_annotated(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw boxes only. Green = recognized, orange = unknown. No SKU text."""
    annotated = image.copy()
    thickness = max(2, round(min(image.shape[:2]) / 400))
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = RECOGNIZED_COLOR if det["status"] == "recognized" else UNKNOWN_COLOR
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
    return annotated
