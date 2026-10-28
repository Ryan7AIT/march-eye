"""Match a folder of product crops against the FAISS gallery.

Loads gallery/faiss.index + metadata.json, embeds every image in --input
(batched), runs cosine search, and writes recognized vs unknown JSON.

Packaging type comes from gallery folders
(`data/single_products/{bottle|can|carton}/{sku_id}/`), not from the detector.
Search is full-gallery by default. Optional `--packaging-aware` still filters
by a `_(bottle|can|carton)_{conf}` filename suffix when you have one.

Decision rule (inter-product margin):
  1. Retrieve top-k gallery *images*.
  2. Keep the best score per (sku_id, packaging_type).
  3. Recognize only if best_score >= --threshold AND
     (best - second_best) >= --margin.
     Margin is skipped when only one distinct product appears in the candidates.

Example:
    python recognize.py
    python recognize.py --input tmp/crops --gallery gallery --threshold 0.28 --margin 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from embed_gallery import (
    DINOV3_MODEL,
    IMAGE_EXTENSIONS,
    PACKAGING_CLASSES,
    DinoV3Embedder,
    resolve_model_id,
)

PACKAGING_RE = re.compile(
    r"_(?P<packaging>bottle|can|carton)_(?P<conf>\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Similarity-search crops against the FAISS gallery.")
    parser.add_argument("--input", type=Path, default=Path("tmp/crops"), help="Folder of crop images")
    parser.add_argument("--gallery", type=Path, default=Path("gallery"), help="Dir with faiss.index + metadata.json")
    parser.add_argument("--out", type=Path, default=Path("tmp/recognize_results.json"), help="Output JSON path")
    parser.add_argument("--threshold", type=float, default=0.28, help="Min top-1 cosine similarity")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Min inter-product gap (best vs second sku+packaging); ignored with one product",
    )
    parser.add_argument(
        "--packaging-aware",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If the crop filename encodes packaging (e.g. _bottle_0.93), restrict "
            "search to that gallery folder. Off by default: packaging comes from "
            "the matched gallery entry."
        ),
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--model",
        default=DINOV3_MODEL,
        help=(
            "Hugging Face DINO id, or alias dinov3; "
            "must match the model that built the gallery"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--k-search",
        type=int,
        default=DEFAULT_K_SEARCH,
        help="FAISS neighbors retrieved before collapsing to unique products",
    )
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    paths = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")
    return paths


def packaging_from_path(path: Path) -> str | None:
    match = PACKAGING_RE.search(path.stem)
    if not match:
        return None
    return match.group("packaging").lower()


def read_gallery_metadata(gallery_dir: Path) -> dict:
    meta_path = gallery_dir / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing gallery metadata: {meta_path}. Run: python embed_gallery.py"
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def gallery_matches(
    meta: dict,
    model_id: str,
    canvas_width: int,
    canvas_height: int,
) -> bool:
    """True when the saved index was built with this model and canvas."""
    canvas = meta.get("image_canvas") or {}
    try:
        width = int(canvas.get("width"))
        height = int(canvas.get("height"))
    except (TypeError, ValueError):
        return False
    return meta.get("model") == model_id and width == canvas_width and height == canvas_height


def require_gallery_match(
    meta: dict,
    model_id: str,
    canvas_width: int,
    canvas_height: int,
) -> None:
    """Refuse to search an index built with a different embedder or canvas."""
    if gallery_matches(meta, model_id, canvas_width, canvas_height):
        return
    canvas = meta.get("image_canvas") or {}
    raise RuntimeError(
        "Gallery index does not match this embedder "
        f"(index model={meta.get('model')!r} "
        f"canvas={canvas.get('width')}x{canvas.get('height')}, "
        f"embedder model={model_id!r} canvas={canvas_width}x{canvas_height}). "
        "Rebuild with: python embed_gallery.py --source data/single_products --out gallery"
    )


def load_gallery(gallery_dir: Path):
    index_path = gallery_dir / "faiss.index"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Missing gallery files under {gallery_dir} (need faiss.index and metadata.json). "
            "Run: python embed_gallery.py"
        )
    index = faiss.read_index(str(index_path))
    meta = read_gallery_metadata(gallery_dir)
    entries = meta["entries"]
    if index.ntotal != len(entries):
        raise ValueError(
            f"Gallery size mismatch: faiss has {index.ntotal} vectors, metadata has {len(entries)} entries"
        )
    return index, entries


def gallery_vectors(index: faiss.Index) -> np.ndarray:
    """Return all vectors stored in an IndexFlat* as (n, d) float32."""
    return np.vstack([index.reconstruct(i) for i in range(index.ntotal)]).astype(np.float32)


def build_packaging_indices(
    vectors: np.ndarray, entries: list[dict]
) -> dict[str, tuple[faiss.Index, list[int]]]:
    """Map packaging -> (sub-index, list of global entry ids)."""
    by_pack: dict[str, list[int]] = {c: [] for c in PACKAGING_CLASSES}
    for i, entry in enumerate(entries):
        pack = entry["packaging_type"]
        if pack in by_pack:
            by_pack[pack].append(i)

    out: dict[str, tuple[faiss.Index, list[int]]] = {}
    dim = vectors.shape[1]
    for pack, ids in by_pack.items():
        if not ids:
            continue
        sub = faiss.IndexFlatIP(dim)
        sub.add(vectors[ids])
        out[pack] = (sub, ids)
    return out


@torch.inference_mode()
def embed_paths(
    paths: list[Path],
    model,
    preprocess,
    device: str,
    batch_size: int,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for start in tqdm(range(0, len(paths), batch_size), desc="Embedding crops"):
        batch = paths[start : start + batch_size]
        tensors = [preprocess(Image.open(p).convert("RGB")) for p in batch]
        pixel_values = torch.stack(tensors).to(device)
        feats = model.encode_image(pixel_values).float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        vectors.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(vectors)


DEFAULT_K_SEARCH = 32


def rank_skus_from_hits(
    hits: list[tuple[int, float]],
    entries: list[dict],
) -> list[dict]:
    """Collapse image-level FAISS hits into one row per (sku_id, packaging).

    ``hits`` is a list of ``(global_entry_id, similarity)`` sorted by score desc.
    Returns product rows sorted by similarity descending. The same sku_id in
    two packaging folders stays two products.
    """
    best: dict[tuple[str, str], dict] = {}
    for entry_id, score in hits:
        entry = entries[entry_id]
        sku = entry["sku_id"]
        packaging = entry["packaging_type"]
        key = (sku, packaging)
        prev = best.get(key)
        if prev is None or score > prev["similarity"]:
            best[key] = {
                "sku_id": sku,
                "packaging_type": packaging,
                "match_path": entry["image_path"],
                "entry_id": entry_id,
                "similarity": float(score),
            }
    return sorted(best.values(), key=lambda r: r["similarity"], reverse=True)


def search_index_hits(
    query: np.ndarray,
    index: faiss.Index,
    global_ids: list[int] | None,
    k_search: int,
) -> list[tuple[int, float]]:
    """Run FAISS search and return ``(global_entry_id, score)`` hits."""
    k = min(max(k_search, 1), index.ntotal)
    if k <= 0:
        return []
    q = query.reshape(1, -1).astype(np.float32)
    scores, ids = index.search(q, k)
    hits: list[tuple[int, float]] = []
    for score, lid in zip(scores[0], ids[0]):
        if lid < 0:
            continue
        entry_id = global_ids[int(lid)] if global_ids is not None else int(lid)
        hits.append((entry_id, float(score)))
    return hits


def decide_match(
    sku_ranks: list[dict],
    threshold: float,
    margin: float,
) -> dict:
    """Apply threshold + inter-product margin; return status fields for one query."""
    if not sku_ranks:
        return {
            "status": "unknown",
            "reason": "no_match",
            "sku_id": None,
            "packaging_type": None,
            "similarity": None,
            "margin": None,
            "runner_up": None,
            "sku_ranks": [],
        }

    best = sku_ranks[0]
    second = sku_ranks[1] if len(sku_ranks) > 1 else None
    sim1 = best["similarity"]
    gap = (sim1 - second["similarity"]) if second is not None else None

    runner_up = None
    if second is not None:
        runner_up = {
            "sku_id": second["sku_id"],
            "packaging_type": second.get("packaging_type"),
            "match_path": second["match_path"],
            "similarity": round(second["similarity"], 4),
        }

    result = {
        "sku_id": best["sku_id"],
        "packaging_type": best["packaging_type"],
        "match_path": best["match_path"],
        "similarity": round(sim1, 4),
        "margin": None if gap is None else round(gap, 4),
        "runner_up": runner_up,
        "sku_ranks": [
            {
                "sku_id": r["sku_id"],
                "packaging_type": r.get("packaging_type"),
                "similarity": round(r["similarity"], 4),
                "match_path": r["match_path"],
            }
            for r in sku_ranks
        ],
    }

    # Compare with the same rounded gap we expose in JSON so CLI KPIs and
    # status never disagree on razor-thin margins (e.g. 0.04996 -> 0.0500).
    gap_cmp = None if gap is None else round(gap, 4)

    if sim1 < threshold:
        result.update(status="unknown", reason="below_threshold")
    elif gap_cmp is not None and gap_cmp < margin:
        result.update(status="unknown", reason="low_margin")
    else:
        result.update(status="recognized", reason=None)
    return result


def search_top_skus(
    query_paths: list[Path],
    vectors: np.ndarray,
    full_index: faiss.Index,
    pack_indices: dict[str, tuple[faiss.Index, list[int]]],
    entries: list[dict],
    packaging_aware: bool,
    k_search: int = DEFAULT_K_SEARCH,
) -> list[dict]:
    """Return per-query inter-product ranked matches."""
    results: list[dict] = []
    for i, path in enumerate(query_paths):
        q = vectors[i]
        packaging = packaging_from_path(path)

        if packaging_aware and packaging is not None:
            if packaging not in pack_indices:
                results.append(
                    {
                        "packaging": packaging,
                        "sku_ranks": [],
                        "reason": "no_gallery_for_packaging",
                    }
                )
                continue
            index, global_ids = pack_indices[packaging]
            hits = search_index_hits(q, index, global_ids, k_search)
            sku_ranks = rank_skus_from_hits(hits, entries)
            results.append(
                {"packaging": packaging, "sku_ranks": sku_ranks, "reason": None}
            )
            continue

        hits = search_index_hits(q, full_index, None, k_search)
        sku_ranks = rank_skus_from_hits(hits, entries)
        note = "packaging_unknown" if (packaging_aware and packaging is None) else None
        results.append(
            {"packaging": packaging, "sku_ranks": sku_ranks, "reason": note}
        )
    return results


def build_results(
    query_paths: list[Path],
    search_rows: list[dict],
    threshold: float,
    margin: float,
) -> dict:
    recognized: list[dict] = []
    unknown: list[dict] = []

    for path, row in zip(query_paths, search_rows):
        packaging_hint = row["packaging"]
        if row["reason"] == "no_gallery_for_packaging":
            unknown.append(
                {
                    "query_path": path.as_posix(),
                    "query_packaging": packaging_hint,
                    "similarity": None,
                    "margin": None,
                    "reason": "no_gallery_for_packaging",
                }
            )
            continue

        decision = decide_match(row["sku_ranks"], threshold, margin)
        payload = {
            "query_path": path.as_posix(),
            "query_packaging": packaging_hint,
            "sku_id": decision["sku_id"],
            "packaging_type": decision.get("packaging_type"),
            "match_path": decision.get("match_path"),
            "similarity": decision["similarity"],
            "margin": decision["margin"],
        }
        if decision["runner_up"] is not None:
            payload["runner_up"] = decision["runner_up"]

        if decision["status"] != "recognized":
            payload["reason"] = decision["reason"]
            unknown.append(payload)
            continue

        if row["reason"] == "packaging_unknown":
            payload["note"] = "packaging_unknown"
        recognized.append(payload)

    return {
        "recognized": recognized,
        "unknown": unknown,
        "summary": {
            "total": len(query_paths),
            "recognized": len(recognized),
            "unknown": len(unknown),
        },
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable; falling back to cpu")
        device = "cpu"

    query_paths = list_images(args.input)
    index, entries = load_gallery(args.gallery)
    vectors_gallery = gallery_vectors(index)
    pack_indices = build_packaging_indices(vectors_gallery, entries)

    logging.info("Queries: %d images from %s", len(query_paths), args.input)
    logging.info("Gallery: %d vectors from %s", index.ntotal, args.gallery)
    logging.info(
        "Packaging-aware=%s threshold=%.3f inter-product margin=%.3f k_search=%d",
        args.packaging_aware,
        args.threshold,
        args.margin,
        args.k_search,
    )
    for pack in PACKAGING_CLASSES:
        n = pack_indices[pack][0].ntotal if pack in pack_indices else 0
        logging.info("  gallery[%s]=%d", pack, n)

    model_id = resolve_model_id(args.model)
    logging.info("Loading %s on %s", model_id, device)
    embedder = DinoV3Embedder(model_id, device)
    require_gallery_match(
        read_gallery_metadata(args.gallery),
        embedder.model_id,
        embedder.canvas_width,
        embedder.canvas_height,
    )
    query_vectors = embedder.embed_paths(query_paths, args.batch_size)
    search_rows = search_top_skus(
        query_paths,
        query_vectors,
        index,
        pack_indices,
        entries,
        args.packaging_aware,
        k_search=args.k_search,
    )

    payload = {
        "source_dir": args.input.as_posix(),
        "gallery_dir": args.gallery.as_posix(),
        "threshold": args.threshold,
        "margin": args.margin,
        "margin_mode": "inter_sku",
        "k_search": args.k_search,
        "packaging_aware": args.packaging_aware,
        **build_results(query_paths, search_rows, args.threshold, args.margin),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info(
        "Done: %d recognized, %d unknown -> %s",
        payload["summary"]["recognized"],
        payload["summary"]["unknown"],
        args.out,
    )


if __name__ == "__main__":
    main()