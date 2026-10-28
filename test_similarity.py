"""Similarity-search one product photo against the FAISS gallery and print KPIs.

Use this to debug Stage 2 on a single crop / reference image without running YOLO.
Crop filenames no longer encode packaging; search is the full gallery unless
you pass --packaging. Packaging on the hit comes from the gallery folder.

Example:
    python test_similarity.py tmp/web/20260721_182158_3e537f/crops/0007_0.93.jpg
    python test_similarity.py path/to/photo.jpg --packaging bottle --top-k 5
    python test_similarity.py path/to/photo.jpg --out tmp/sim_kpis.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from embed_gallery import (
    DINOV3_MODEL,
    IMAGE_EXTENSIONS,
    PACKAGING_CLASSES,
    DinoV3Embedder,
    build_faiss_index,
    resolve_model_id,
)
from recognize import (
    DEFAULT_K_SEARCH,
    build_packaging_indices,
    decide_match,
    gallery_matches,
    gallery_vectors,
    load_gallery,
    packaging_from_path,
    rank_skus_from_hits,
    read_gallery_metadata,
    search_index_hits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed one photo and similarity-search the gallery (inter-SKU KPIs)."
    )
    parser.add_argument("image", type=Path, help="Path to a product photo / crop")
    parser.add_argument("--gallery", type=Path, default=Path("gallery"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--model",
        default=DINOV3_MODEL,
        help=(
            "Hugging Face DINO id, or alias dinov3 "
            f"(default: {DINOV3_MODEL})"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size when re-embedding the gallery for a model that did not build the index",
    )
    parser.add_argument("--threshold", type=float, default=0.28)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Min inter-SKU gap (best_sku - second_best_sku)",
    )
    parser.add_argument(
        "--k-search",
        type=int,
        default=DEFAULT_K_SEARCH,
        help="FAISS neighbors before collapsing to unique SKUs",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many SKUs to show in the ranking table",
    )
    parser.add_argument(
        "--packaging",
        choices=PACKAGING_CLASSES,
        default=None,
        help="Optional gallery packaging filter (default: parse from filename, else search all)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path (default: tmp/sim_<stem>.json)",
    )
    return parser.parse_args()


def resolve_packaging(args: argparse.Namespace, image: Path) -> str | None:
    if args.packaging:
        return args.packaging
    return packaging_from_path(image)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, p))


def print_report(report: dict) -> None:
    q = report["query"]
    d = report["decision"]
    kpis = report["kpis"]

    print()
    print("=" * 64)
    print("QUERY")
    print(f"  path        : {q['path']}")
    print(f"  size        : {q['width']}x{q['height']}")
    print(f"  packaging   : {q['packaging'] or '(all gallery)'}")
    print()
    print("MODEL / SEARCH")
    print(f"  embedder    : {report['embedder_key']} {report['model']}")
    print("  index       : FAISS IndexFlatIP (cosine via L2-normalized vectors)")
    print(
        f"  gallery     : {report['gallery']['num_vectors']} vectors, "
        f"{report['gallery']['num_skus']} SKUs"
    )
    print("  margin mode : inter_sku")
    print(f"  threshold   : {report['threshold']}")
    print(f"  margin      : {report['margin']}")
    print()
    print("DECISION")
    print(f"  status      : {d['status']}")
    print(f"  reason      : {d['reason']}")
    print(f"  best SKU    : {d['sku_id']}")
    print(f"  similarity  : {d['similarity']}")
    print(f"  inter-SKU D : {d['margin']}")
    if d.get("runner_up"):
        ru = d["runner_up"]
        print(f"  runner-up   : {ru['sku_id']} ({ru['similarity']})")
    print()
    print(f"TOP-{len(report['top_skus'])} SKUs")
    for i, row in enumerate(report["top_skus"], 1):
        print(
            f"  {i:>2}. {row['sku_id']:<28} "
            f"sim={row['similarity']:.4f}  {row['match_path']}"
        )
    print()
    print("KPIs")
    print(f"  best_similarity          : {kpis['best_similarity']}")
    print(f"  second_sku_similarity    : {kpis['second_sku_similarity']}")
    print(f"  inter_sku_margin         : {kpis['inter_sku_margin']}")
    print(f"  distinct_skus_in_topk    : {kpis['distinct_skus_in_hits']}")
    print(f"  passes_threshold         : {kpis['passes_threshold']}")
    print(f"  passes_margin            : {kpis['passes_margin']}")
    print(f"  recognized               : {kpis['recognized']}")
    print(f"  embed_ms                 : {kpis['embed_ms']}")
    print(f"  search_ms                : {kpis['search_ms']}")
    print(f"  total_ms                 : {kpis['total_ms']}")
    if kpis.get("hit_score_p50") is not None:
        print(
            f"  hit_score p50/p90/max    : "
            f"{kpis['hit_score_p50']:.4f} / {kpis['hit_score_p90']:.4f} / "
            f"{kpis['hit_score_max']:.4f}"
        )
    print("=" * 64)
    print()


def entries_on_disk(entries: list[dict]) -> tuple[list[dict], list[Path]]:
    """Drop gallery rows whose image file is gone, keeping vector order aligned."""
    kept: list[dict] = []
    paths: list[Path] = []
    for entry in entries:
        path = Path(entry["image_path"])
        if path.is_file():
            kept.append(entry)
            paths.append(path)
        else:
            logging.warning("Skipping missing gallery image: %s", path)
    if not paths:
        raise FileNotFoundError("No gallery images left on disk to re-embed.")
    return kept, paths


def index_for_embedder(
    embedder: DinoV3Embedder,
    saved_index,
    saved_entries: list[dict],
    meta: dict,
    batch_size: int,
):
    """Use the saved FAISS index when it matches; otherwise re-embed the gallery."""
    if gallery_matches(
        meta, embedder.model_id, embedder.canvas_width, embedder.canvas_height
    ):
        logging.info("Using saved FAISS index for %s", embedder.model_id)
        return saved_index, saved_entries, "saved_index"

    logging.info(
        "Saved index is %s; re-embedding gallery with %s",
        meta.get("model"),
        embedder.model_id,
    )
    entries, paths = entries_on_disk(saved_entries)
    vectors = embedder.embed_paths(paths, batch_size)
    return build_faiss_index(vectors), entries, "reembedded"


def run_embedder(
    args: argparse.Namespace,
    image: Path,
    device: str,
    width: int,
    height: int,
    packaging: str | None,
    saved_index,
    saved_entries: list[dict],
    meta: dict,
) -> dict:
    model_id = resolve_model_id(args.model)
    t_start = time.perf_counter()
    logging.info("Loading %s on %s", model_id, device)
    embedder = DinoV3Embedder(model_id, device)
    index, entries, index_source = index_for_embedder(
        embedder, saved_index, saved_entries, meta, args.batch_size
    )
    vectors = gallery_vectors(index)
    pack_indices = build_packaging_indices(vectors, entries)
    num_skus = len({entry["sku_id"] for entry in entries})

    t0 = time.perf_counter()
    query = embedder.embed_paths([image], batch_size=1)[0]
    embed_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    if packaging is not None:
        if packaging not in pack_indices:
            hits: list[tuple[int, float]] = []
            search_scope = packaging
        else:
            sub, global_ids = pack_indices[packaging]
            hits = search_index_hits(query, sub, global_ids, args.k_search)
            search_scope = packaging
    else:
        hits = search_index_hits(query, index, None, args.k_search)
        search_scope = "all"
    sku_ranks = rank_skus_from_hits(hits, entries)
    search_ms = (time.perf_counter() - t0) * 1000

    decision = decide_match(sku_ranks, args.threshold, args.margin)
    total_ms = (time.perf_counter() - t_start) * 1000

    hit_scores = [score for _, score in hits]
    top_skus = [
        {
            "sku_id": row["sku_id"],
            "similarity": round(row["similarity"], 4),
            "match_path": row["match_path"],
            "packaging_type": row["packaging_type"],
        }
        for row in sku_ranks[: max(args.top_k, 1)]
    ]

    best_sim = decision["similarity"]
    second_sim = decision["runner_up"]["similarity"] if decision["runner_up"] else None
    gap = decision["margin"]
    passes_threshold = best_sim is not None and best_sim >= args.threshold
    passes_margin = gap is None or gap >= args.margin

    return {
        "query": {
            "path": image.as_posix(),
            "width": width,
            "height": height,
            "packaging": packaging,
            "search_scope": search_scope,
        },
        "embedder_key": "dinov3",
        "model": embedder.model_id,
        "canvas": {
            "width": embedder.canvas_width,
            "height": embedder.canvas_height,
        },
        "index_source": index_source,
        "pretrained": None,
        "device": device,
        "gallery": {
            "dir": args.gallery.as_posix(),
            "num_vectors": index.ntotal,
            "num_skus": num_skus,
            "embedding_dim": int(query.shape[0]),
        },
        "threshold": args.threshold,
        "margin": args.margin,
        "margin_mode": "inter_sku",
        "k_search": args.k_search,
        "decision": {
            "status": decision["status"],
            "reason": decision["reason"],
            "sku_id": decision["sku_id"],
            "similarity": decision["similarity"],
            "margin": decision["margin"],
            "match_path": decision.get("match_path"),
            "runner_up": decision["runner_up"],
        },
        "top_skus": top_skus,
        "kpis": {
            "best_similarity": best_sim,
            "second_sku_similarity": second_sim,
            "inter_sku_margin": gap,
            "distinct_skus_in_hits": len(sku_ranks),
            "faiss_hits": len(hits),
            "passes_threshold": bool(passes_threshold),
            "passes_margin": bool(passes_margin),
            "recognized": decision["status"] == "recognized",
            "hit_score_p50": None if not hit_scores else round(percentile(hit_scores, 50), 4),
            "hit_score_p90": None if not hit_scores else round(percentile(hit_scores, 90), 4),
            "hit_score_max": None if not hit_scores else round(max(hit_scores), 4),
            "embed_ms": round(embed_ms, 1),
            "search_ms": round(search_ms, 1),
            "total_ms": round(total_ms, 1),
        },
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    image = args.image
    if not image.is_file():
        raise FileNotFoundError(f"Image not found: {image}")
    if image.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type: {image.suffix}")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA unavailable; falling back to cpu")
        device = "cpu"

    saved_index, saved_entries = load_gallery(args.gallery)
    meta = read_gallery_metadata(args.gallery)
    packaging = resolve_packaging(args, image)
    with Image.open(image) as im:
        width, height = im.size

    report = run_embedder(
        args,
        image,
        device,
        width,
        height,
        packaging,
        saved_index,
        saved_entries,
        meta,
    )
    print_report(report)
    payload = report

    out = args.out
    if out is None:
        out = Path("tmp") / f"sim_{image.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Wrote %s", out)


if __name__ == "__main__":
    main()
