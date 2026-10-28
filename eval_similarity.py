"""Evaluate Stage-2 similarity: best-score histograms for known vs unknown crops.

Uses the exact search + decision logic of the web pipeline
(`search_index_hits` -> `rank_skus_from_hits` -> `decide_match`) and plots how
the best similarity is distributed for:

  * known products   - crops of SKUs that ARE in the gallery
                       (split into correct top-1 vs wrong top-1)
  * unknown products - crops of products that are NOT in the gallery

If the two distributions overlap heavily, no threshold can separate them and
the embedder (or gallery) must improve. Otherwise pick the threshold in the gap.

Eval-set layout (real shelf crops, not the gallery photos themselves):
    data/eval/known/{bottle|can|carton}/{sku_id}/*.jpg   # same folder names as data/single_products
    data/eval/unknown/**/*.jpg                           # any structure

Without an eval set, --self-test runs leave-one-out on the gallery itself:
each gallery image is queried against the others (known), and against the
gallery minus its whole product (simulated unknown). Optimistic, since gallery
photos are cleaner than shelf crops, but useful for comparing settings.

Examples:
    python eval_similarity.py
    python eval_similarity.py --center
    python eval_similarity.py --embedder dinov3 --center
    python eval_similarity.py --embedder clip --model ViT-B-32 --pretrained openai --center
    python eval_similarity.py --embedder siglip --center
    python eval_similarity.py --self-test --center
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import faiss
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from embed_gallery import (
    DINOV3_MODEL,
    IMAGE_EXTENSIONS,
    PACKAGING_CLASSES,
    SIGLIP_MODEL,
    DinoV3Embedder,
    SiglipEmbedder,
    load_open_clip,
)
from pipeline import PipelineConfig
from recognize import (
    decide_match,
    embed_paths,
    load_gallery,
    rank_skus_from_hits,
    search_index_hits,
)

KNOWN_CORRECT_COLOR = "#4caf50"
KNOWN_WRONG_COLOR = "#e53935"
UNKNOWN_COLOR = "#ff9800"


def parse_args() -> argparse.Namespace:
    cfg = PipelineConfig()
    parser = argparse.ArgumentParser(
        description="Plot best-similarity histograms for known vs unknown products."
    )
    parser.add_argument("--known", type=Path, default=Path("data/eval/known"))
    parser.add_argument("--unknown", type=Path, default=Path("data/eval/unknown"))
    parser.add_argument("--gallery", type=Path, default=cfg.gallery_dir)
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/eval"))
    parser.add_argument("--model", default="ViT-B-32", help="OpenCLIP name used with --embedder clip")
    parser.add_argument("--pretrained", default="openai", help="OpenCLIP tag used with --embedder clip")
    parser.add_argument("--device", default=cfg.device)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--k-search", type=int, default=cfg.k_search)
    parser.add_argument(
        "--threshold",
        type=float,
        default=cfg.threshold,
        help="Threshold marked on the plot (default: current pipeline value)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=cfg.margin,
        help="Inter-product margin applied in the threshold sweep (default: pipeline value)",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Subtract the gallery mean vector from gallery + queries, then re-normalize",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Leave-one-out on the gallery itself instead of data/eval",
    )
    parser.add_argument(
        "--embedder",
        choices=("clip", "dinov3", "siglip"),
        default="dinov3",
        help="dinov3 matches the API gallery; clip and siglip re-embed the catalog for comparison",
    )
    parser.add_argument(
        "--dinov3-model",
        default=DINOV3_MODEL,
        help="Hugging Face id used when --embedder dinov3 (see DINOV3_MODEL in embed_gallery.py)",
    )
    parser.add_argument(
        "--siglip-model",
        default=SIGLIP_MODEL,
        help="Hugging Face id used when --embedder siglip (see SIGLIP_MODEL in embed_gallery.py)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------- #
# Data collection


def collect_known(root: Path, gallery_products: set[tuple[str, str]]) -> list[tuple[Path, tuple[str, str]]]:
    items: list[tuple[Path, tuple[str, str]]] = []
    if not root.is_dir():
        return items
    for packaging in PACKAGING_CLASSES:
        class_dir = root / packaging
        if not class_dir.is_dir():
            continue
        for sku_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            product = (sku_dir.name, packaging)
            if product not in gallery_products:
                logging.warning(
                    "Skipping %s: no gallery folder data/single_products/%s/%s",
                    sku_dir,
                    packaging,
                    sku_dir.name,
                )
                continue
            for path in sorted(sku_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    items.append((path, product))
    return items


def collect_unknown(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True).clip(min=1e-12)
    return (vectors / norms).astype(np.float32)


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable; falling back to cpu")
        return "cpu"
    return requested


def embedder_label(args: argparse.Namespace) -> str:
    if args.embedder == "dinov3":
        return f"DINOv3 {args.dinov3_model.split('/')[-1]}"
    if args.embedder == "siglip":
        return f"SigLIP {args.siglip_model.split('/')[-1]}"
    return f"OpenCLIP {args.model}/{args.pretrained}"


def embedder_tag(args: argparse.Namespace) -> str:
    if args.embedder == "dinov3":
        return args.dinov3_model.split("/")[-1]
    if args.embedder == "siglip":
        return args.siglip_model.split("/")[-1]
    return f"{args.model}_{args.pretrained}"


def selected_model_name(args: argparse.Namespace) -> str:
    if args.embedder == "dinov3":
        return args.dinov3_model
    if args.embedder == "siglip":
        return args.siglip_model
    return args.model


def gallery_paths_on_disk(entries: list[dict]) -> list[Path]:
    """Drop metadata rows whose image file is gone. Mutates entries in place."""
    paths = [Path(entry["image_path"]) for entry in entries]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        logging.warning(
            "Skipping %d gallery images that are no longer on disk (first: %s)",
            len(missing),
            missing[0],
        )
        keep = [i for i, path in enumerate(paths) if path.is_file()]
        entries[:] = [entries[i] for i in keep]
        paths = [paths[i] for i in keep]
    if not paths:
        raise FileNotFoundError("No gallery images left to embed.")
    return paths


def load_gallery_vectors(
    args: argparse.Namespace,
    index: faiss.Index,
    entries: list[dict],
    device: str,
):
    """Return (embed_fn, vectors). Every embedder re-embeds the catalog photos.

    The saved FAISS index is DINOv3. Reusing it for CLIP or SigLIP mixes
    vector sizes (for example 512 vs 768) and breaks --center.
    """
    del index  # comparison runs do not search the saved index
    gallery_paths = gallery_paths_on_disk(entries)

    if args.embedder == "clip":
        logging.info("Loading OpenCLIP %s / %s on %s", args.model, args.pretrained, device)
        model, preprocess = load_open_clip(args.model, args.pretrained, device)

        def embed_fn(paths: list[Path]) -> np.ndarray:
            return embed_paths(paths, model, preprocess, device, args.batch_size)

        label = f"OpenCLIP {args.model}/{args.pretrained}"
    elif args.embedder == "siglip":
        logging.info("Loading %s on %s", args.siglip_model, device)
        embedder = SiglipEmbedder(args.siglip_model, device)

        def embed_fn(paths: list[Path]) -> np.ndarray:
            return embedder.embed_paths(paths, args.batch_size)

        label = args.siglip_model
    else:
        logging.info("Loading %s on %s", args.dinov3_model, device)
        embedder = DinoV3Embedder(args.dinov3_model, device)

        def embed_fn(paths: list[Path]) -> np.ndarray:
            return embedder.embed_paths(paths, args.batch_size)

        label = args.dinov3_model

    logging.info("Re-embedding %d gallery images with %s", len(gallery_paths), label)
    return embed_fn, embed_fn(gallery_paths)


# ---------------------------------------------------------------------- #
# Evaluation


def make_row(
    name: str,
    kind: str,
    truth: tuple[str, str] | None,
    decision: dict,
) -> dict:
    predicted = (
        None
        if decision["sku_id"] is None
        else (decision["sku_id"], decision["packaging_type"])
    )
    return {
        "path": name,
        "kind": kind,
        "true_sku": None if truth is None else truth[0],
        "true_packaging": None if truth is None else truth[1],
        "best_sku": decision["sku_id"],
        "best_packaging": decision["packaging_type"],
        "best_path": decision.get("match_path"),
        "similarity": decision["similarity"],
        "margin": decision["margin"],
        "runner_up": decision["runner_up"],
        "status": decision["status"],
        "reason": decision["reason"],
        "top1_correct": None if truth is None else predicted == truth,
    }


def evaluate_eval_set(
    args: argparse.Namespace,
    index: faiss.Index,
    entries: list[dict],
    mean: np.ndarray | None,
    embed_fn,
) -> list[dict]:
    gallery_products = {(e["sku_id"], e["packaging_type"]) for e in entries}
    known = collect_known(args.known, gallery_products)
    unknown = collect_unknown(args.unknown)
    if not known and not unknown:
        raise FileNotFoundError(
            f"No eval images under {args.known} or {args.unknown}. "
            "Add crops there (see module docstring) or run with --self-test."
        )
    logging.info("Eval set: %d known crops, %d unknown crops", len(known), len(unknown))

    paths = [p for p, _ in known] + unknown
    truths: list[tuple[str, str] | None] = [t for _, t in known] + [None] * len(unknown)
    queries = embed_fn(paths)
    if mean is not None:
        queries = normalize(queries - mean)

    rows: list[dict] = []
    for path, truth, vector in zip(paths, truths, queries):
        hits = search_index_hits(vector, index, None, args.k_search)
        ranks = rank_skus_from_hits(hits, entries)
        decision = decide_match(ranks, args.threshold, args.margin)
        kind = "known" if truth is not None else "unknown"
        rows.append(make_row(path.as_posix(), kind, truth, decision))
    return rows


def evaluate_self_test(
    args: argparse.Namespace,
    vectors: np.ndarray,
    entries: list[dict],
) -> list[dict]:
    products = [(e["sku_id"], e["packaging_type"]) for e in entries]
    sims = vectors @ vectors.T
    product_sizes: dict[tuple[str, str], int] = {}
    for p in products:
        product_sizes[p] = product_sizes.get(p, 0) + 1

    def decide(i: int, allowed: np.ndarray) -> dict:
        ids = np.flatnonzero(allowed)
        order = ids[np.argsort(-sims[i, ids])][: args.k_search]
        hits = [(int(j), float(sims[i, j])) for j in order]
        ranks = rank_skus_from_hits(hits, entries)
        return decide_match(ranks, args.threshold, args.margin)

    rows: list[dict] = []
    all_ids = np.arange(len(entries))
    for i, entry in enumerate(entries):
        product = products[i]
        if product_sizes[product] > 1:
            allowed = all_ids != i
            rows.append(make_row(entry["image_path"], "known", product, decide(i, allowed)))
        allowed = np.array([p != product for p in products])
        rows.append(make_row(entry["image_path"], "unknown", None, decide(i, allowed)))

    singles = sum(1 for n in product_sizes.values() if n == 1)
    if singles:
        logging.info(
            "Self-test: %d products have a single gallery image and only count as unknowns",
            singles,
        )
    return rows


def accepted(row: dict, threshold: float, margin: float) -> bool:
    if row["similarity"] is None or row["similarity"] < threshold:
        return False
    return row["margin"] is None or row["margin"] >= margin


def sweep(rows: list[dict], margin: float, thresholds: np.ndarray) -> list[dict]:
    known = [r for r in rows if r["kind"] == "known"]
    unknown = [r for r in rows if r["kind"] == "unknown"]
    out: list[dict] = []
    for t in thresholds:
        t = float(t)
        k_ok = sum(1 for r in known if r["top1_correct"] and accepted(r, t, margin))
        k_wrong = sum(1 for r in known if not r["top1_correct"] and accepted(r, t, margin))
        u_acc = sum(1 for r in unknown if accepted(r, t, margin))
        out.append(
            {
                "threshold": round(t, 3),
                "known_recognized": k_ok / len(known) if known else None,
                "known_misidentified": k_wrong / len(known) if known else None,
                "known_missed": (len(known) - k_ok - k_wrong) / len(known) if known else None,
                "unknown_false_accept": u_acc / len(unknown) if unknown else None,
            }
        )
    return out


def suggest_threshold(points: list[dict]) -> dict | None:
    """Threshold maximizing known_recognized - unknown_false_accept - known_misidentified."""
    valid = [
        p
        for p in points
        if p["known_recognized"] is not None and p["unknown_false_accept"] is not None
    ]
    if not valid:
        return None
    return max(
        valid,
        key=lambda p: p["known_recognized"] - p["unknown_false_accept"] - p["known_misidentified"],
    )


# ---------------------------------------------------------------------- #
# Reporting


def plot(
    rows: list[dict],
    points: list[dict],
    current: float,
    suggested: dict | None,
    title: str,
    out_png: Path,
) -> None:
    known_ok = [r["similarity"] for r in rows if r["kind"] == "known" and r["top1_correct"]]
    known_bad = [r["similarity"] for r in rows if r["kind"] == "known" and not r["top1_correct"]]
    unknown = [r["similarity"] for r in rows if r["kind"] == "unknown"]
    scores = [s for s in known_ok + known_bad + unknown if s is not None]
    bins = np.linspace(min(scores) - 0.01, max(scores) + 0.01, 40)

    fig, (ax_hist, ax_rates) = plt.subplots(1, 2, figsize=(15, 5.5))

    for values, color, label in (
        (known_ok, KNOWN_CORRECT_COLOR, f"known, correct top-1 (n={len(known_ok)})"),
        (known_bad, KNOWN_WRONG_COLOR, f"known, wrong top-1 (n={len(known_bad)})"),
        (unknown, UNKNOWN_COLOR, f"unknown (n={len(unknown)})"),
    ):
        if values:
            ax_hist.hist(values, bins=bins, alpha=0.55, color=color, label=label, edgecolor="white")
    ax_hist.set_xlabel("best similarity (top-1 product)")
    ax_hist.set_ylabel("crops")
    ax_hist.set_title("Best-score distribution")

    ts = [p["threshold"] for p in points]
    for key, color, label in (
        ("known_recognized", KNOWN_CORRECT_COLOR, "known recognized correctly"),
        ("known_misidentified", KNOWN_WRONG_COLOR, "known accepted as wrong SKU"),
        ("known_missed", "#757575", "known rejected (missed)"),
        ("unknown_false_accept", UNKNOWN_COLOR, "unknown accepted (false match)"),
    ):
        values = [p[key] for p in points]
        if values[0] is not None:
            ax_rates.plot(ts, values, color=color, label=label, linewidth=2)
    ax_rates.set_xlabel("threshold")
    ax_rates.set_ylabel("rate")
    ax_rates.set_ylim(-0.02, 1.02)
    ax_rates.set_title("Rates vs threshold")

    for ax in (ax_hist, ax_rates):
        if bins[0] <= current <= bins[-1] or ax is ax_rates:
            ax.axvline(current, color="black", linestyle="--", label=f"current threshold {current:.2f}")
        if suggested is not None:
            ax.axvline(
                suggested["threshold"],
                color="#1e88e5",
                linestyle=":",
                linewidth=2,
                label=f"suggested {suggested['threshold']:.2f}",
            )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def print_summary(rows: list[dict], points: list[dict], args: argparse.Namespace, suggested: dict | None) -> None:
    known = [r for r in rows if r["kind"] == "known"]
    unknown = [r for r in rows if r["kind"] == "unknown"]

    def stats(values: list[float]) -> str:
        if not values:
            return "-"
        arr = np.asarray(values)
        return (
            f"n={len(arr):<4} min={arr.min():.3f}  p10={np.percentile(arr, 10):.3f}  "
            f"median={np.median(arr):.3f}  p90={np.percentile(arr, 90):.3f}  max={arr.max():.3f}"
        )

    def at(threshold: float) -> dict:
        return sweep(rows, args.margin, np.array([threshold]))[0]

    def fmt(p: dict) -> str:
        parts = []
        if p["known_recognized"] is not None:
            parts.append(
                f"known recognized {p['known_recognized']:.1%}, "
                f"misidentified {p['known_misidentified']:.1%}, "
                f"missed {p['known_missed']:.1%}"
            )
        if p["unknown_false_accept"] is not None:
            parts.append(f"unknown false-accept {p['unknown_false_accept']:.1%}")
        return " | ".join(parts)

    print()
    print("=" * 72)
    print("BEST SIMILARITY")
    print(f"  known, correct top-1 : {stats([r['similarity'] for r in known if r['top1_correct']])}")
    print(f"  known, wrong top-1   : {stats([r['similarity'] for r in known if not r['top1_correct']])}")
    print(f"  unknown              : {stats([r['similarity'] for r in unknown])}")
    if known:
        top1 = sum(1 for r in known if r["top1_correct"]) / len(known)
        print(f"  known top-1 accuracy (no threshold): {top1:.1%}")
    print()
    print(f"AT CURRENT threshold={args.threshold:.2f} margin={args.margin:.2f}")
    print(f"  {fmt(at(args.threshold))}")
    if suggested is not None:
        print(f"SUGGESTED threshold={suggested['threshold']:.2f} (same margin)")
        print(f"  {fmt(suggested)}")
    print("=" * 72)
    print()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    saved_index, entries = load_gallery(args.gallery)
    device = resolve_device(args.device)
    embed_fn, vectors = load_gallery_vectors(args, saved_index, entries, device)
    mean: np.ndarray | None = None
    if args.center:
        mean = vectors.mean(axis=0, keepdims=True)
        vectors = normalize(vectors - mean)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    if args.self_test:
        rows = evaluate_self_test(args, vectors, entries)
    else:
        rows = evaluate_eval_set(args, index, entries, mean, embed_fn)

    points = sweep(rows, args.margin, np.arange(0.0, 1.0001, 0.01))
    suggested = suggest_threshold(points)

    mode = "selftest" if args.self_test else "evalset"
    tag = f"{mode}_{embedder_tag(args)}{'_centered' if args.center else ''}"
    out_png = args.out_dir / f"similarity_{tag}.png"
    out_json = args.out_dir / f"similarity_{tag}.json"
    title = (
        f"{'Gallery leave-one-out' if args.self_test else 'Eval set'} | "
        f"{embedder_label(args)}"
        f"{' | centered' if args.center else ''} | margin {args.margin:.2f}"
    )
    plot(rows, points, args.threshold, suggested, title, out_png)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "config": {
                    "mode": mode,
                    "embedder": args.embedder,
                    "model": selected_model_name(args),
                    "pretrained": args.pretrained if args.embedder == "clip" else None,
                    "centered": args.center,
                    "threshold": args.threshold,
                    "margin": args.margin,
                    "k_search": args.k_search,
                },
                "suggested": suggested,
                "sweep": points,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print_summary(rows, points, args, suggested)
    logging.info("Wrote %s", out_png)
    logging.info("Wrote %s", out_json)


if __name__ == "__main__":
    main()
