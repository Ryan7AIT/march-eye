"""Compare same-SKU vs different-SKU cosine similarities in the gallery.

Uses the saved FAISS index (built from data/single_products). Vectors are
L2-normalized, so V @ V.T is cosine similarity. Identity is
(packaging_type, sku_id): the same folder name under bottle/ and carton/ is
two products.

Example:
    python test_gallery_pairs.py
    python test_gallery_pairs.py --center
    python test_gallery_pairs.py --gallery gallery --out tmp/gallery_pairs.json
    python test_gallery_pairs.py --packaging bottle --no-plot
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recognize import gallery_vectors, load_gallery, read_gallery_metadata


INTRA_PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
INTER_PERCENTILES = (1, 5, 50, 90, 95, 99)
CONFUSING_PAIRS = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic: intra-SKU vs inter-SKU cosine similarity in the gallery "
            "(images from data/single_products)."
        )
    )
    parser.add_argument("--gallery", type=Path, default=Path("gallery"))
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/single_products"),
        help="Expected catalog root; used only to report missing files",
    )
    parser.add_argument(
        "--packaging",
        choices=("bottle", "can", "carton"),
        default=None,
        help="Restrict the pairwise matrix to one packaging folder",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help=(
            "Subtract the gallery mean vector, then re-normalize, before pairwise "
            "cosine. Production search does not do this; use it only as an A/B."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.28,
        help="Pipeline match threshold, marked on the plot and in the report",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tmp/gallery_pairs.json"),
        help="JSON report path",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("tmp/gallery_pairs.png"),
        help="Histogram path (ignored with --no-plot)",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--top-confusions",
        type=int,
        default=CONFUSING_PAIRS,
        help="How many highest different-SKU pairs to print",
    )
    return parser.parse_args()


def product_key(entry: dict) -> str:
    return f"{entry.get('packaging_type', '')}/{entry['sku_id']}"


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True).clip(min=1e-12)
    return (vectors / norms).astype(np.float32)


def apply_default_centered_paths(args: argparse.Namespace) -> None:
    """Keep uncentered plots when --center uses the default out/plot paths."""
    if not args.center:
        return
    if args.out == Path("tmp/gallery_pairs.json"):
        args.out = Path("tmp/gallery_pairs_centered.json")
    if args.plot == Path("tmp/gallery_pairs.png"):
        args.plot = Path("tmp/gallery_pairs_centered.png")


def percentile_dict(values: np.ndarray, ps: tuple[int, ...]) -> dict[str, float] | None:
    if values.size == 0:
        return None
    raw = np.percentile(values, list(ps))
    return {f"p{p}": round(float(v), 4) for p, v in zip(ps, raw)}


def fmt_pct(stats: dict[str, float] | None, keys: tuple[int, ...]) -> str:
    if not stats:
        return "(none — need at least two views of the same product)"
    return "  ".join(f"p{p}={stats[f'p{p}']:.4f}" for p in keys)


def select_entries(
    entries: list[dict], packaging: str | None
) -> tuple[list[dict], np.ndarray]:
    if packaging is None:
        return entries, np.arange(len(entries))
    ids = [i for i, e in enumerate(entries) if e.get("packaging_type") == packaging]
    if not ids:
        raise ValueError(f"No gallery entries with packaging_type={packaging!r}")
    return [entries[i] for i in ids], np.asarray(ids, dtype=np.int64)


def pair_stats(
    vectors: np.ndarray, entries: list[dict]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (sims, same_mask, i_idx, j_idx, keys) for unique upper-triangle pairs."""
    keys = np.array([product_key(e) for e in entries])
    sims = vectors @ vectors.T
    i_idx, j_idx = np.triu_indices(len(entries), k=1)
    pair_sims = sims[i_idx, j_idx]
    same = keys[i_idx] == keys[j_idx]
    return pair_sims, same, i_idx, j_idx, keys


def per_product_intra(
    pair_sims: np.ndarray,
    same: np.ndarray,
    i_idx: np.ndarray,
    keys: np.ndarray,
) -> list[dict]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for sim, is_same, i in zip(pair_sims, same, i_idx):
        if is_same:
            buckets[str(keys[i])].append(float(sim))

    counts = Counter(str(k) for k in keys)
    rows: list[dict] = []
    for key, n in sorted(counts.items()):
        vals = np.asarray(buckets.get(key, []), dtype=np.float64)
        rows.append(
            {
                "product": key,
                "n_images": n,
                "n_pairs": int(vals.size),
                "intra_min": None if vals.size == 0 else round(float(vals.min()), 4),
                "intra_p50": None if vals.size == 0 else round(float(np.median(vals)), 4),
                "intra_mean": None if vals.size == 0 else round(float(vals.mean()), 4),
            }
        )
    return rows


def confusing_pairs(
    pair_sims: np.ndarray,
    same: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    entries: list[dict],
    top_n: int,
) -> list[dict]:
    if top_n <= 0:
        return []
    inter_ids = np.flatnonzero(~same)
    if inter_ids.size == 0:
        return []
    order = inter_ids[np.argsort(-pair_sims[inter_ids])][:top_n]
    rows: list[dict] = []
    for k in order:
        a = entries[int(i_idx[k])]
        b = entries[int(j_idx[k])]
        rows.append(
            {
                "similarity": round(float(pair_sims[k]), 4),
                "a": product_key(a),
                "b": product_key(b),
                "a_path": a["image_path"],
                "b_path": b["image_path"],
            }
        )
    return rows


def weakest_intra(product_rows: list[dict], limit: int = 10) -> list[dict]:
    scored = [r for r in product_rows if r["intra_p50"] is not None]
    scored.sort(key=lambda r: r["intra_p50"])
    return scored[:limit]


def interpret(
    intra: np.ndarray,
    inter: np.ndarray,
    intra_stats: dict[str, float] | None,
    inter_stats: dict[str, float] | None,
    threshold: float,
) -> list[str]:
    notes: list[str] = []
    if intra.size == 0:
        notes.append(
            "No same-SKU pairs: every product has a single gallery image. "
            "Add a second view per SKU under data/single_products/{packaging}/{sku}/."
        )
        return notes
    if inter.size == 0:
        notes.append("No different-SKU pairs (only one product in this slice).")
        return notes

    intra_p5 = intra_stats["p5"]
    intra_p50 = intra_stats["p50"]
    inter_p95 = inter_stats["p95"]
    inter_p99 = inter_stats["p99"]
    gap_p5_p95 = intra_p5 - inter_p95
    notes.append(
        f"intra p5 ({intra_p5:.4f}) vs inter p95 ({inter_p95:.4f}): "
        f"gap={gap_p5_p95:+.4f}"
    )
    if gap_p5_p95 >= 0.05:
        notes.append(
            "Same-SKU views sit clearly above almost all different-SKU pairs. "
            "A threshold in that gap is usable on gallery photos."
        )
    elif gap_p5_p95 >= 0:
        notes.append(
            "The intra p5 / inter p95 gap is razor-thin. Same-SKU and lookalike "
            "pairs occupy the same score band; a single threshold cannot separate them."
        )
    else:
        notes.append(
            "Distributions overlap: some different SKUs look more alike than some "
            "views of the same SKU. Threshold/margin alone cannot fix those pairs."
        )

    notes.append(
        f"intra p50 ({intra_p50:.4f}) vs inter p99 ({inter_p99:.4f}): "
        f"gap={intra_p50 - inter_p99:+.4f}"
    )
    intra_pass = float((intra >= threshold).mean())
    inter_pass = float((inter >= threshold).mean())
    notes.append(
        f"Pairs >= threshold {threshold:.2f}: "
        f"intra {intra_pass:.1%}  |  inter {inter_pass:.1%}"
    )
    if inter_pass > 0.25:
        notes.append(
            f"Threshold {threshold:.2f} sits under most different-SKU pairs too. "
            "It will not reject gallery lookalikes; raise it or rely on inter-SKU margin."
        )
    notes.append(
        "Gallery photos are cleaner than shelf crops; treat these numbers as an "
        "upper bound on how well Stage 2 can separate products."
    )
    return notes


def plot_hist(
    intra: np.ndarray,
    inter: np.ndarray,
    threshold: float,
    title: str,
    out_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    scores = np.concatenate(
        [a for a in (intra, inter) if a.size]
    )
    lo = float(min(scores.min() - 0.02, threshold - 0.02, -0.05))
    hi = float(max(scores.max() + 0.02, threshold + 0.02, 1.0))
    bins = np.linspace(lo, hi, 50)

    if intra.size:
        ax.hist(
            intra,
            bins=bins,
            density=True,
            alpha=0.55,
            color="#4caf50",
            label=f"same SKU (n={intra.size})",
            edgecolor="white",
        )
    if inter.size:
        ax.hist(
            inter,
            bins=bins,
            density=True,
            alpha=0.45,
            color="#e53935",
            label=f"different SKU (n={inter.size})",
            edgecolor="white",
        )

    ax.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=f"pipeline threshold {threshold:.2f}",
    )
    if intra.size:
        ax.axvline(
            float(np.percentile(intra, 5)),
            color="#2e7d32",
            linestyle=":",
            linewidth=1.4,
            label=f"intra p5 {np.percentile(intra, 5):.3f}",
        )
    if inter.size:
        ax.axvline(
            float(np.percentile(inter, 95)),
            color="#b71c1c",
            linestyle=":",
            linewidth=1.4,
            label=f"inter p95 {np.percentile(inter, 95):.3f}",
        )

    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def print_report(payload: dict) -> None:
    g = payload["gallery"]
    print()
    print("=" * 72)
    print("GALLERY PAIR DIAGNOSTIC")
    print(f"  index       : {g['dir']}  ({g['model']})")
    print(f"  canvas      : {g['canvas']['width']}x{g['canvas']['height']}")
    print(f"  images      : {g['n_images']}  (on disk: {g['n_on_disk']})")
    print(f"  products    : {g['n_products']}  (sku_id + packaging)")
    print(f"  singletons  : {g['n_singletons']} products with 1 image (no intra pair)")
    print(f"  packaging   : {payload['packaging'] or 'all'}")
    print(f"  centered    : {payload['centered']}")
    print(f"  source      : {payload['source']}")
    print()
    print("PAIR COUNTS (unique unordered pairs)")
    print(f"  same SKU        : {payload['n_intra']}")
    print(f"  different SKU   : {payload['n_inter']}")
    print()
    print("INTRA  (same SKU, different views)")
    print(f"  {fmt_pct(payload['intra'], (1, 5, 50))}")
    if payload["intra"]:
        print(f"  {fmt_pct(payload['intra'], (10, 25, 75, 90, 95, 99))}")
        print(
            f"  min={payload['intra_ext']['min']:.4f}  "
            f"max={payload['intra_ext']['max']:.4f}  "
            f"mean={payload['intra_ext']['mean']:.4f}"
        )
    print()
    print("INTER  (different SKUs)")
    print(f"  {fmt_pct(payload['inter'], (50, 95, 99))}")
    if payload["inter"]:
        print(f"  {fmt_pct(payload['inter'], (1, 5, 90))}")
        print(
            f"  min={payload['inter_ext']['min']:.4f}  "
            f"max={payload['inter_ext']['max']:.4f}  "
            f"mean={payload['inter_ext']['mean']:.4f}"
        )
    print()
    print("READ THIS")
    for line in payload["notes"]:
        print(f"  - {line}")
    if payload["weakest_intra"]:
        print()
        print("WEAKEST SAME-SKU PRODUCTS (lowest intra median)")
        for row in payload["weakest_intra"]:
            print(
                f"  {row['intra_p50']:.4f}  {row['product']:<40} "
                f"n={row['n_images']} pairs={row['n_pairs']}"
            )
    if payload["confusions"]:
        print()
        print(f"MOST SIMILAR DIFFERENT-SKU PAIRS (top {len(payload['confusions'])})")
        for row in payload["confusions"]:
            print(
                f"  {row['similarity']:.4f}  {row['a']}  <->  {row['b']}"
            )
            print(f"           {row['a_path']}")
            print(f"           {row['b_path']}")
    print("=" * 72, flush=True)
    print(flush=True)


def main() -> None:
    args = parse_args()
    apply_default_centered_paths(args)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    index, entries = load_gallery(args.gallery)
    meta = read_gallery_metadata(args.gallery)
    vectors = gallery_vectors(index)
    if args.center:
        mean = vectors.mean(axis=0, keepdims=True)
        vectors = normalize(vectors - mean)
        logging.info("Centered: subtracted full-gallery mean, then L2-normalized")
    entries, keep = select_entries(entries, args.packaging)
    vectors = vectors[keep]

    n_on_disk = sum(1 for e in entries if Path(e["image_path"]).is_file())
    missing = [e["image_path"] for e in entries if not Path(e["image_path"]).is_file()]
    for path in missing[:8]:
        logging.warning("Gallery image missing on disk: %s", path)
    if len(missing) > 8:
        logging.warning("... and %d more missing files", len(missing) - 8)

    pair_sims, same, i_idx, j_idx, keys = pair_stats(vectors, entries)
    intra = pair_sims[same]
    inter = pair_sims[~same]
    intra_stats = percentile_dict(intra, INTRA_PERCENTILES)
    inter_stats = percentile_dict(inter, INTER_PERCENTILES)
    product_rows = per_product_intra(pair_sims, same, i_idx, keys)
    counts = Counter(str(k) for k in keys)

    intra_ext = None
    if intra.size:
        intra_ext = {
            "min": round(float(intra.min()), 4),
            "max": round(float(intra.max()), 4),
            "mean": round(float(intra.mean()), 4),
        }
    inter_ext = None
    if inter.size:
        inter_ext = {
            "min": round(float(inter.min()), 4),
            "max": round(float(inter.max()), 4),
            "mean": round(float(inter.mean()), 4),
        }

    canvas = meta.get("image_canvas") or {}
    notes = interpret(intra, inter, intra_stats, inter_stats, args.threshold)
    if args.center:
        notes.append(
            "This run used gallery-mean centering; production search does not. "
            "Compare to a run without --center before changing the pipeline."
        )
    payload = {
        "gallery": {
            "dir": args.gallery.as_posix(),
            "model": meta.get("model"),
            "embedder": meta.get("embedder"),
            "canvas": {
                "width": canvas.get("width"),
                "height": canvas.get("height"),
            },
            "n_images": len(entries),
            "n_on_disk": n_on_disk,
            "n_products": len(counts),
            "n_singletons": sum(1 for n in counts.values() if n == 1),
        },
        "source": args.source.as_posix(),
        "packaging": args.packaging,
        "centered": bool(args.center),
        "threshold": args.threshold,
        "n_intra": int(intra.size),
        "n_inter": int(inter.size),
        "intra": intra_stats,
        "inter": inter_stats,
        "intra_ext": intra_ext,
        "inter_ext": inter_ext,
        "notes": notes,
        "weakest_intra": weakest_intra(product_rows),
        "confusions": confusing_pairs(
            pair_sims, same, i_idx, j_idx, entries, args.top_confusions
        ),
        "products": product_rows,
    }

    # Same headline numbers as the README snippet (unique pairs; percentiles match).
    if intra.size:
        p1, p5, p50 = np.percentile(intra, [1, 5, 50])
        print(f"intra p1/p5/p50: [{p1:.6f} {p5:.6f} {p50:.6f}]", flush=True)
    else:
        print("intra p1/p5/p50: (no same-SKU pairs)", flush=True)
    if inter.size:
        p50, p95, p99 = np.percentile(inter, [50, 95, 99])
        print(f"inter p50/p95/p99: [{p50:.6f} {p95:.6f} {p99:.6f}]", flush=True)
    else:
        print("inter p50/p95/p99: (no different-SKU pairs)", flush=True)

    print_report(payload)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Wrote %s", args.out)

    if not args.no_plot and (intra.size or inter.size):
        title = (
            f"Gallery pair similarities"
            f"{'' if args.packaging is None else ' | ' + args.packaging}"
            f" | {meta.get('embedder', 'embedder')}"
            f"{' | centered' if args.center else ''}"
        )
        plot_hist(intra, inter, args.threshold, title, args.plot)
        logging.info("Wrote %s", args.plot)


if __name__ == "__main__":
    main()
