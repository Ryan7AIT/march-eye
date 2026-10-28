"""Data dashboard for the SKU catalog: gallery photos, eval crops, similarity.

Reads data/single_products, data/eval and the gallery/ FAISS index. No models
and no GPU are loaded, so it is safe to run next to training.

Standalone:
    python -m uvicorn dashboard:app --host 127.0.0.1 --port 8001
    -> http://127.0.0.1:8001/

Inside the main app (see app.py):
    http://127.0.0.1:8000/dashboard.html

Endpoints (all under /api/dashboard):
    GET /overview     counts, KPIs, data-quality findings, readiness checks
    GET /similarity   most similar SKU pairs from the gallery index
    GET /media/...    image files under data/
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from catalog import router as catalog_router

PACKAGING = ("bottle", "can", "carton")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DATA_ROOT = Path("data")
GALLERY_ROOT = DATA_ROOT / "single_products"
EVAL_ROOT = DATA_ROOT / "eval"
GALLERY_DIR = Path("gallery")
WEB_DIR = Path("web")

TINY_PX = 64  # short side below this is flagged
MIN_GOOD_IMAGES = 3  # per-SKU minimum recommended for the gallery
HIST_EDGES = np.round(np.arange(-0.2, 1.0001, 0.05), 2)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


# ---------------------------------------------------------------------- #
# Disk scanning


_file_cache: dict[tuple[str, int, int], dict] = {}


def file_info(path: Path) -> dict:
    """md5 + pixel size, cached on (path, mtime, size)."""
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    hit = _file_cache.get(key)
    if hit is not None:
        return hit
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    try:
        with Image.open(path) as im:
            width, height = im.size
    except Exception:
        width = height = 0
    info = {"md5": h.hexdigest(), "w": width, "h": height, "bytes": st.st_size}
    _file_cache[key] = info
    return info


def media_url(path: Path | str) -> str | None:
    try:
        rel = Path(path).relative_to(DATA_ROOT).as_posix()
    except ValueError:
        return None
    return "/api/dashboard/media/" + quote(rel)


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def child_dir(parent: Path, name: str) -> Path | None:
    """Case-insensitive subfolder lookup (data/eval/Known vs known on Windows)."""
    if not parent.is_dir():
        return None
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == name:
            return child
    return None


def scan_sku_tree(root: Path) -> tuple[dict[tuple[str, str], list[Path]], list[tuple[str, str]]]:
    """{(sku_id, packaging): [images]} and the SKU folders that hold no images."""
    products: dict[tuple[str, str], list[Path]] = {}
    empty: list[tuple[str, str]] = []
    for packaging in PACKAGING:
        class_dir = root / packaging
        if not class_dir.is_dir():
            continue
        for sku_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            images = list_images(sku_dir)
            if images:
                products[(sku_dir.name, packaging)] = images
            else:
                empty.append((sku_dir.name, packaging))
    return products, empty


def scan_eval() -> tuple[dict[tuple[str, str], list[Path]], list[Path]]:
    known_dir = child_dir(EVAL_ROOT, "known")
    unknown_dir = child_dir(EVAL_ROOT, "unknown")
    known: dict[tuple[str, str], list[Path]] = {}
    if known_dir is not None:
        known, _ = scan_sku_tree(known_dir)
    unknown: list[Path] = []
    if unknown_dir is not None:
        unknown = sorted(
            p for p in unknown_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    return known, unknown


def read_meta() -> dict | None:
    meta_path = GALLERY_DIR / "metadata.json"
    if not meta_path.is_file() or not (GALLERY_DIR / "faiss.index").is_file():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def stats(values: list[float]) -> dict:
    if not values:
        return {"min": 0, "p25": 0, "median": 0, "mean": 0, "p75": 0, "max": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": round(float(arr.mean()), 2),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


# ---------------------------------------------------------------------- #
# Overview


def find_duplicates(
    gallery: dict[tuple[str, str], list[Path]],
    eval_known: dict[tuple[str, str], list[Path]],
    eval_unknown: list[Path],
) -> tuple[list[dict], dict[str, int], dict[str, set]]:
    """Group byte-identical files and classify what kind of problem each is."""
    by_md5: dict[str, list[tuple[str, tuple[str, str] | None, Path]]] = defaultdict(list)
    for key, paths in gallery.items():
        for p in paths:
            by_md5[file_info(p)["md5"]].append(("gallery", key, p))
    for key, paths in eval_known.items():
        for p in paths:
            by_md5[file_info(p)["md5"]].append(("eval_known", key, p))
    for p in eval_unknown:
        by_md5[file_info(p)["md5"]].append(("eval_unknown", None, p))

    counts = {"cross_sku": 0, "label_conflict": 0, "eval_overlap": 0, "same_sku": 0, "eval_dup": 0}
    flagged: dict[str, set] = {"dup": set(), "leak": set()}
    groups: list[dict] = []
    for md5, members in by_md5.items():
        if len(members) < 2:
            continue
        g_members = [m for m in members if m[0] == "gallery"]
        e_members = [m for m in members if m[0] != "gallery"]
        g_keys = {m[1] for m in g_members}
        if len(g_keys) > 1:
            kind = "cross_sku"
        elif g_members and any(m[1] is not None and m[1] not in g_keys for m in e_members):
            kind = "label_conflict"
        elif g_members and e_members:
            kind = "eval_overlap"
        elif len(g_members) > 1:
            kind = "same_sku"
        else:
            kind = "eval_dup"
        counts[kind] += 1
        keys = {m[1] for m in members if m[1] is not None}
        if kind in ("cross_sku", "label_conflict"):
            flagged["dup"] |= keys
        elif kind == "eval_overlap":
            flagged["leak"] |= keys
        groups.append(
            {
                "kind": kind,
                "md5": md5[:10],
                "files": [
                    {
                        "where": where,
                        "sku_id": key[0] if key else None,
                        "packaging_type": key[1] if key else None,
                        "name": p.name,
                        "url": media_url(p),
                    }
                    for where, key, p in members
                ],
            }
        )
    order = {"cross_sku": 0, "label_conflict": 1, "eval_overlap": 2, "same_sku": 3, "eval_dup": 4}
    groups.sort(key=lambda g: order[g["kind"]])
    return groups, counts, flagged


def index_health(gallery: dict[tuple[str, str], list[Path]]) -> dict:
    meta = read_meta()
    if meta is None:
        return {"exists": False}
    entry_paths = {e["image_path"] for e in meta["entries"]}
    disk_paths = {p.as_posix() for paths in gallery.values() for p in paths}
    new_on_disk = disk_paths - entry_paths
    gone = entry_paths - disk_paths
    built = (GALLERY_DIR / "faiss.index").stat().st_mtime
    return {
        "exists": True,
        "model": meta.get("model"),
        "embedder": meta.get("embedder"),
        "num_vectors": meta.get("num_vectors"),
        "embedding_dim": meta.get("embedding_dim"),
        "canvas": meta.get("image_canvas"),
        "built_at": datetime.fromtimestamp(built).isoformat(timespec="minutes"),
        "new_on_disk": len(new_on_disk),
        "missing_on_disk": len(gone),
        "stale": bool(new_on_disk or gone),
    }


def build_checks(k: dict, dup_counts: dict, index: dict, orphan_eval: list) -> list[dict]:
    def check(label: str, status: str, detail: str) -> dict:
        return {"label": label, "status": status, "detail": detail}

    out = []
    n_cross = dup_counts["cross_sku"] + dup_counts["label_conflict"]
    out.append(
        check(
            "Same photo filed under two SKUs",
            "bad" if n_cross else "ok",
            f"{n_cross} duplicate groups. Teaches the model that two products are the same."
            if n_cross
            else "None found.",
        )
    )
    out.append(
        check(
            "Eval crops that are copies of gallery photos",
            "bad" if dup_counts["eval_overlap"] else "ok",
            f"{dup_counts['eval_overlap']} identical files. They inflate eval accuracy."
            if dup_counts["eval_overlap"]
            else "None found.",
        )
    )
    out.append(
        check(
            "Gallery index is up to date",
            "ok" if index.get("exists") and not index.get("stale") else "bad",
            "No gallery index. Run embed_gallery.py."
            if not index.get("exists")
            else (
                f"{index['new_on_disk']} new and {index['missing_on_disk']} removed images since it was built. Re-run embed_gallery.py."
                if index["stale"]
                else f"Built {index['built_at']} with {index.get('model')}."
            ),
        )
    )
    out.append(
        check(
            f"Every SKU has at least {MIN_GOOD_IMAGES} gallery images",
            "ok" if k["below_min"] == 0 else ("bad" if k["singletons"] else "warn"),
            f"{k['below_min']} SKUs below {MIN_GOOD_IMAGES} ({k['singletons']} with a single image)."
            if k["below_min"]
            else "All SKUs meet the minimum.",
        )
    )
    cov = k["eval_coverage"]
    out.append(
        check(
            "Eval covers the catalog",
            "ok" if cov >= 0.8 else ("warn" if cov >= 0.5 else "bad"),
            f"{k['skus_with_eval']} of {k['n_skus']} SKUs have eval crops ({cov:.0%}).",
        )
    )
    out.append(
        check(
            "No empty SKU folders",
            "warn" if k["n_empty_folders"] else "ok",
            f"{k['n_empty_folders']} folders contain no images (ignored by embed_gallery)."
            if k["n_empty_folders"]
            else "None found.",
        )
    )
    out.append(
        check(
            "Eval SKUs exist in the gallery",
            "warn" if orphan_eval else "ok",
            f"{len(orphan_eval)} eval SKU folders have no gallery folder (skipped by eval)."
            if orphan_eval
            else "All eval SKU folders match a gallery folder.",
        )
    )
    out.append(
        check(
            "No tiny images",
            "warn" if k["n_tiny"] else "ok",
            f"{k['n_tiny']} gallery images have a short side under {TINY_PX}px."
            if k["n_tiny"]
            else "None found.",
        )
    )
    return out


def build_overview() -> dict:
    gallery, empty = scan_sku_tree(GALLERY_ROOT)
    eval_all, eval_unknown = scan_eval()
    gallery_keys = set(gallery)
    eval_known = {k: v for k, v in eval_all.items() if k in gallery_keys}
    orphan_eval = [
        {"sku_id": k[0], "packaging_type": k[1], "n_images": len(v)}
        for k, v in sorted(eval_all.items())
        if k not in gallery_keys
    ]

    dup_groups, dup_counts, flagged = find_duplicates(gallery, eval_all, eval_unknown)

    rows: list[dict] = []
    tiny_items: list[dict] = []
    n_tiny = 0
    for key in sorted(gallery):
        sku, pack = key
        paths = gallery[key]
        n_tiny_here = 0
        for p in paths:
            info = file_info(p)
            if info["w"] and min(info["w"], info["h"]) < TINY_PX:
                n_tiny_here += 1
                if len(tiny_items) < 40:
                    tiny_items.append(
                        {
                            "sku_id": sku,
                            "packaging_type": pack,
                            "name": p.name,
                            "w": info["w"],
                            "h": info["h"],
                            "url": media_url(p),
                        }
                    )
        n_tiny += n_tiny_here
        eval_paths = eval_known.get(key, [])
        flags: list[str] = []
        if len(paths) == 1:
            flags.append("single")
        elif len(paths) < MIN_GOOD_IMAGES:
            flags.append("few")
        if not eval_paths:
            flags.append("no_eval")
        if key in flagged["dup"]:
            flags.append("dup")
        if key in flagged["leak"]:
            flags.append("leak")
        if n_tiny_here:
            flags.append("tiny")
        rows.append(
            {
                "sku_id": sku,
                "packaging_type": pack,
                "n_images": len(paths),
                "n_eval": len(eval_paths),
                "n_tiny": n_tiny_here,
                "flags": flags,
                "images": [{"name": p.name, "url": media_url(p)} for p in paths],
                "eval_images": [{"name": p.name, "url": media_url(p)} for p in eval_paths],
            }
        )
    for sku, pack in empty:
        rows.append(
            {
                "sku_id": sku,
                "packaging_type": pack,
                "n_images": 0,
                "n_eval": len(eval_known.get((sku, pack), [])),
                "n_tiny": 0,
                "flags": ["empty"],
                "images": [],
                "eval_images": [],
            }
        )
    rows.sort(key=lambda r: (r["packaging_type"], r["sku_id"]))

    counts = [r["n_images"] for r in rows if r["n_images"] > 0]
    by_pack = {
        pack: {
            "skus": sum(1 for r in rows if r["packaging_type"] == pack and r["n_images"] > 0),
            "images": sum(r["n_images"] for r in rows if r["packaging_type"] == pack),
            "eval_crops": sum(r["n_eval"] for r in rows if r["packaging_type"] == pack),
        }
        for pack in PACKAGING
    }
    buckets = [("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5-9", 5, 9), ("10-19", 10, 19), ("20+", 20, 10**9)]
    distribution = [
        {"label": label, "skus": sum(1 for c in counts if lo <= c <= hi)} for label, lo, hi in buckets
    ]
    median = float(np.median(counts)) if counts else 0.0
    skus_with_eval = sum(1 for r in rows if r["n_images"] > 0 and r["n_eval"] > 0)
    n_skus = len(counts)
    kpis = {
        "n_skus": n_skus,
        "n_empty_folders": len(empty),
        "n_images": sum(counts),
        "images_per_sku": stats(counts),
        "singletons": sum(1 for c in counts if c == 1),
        "below_min": sum(1 for c in counts if c < MIN_GOOD_IMAGES),
        "at_least_10": sum(1 for c in counts if c >= 10),
        "imbalance_ratio": round(max(counts) / median, 1) if counts and median else None,
        "n_eval_known": sum(len(v) for v in eval_known.values()),
        "skus_with_eval": skus_with_eval,
        "eval_coverage": (skus_with_eval / n_skus) if n_skus else 0.0,
        "n_eval_unknown": len(eval_unknown),
        "n_tiny": n_tiny,
        "by_packaging": by_pack,
    }
    index = index_health(gallery)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "kpis": kpis,
        "distribution": distribution,
        "checks": build_checks(kpis, dup_counts, index, orphan_eval),
        "index": index,
        "skus": rows,
        "duplicates": {"counts": dup_counts, "groups": dup_groups[:60]},
        "orphan_eval": orphan_eval,
        "tiny": tiny_items,
        "min_good_images": MIN_GOOD_IMAGES,
    }


# ---------------------------------------------------------------------- #
# Similarity (from the gallery index; same vectors the API searches)


_sim_cache: dict = {"stamp": None, "core": None}


def similarity_core() -> dict:
    import faiss

    index_path = GALLERY_DIR / "faiss.index"
    meta_path = GALLERY_DIR / "metadata.json"
    if not (index_path.is_file() and meta_path.is_file()):
        raise HTTPException(404, "No gallery index. Run: python embed_gallery.py")
    stamp = (index_path.stat().st_mtime_ns, meta_path.stat().st_mtime_ns)
    if _sim_cache["stamp"] == stamp:
        return _sim_cache["core"]

    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    entries = meta["entries"]
    vectors = index.reconstruct_n(0, index.ntotal).astype(np.float32)
    sims = vectors @ vectors.T

    keys = [(e["sku_id"], e["packaging_type"]) for e in entries]
    products = sorted(set(keys))
    pid = {k: i for i, k in enumerate(products)}
    labels = np.array([pid[k] for k in keys])
    n_products = len(products)

    order = np.argsort(labels, kind="stable")
    sorted_sims = sims[np.ix_(order, order)]
    sorted_labels = labels[order]
    starts = np.flatnonzero(np.r_[True, sorted_labels[1:] != sorted_labels[:-1]])
    best = np.maximum.reduceat(np.maximum.reduceat(sorted_sims, starts, axis=0), starts, axis=1)
    np.fill_diagonal(best, -np.inf)

    members = [np.flatnonzero(labels == p) for p in range(n_products)]
    intra_all: list[np.ndarray] = []
    per_product: list[dict] = []
    for p in range(n_products):
        m = members[p]
        intra_mean = intra_min = None
        if len(m) >= 2:
            sub = sims[np.ix_(m, m)]
            vals = sub[np.triu_indices(len(m), 1)]
            intra_all.append(vals)
            intra_mean, intra_min = float(vals.mean()), float(vals.min())
        nearest = int(np.argmax(best[p])) if n_products > 1 else None
        nearest_sim = float(best[p, nearest]) if nearest is not None else None
        per_product.append(
            {
                "sku_id": products[p][0],
                "packaging_type": products[p][1],
                "n_images": len(m),
                "nearest_sku": products[nearest][0] if nearest is not None else None,
                "nearest_packaging": products[nearest][1] if nearest is not None else None,
                "nearest_sim": None if nearest_sim is None else round(nearest_sim, 4),
                "intra_mean": None if intra_mean is None else round(intra_mean, 4),
                "intra_min": None if intra_min is None else round(intra_min, 4),
                "separation": (
                    None
                    if intra_mean is None or nearest_sim is None
                    else round(intra_mean - nearest_sim, 4)
                ),
            }
        )

    intra = np.concatenate(intra_all) if intra_all else np.zeros(0, dtype=np.float32)
    different = labels[:, None] != labels[None, :]
    upper = np.triu(np.ones_like(different, dtype=bool), 1)
    inter = sims[different & upper]

    def hist(values: np.ndarray) -> list[float]:
        if values.size == 0:
            return [0.0] * (len(HIST_EDGES) - 1)
        counts, _ = np.histogram(values, bins=HIST_EDGES)
        return [round(float(c) / values.size, 5) for c in counts]

    def pct(values: np.ndarray, q: float) -> float | None:
        return None if values.size == 0 else round(float(np.percentile(values, q)), 4)

    summary = {
        "n_intra_pairs": int(intra.size),
        "n_inter_pairs": int(inter.size),
        "intra_p5": pct(intra, 5),
        "intra_p50": pct(intra, 50),
        "inter_p95": pct(inter, 95),
        "inter_p99": pct(inter, 99),
        "inter_max": None if inter.size == 0 else round(float(inter.max()), 4),
    }
    if summary["intra_p5"] is not None and summary["inter_p95"] is not None:
        summary["gap_p5_p95"] = round(summary["intra_p5"] - summary["inter_p95"], 4)
    else:
        summary["gap_p5_p95"] = None

    core = {
        "products": products,
        "entries": entries,
        "sims": sims,
        "best": best,
        "members": members,
        "per_product": per_product,
        "hist": {"edges": HIST_EDGES.tolist(), "intra": hist(intra), "inter": hist(inter)},
        "summary": summary,
        "model": meta.get("model"),
        "num_vectors": meta.get("num_vectors"),
    }
    _sim_cache["stamp"] = stamp
    _sim_cache["core"] = core
    return core


def build_similarity(top: int, same_packaging: bool, min_sim: float) -> dict:
    core = similarity_core()
    products = core["products"]
    best = core["best"]
    entries = core["entries"]
    sims = core["sims"]
    members = core["members"]
    n = len(products)

    ai, bi = np.triu_indices(n, 1)
    values = best[ai, bi]
    keep = np.isfinite(values) & (values >= min_sim)
    if same_packaging:
        keep &= np.array([products[a][1] == products[b][1] for a, b in zip(ai, bi)], dtype=bool)
    idx = np.flatnonzero(keep)
    idx = idx[np.argsort(-values[idx])][:top]

    pairs = []
    for t in idx:
        a, b = int(ai[t]), int(bi[t])
        sub = sims[np.ix_(members[a], members[b])]
        i, j = np.unravel_index(int(sub.argmax()), sub.shape)
        path_a = entries[int(members[a][i])]["image_path"]
        path_b = entries[int(members[b][j])]["image_path"]
        pairs.append(
            {
                "a": {
                    "sku_id": products[a][0],
                    "packaging_type": products[a][1],
                    "n_images": len(members[a]),
                    "image_url": media_url(path_a),
                },
                "b": {
                    "sku_id": products[b][0],
                    "packaging_type": products[b][1],
                    "n_images": len(members[b]),
                    "image_url": media_url(path_b),
                },
                "similarity": round(float(values[t]), 4),
            }
        )
    return {
        "model": core["model"],
        "num_vectors": core["num_vectors"],
        "n_products": n,
        "pairs": pairs,
        "products": core["per_product"],
        "hist": core["hist"],
        "summary": core["summary"],
    }


# ---------------------------------------------------------------------- #
# Routes


@router.get("/overview")
def overview() -> dict:
    return build_overview()


@router.get("/similarity")
def similarity(
    top: int = Query(25, ge=1, le=200),
    same_packaging: bool = False,
    min_sim: float = Query(0.0, ge=-1.0, le=1.0),
) -> dict:
    return build_similarity(top, same_packaging, min_sim)


@router.get("/media/{rel:path}")
def media(rel: str):
    root = DATA_ROOT.resolve()
    target = (DATA_ROOT / rel).resolve()
    if (
        not target.is_relative_to(root)
        or target.suffix.lower() not in IMAGE_EXTENSIONS
        or not target.is_file()
    ):
        raise HTTPException(404)
    return FileResponse(target)


# ---------------------------------------------------------------------- #
# Standalone app


app = FastAPI(title="Catalog data dashboard")
app.include_router(router)
app.include_router(catalog_router)
app.state.pipeline = None


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/dashboard.html")


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
