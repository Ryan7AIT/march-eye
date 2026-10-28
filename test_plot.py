"""Embed a few crops and compare cosine similarity before/after gallery-mean centering.

    .venv\\Scripts\\python.exe test_plot.py a.jpg b.jpg c.jpg
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from embed_gallery import DINOV3_MODEL, DinoV3Embedder
from recognize import gallery_vectors, load_gallery


def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(min=1e-12)


paths = [Path(p) for p in sys.argv[1:]]
labels = [p.stem[:18] for p in paths]

index, _ = load_gallery(Path("gallery"))
gallery = gallery_vectors(index)          # already L2-normalized, 768-d
mean = gallery.mean(axis=0, keepdims=True)

embedder = DinoV3Embedder(DINOV3_MODEL, "cpu")
raw = embedder.embed_paths(paths, batch_size=4)   # letterboxed + normalized
centered = normalize(raw - mean)

for name, vecs in ("raw", raw), ("centered", centered):
    sims = vecs @ vecs.T
    print(f"\n{name}")
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            print(f"  {labels[i]:<18} vs {labels[j]:<18}  {sims[i, j]:+.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, vecs, title in zip(axes, (raw, centered), ("raw cosine", "gallery-mean centered")):
    sims = vecs @ vecs.T
    im = ax.imshow(sims, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{sims[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=axes, fraction=0.03)
fig.savefig("tmp/vector_compare.png", dpi=120, bbox_inches="tight")
print("\nWrote tmp/vector_compare.png")