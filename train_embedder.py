"""Fine-tune DINOv3 with metric learning so lookalike SKUs separate.

Quick experiment script: is fine-tuning the right path?

Training data (SKU = (packaging, sku_id), same folder layout as the gallery):
    data/single_products/{bottle|can|carton}/{sku_id}/*.jpg     (default)
    + optional extra folders via --extra (e.g. data/train)

NEVER train on data/eval. Any training image whose bytes are identical to a
file under data/eval is skipped automatically (leak guard).

Recipe:
  * freeze the backbone, unfreeze only the last N transformer blocks (+ final norm)
  * P x K batches (P SKUs, K crops each); about half the SKUs in a batch come
    from the same brand/size family so lookalikes are hard negatives
  * supervised contrastive loss on the L2-normalized CLS token
  * shelf-like augmentation (blur, glare, color, small crop/rotation; no flips)

Output is a regular Hugging Face folder, so the existing scripts accept it:
    python train_embedder.py --out runs/dinov3_ft
    python eval_similarity.py --dinov3-model runs/dinov3_ft
    python embed_gallery.py --model runs/dinov3_ft --out gallery_ft

eval_similarity.py re-embeds the gallery with whatever model you pass, so you
do not need to rebuild gallery/ just to evaluate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from embed_gallery import (
    DINOV3_MODEL,
    IMAGE_EXTENSIONS,
    PACKAGING_CLASSES,
    canvas_for_patch,
    letterbox_rect,
    patch_size_of,
    resolve_model_id,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Metric-learning fine-tune of DINOv3.")
    p.add_argument("--source", type=Path, default=Path("data/single_products"))
    p.add_argument("--extra", type=Path, nargs="*", default=[], help="More SKU-layout folders")
    p.add_argument("--eval-dir", type=Path, default=Path("data/eval"), help="Leak guard only")
    p.add_argument("--model", default=DINOV3_MODEL)
    p.add_argument("--out", type=Path, default=Path("runs/dinov3_ft"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=300, help="Optimizer steps")
    p.add_argument("--skus-per-batch", type=int, default=8, help="P")
    p.add_argument("--imgs-per-sku", type=int, default=4, help="K")
    p.add_argument("--unfreeze-blocks", type=int, default=4, help="Last N blocks to train")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-amp", action="store_true", help="Disable fp16 autocast")
    return p.parse_args()


# ---------------------------------------------------------------------- #
# Data


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_train(roots: list[Path], eval_dir: Path) -> list[tuple[Path, int]]:
    eval_hashes: set[str] = set()
    if eval_dir.is_dir():
        for p in eval_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                eval_hashes.add(file_md5(p))

    products: dict[tuple[str, str], list[Path]] = defaultdict(list)
    leaked = 0
    for root in roots:
        for packaging in PACKAGING_CLASSES:
            class_dir = root / packaging
            if not class_dir.is_dir():
                continue
            for sku_dir in sorted(d for d in class_dir.iterdir() if d.is_dir()):
                for p in sorted(sku_dir.iterdir()):
                    if not (p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
                        continue
                    if eval_hashes and file_md5(p) in eval_hashes:
                        leaked += 1
                        continue
                    products[(sku_dir.name, packaging)].append(p)
    if leaked:
        logging.warning("Skipped %d training images identical to eval files", leaked)
    if not products:
        raise FileNotFoundError(f"No training images under {roots}")

    items: list[tuple[Path, int]] = []
    for label, key in enumerate(sorted(products)):
        for p in products[key]:
            items.append((p, label))
    return items


def family_of(sku_id: str, packaging: str) -> str:
    """Rough family for hard negatives: packaging + brand token."""
    return f"{packaging}:{sku_id.split('_')[0]}"


class SkuBatchSampler:
    """Yield P x K index batches. About half the SKUs share one family."""

    def __init__(self, labels: list[int], keys: list[tuple[str, str]], p: int, k: int, steps: int, seed: int):
        self.by_label: dict[int, list[int]] = defaultdict(list)
        for idx, lab in enumerate(labels):
            self.by_label[lab].append(idx)
        self.p, self.k, self.steps = p, k, steps
        self.rng = random.Random(seed)
        self.labels = sorted(self.by_label)
        self.by_family: dict[str, list[int]] = defaultdict(list)
        for lab in self.labels:
            sku, pack = keys[lab]
            self.by_family[family_of(sku, pack)].append(lab)
        self.families = [f for f, v in self.by_family.items() if len(v) >= 2]

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            chosen: list[int] = []
            if self.families:
                fam = self.by_family[self.rng.choice(self.families)]
                chosen += self.rng.sample(fam, min(len(fam), max(2, self.p // 2)))
            rest = [lab for lab in self.labels if lab not in chosen]
            need = self.p - len(chosen)
            if need > 0:
                chosen += self.rng.sample(rest, min(need, len(rest)))
            batch: list[int] = []
            for lab in chosen:
                pool = self.by_label[lab]
                if len(pool) >= self.k:
                    batch += self.rng.sample(pool, self.k)
                else:  # few photos: repeat, augmentation makes each view differ
                    batch += [self.rng.choice(pool) for _ in range(self.k)]
            yield batch


class RandomBlur:
    def __init__(self, p: float = 0.4, max_radius: float = 1.8):
        self.p, self.max_radius = p, max_radius

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, self.max_radius)))
        return img


class RandomGlare:
    """Soft bright blob, imitates shelf lighting / plastic reflections."""

    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        w, h = img.size
        arr = np.asarray(img).astype(np.float32)
        cx, cy = random.uniform(0, w), random.uniform(0, h)
        sigma = random.uniform(0.15, 0.4) * max(w, h)
        yy, xx = np.mgrid[0:h, 0:w]
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))
        arr = arr + blob[..., None] * random.uniform(40, 120)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class RandomDownscale:
    """Low-resolution shelf crops: shrink then upsample."""

    def __init__(self, p: float = 0.4):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        w, h = img.size
        s = random.uniform(0.3, 0.8)
        small = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR)
        return small.resize((w, h), Image.BICUBIC)


class SkuDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]], width: int, height: int, mean, std):
        self.items = items
        self.width, self.height = width, height
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.aug = T.Compose(
            [
                # mild geometry: keep label/volume text visible, no flips (they break text)
                T.RandomAffine(degrees=4, translate=(0.04, 0.04), scale=(0.9, 1.1)),
                T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.3, hue=0.02),
                RandomGlare(),
                RandomDownscale(),
                RandomBlur(),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        # random border trim, imitates imperfect detector boxes
        l, r = int(w * random.uniform(0, 0.06)), int(w * random.uniform(0, 0.06))
        t, b = int(h * random.uniform(0, 0.04)), int(h * random.uniform(0, 0.04))
        if w - l - r > 8 and h - t - b > 8:
            img = img.crop((l, t, w - r, h - b))
        img = self.aug(img)
        img = letterbox_rect(img, self.width, self.height)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
        x = (x - self.mean) / self.std
        return x, label


# ---------------------------------------------------------------------- #
# Model + loss


def freeze_all_but_last_blocks(model, n_blocks: int) -> list[str]:
    total = int(model.config.num_hidden_layers)
    first = total - n_blocks
    trainable: list[str] = []
    for name, param in model.named_parameters():
        train = False
        parts = name.split(".")
        if name.startswith("norm."):
            train = True  # final layernorm
        elif "layer" in parts:  # e.g. model.layer.11.attention.k_proj.weight
            i = parts.index("layer") + 1
            train = i < len(parts) and parts[i].isdigit() and int(parts[i]) >= first
        param.requires_grad = train
        if train:
            trainable.append(name)
    if not trainable:
        raise RuntimeError("Nothing to train; check the parameter names of this checkpoint.")
    return trainable


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    sim = (z @ z.T) / temperature
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    has_pos = pos.any(dim=1)
    mean_log_pos = (log_prob.masked_fill(~pos, 0.0)).sum(dim=1) / pos.sum(dim=1).clamp(min=1)
    return -(mean_log_pos[has_pos]).mean()


def batch_recall_at_1(z: torch.Tensor, labels: torch.Tensor) -> float:
    """How often a crop's nearest neighbor in the batch is the same SKU."""
    sim = z @ z.T
    sim.fill_diagonal_(float("-inf"))
    nearest = sim.argmax(dim=1)
    return float((labels[nearest] == labels).float().mean().item())


def plot_training(rows: list[dict], out_png: Path) -> None:
    """Loss, learning rate and in-batch recall@1 vs step."""
    if not rows:
        return
    steps = [r["step"] for r in rows]
    fig, (ax_loss, ax_rec) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax_loss.plot(steps, [r["loss"] for r in rows], color="#90caf9", alpha=0.6, label="loss (per step)")
    ax_loss.plot(steps, [r["ema_loss"] for r in rows], color="#1565c0", linewidth=2, label="loss (smoothed)")
    ax_loss.set_ylabel("contrastive loss")
    ax_loss.set_title("Training loss (lower is better)")
    ax_loss.grid(alpha=0.3)
    ax_lr = ax_loss.twinx()
    ax_lr.plot(steps, [r["lr"] for r in rows], color="#fb8c00", linestyle="--", label="learning rate")
    ax_lr.set_ylabel("learning rate")
    h1, l1 = ax_loss.get_legend_handles_labels()
    h2, l2 = ax_lr.get_legend_handles_labels()
    ax_loss.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    window = min(25, len(rows))
    rec = np.array([r["batch_recall1"] for r in rows], dtype=np.float32)
    smooth = np.convolve(rec, np.ones(window) / window, mode="valid")
    ax_rec.plot(steps, rec, color="#a5d6a7", alpha=0.6, label="in-batch recall@1")
    ax_rec.plot(steps[window - 1 :], smooth, color="#2e7d32", linewidth=2, label=f"{window}-step average")
    ax_rec.set_ylim(-0.02, 1.02)
    ax_rec.set_ylabel("recall@1")
    ax_rec.set_xlabel("step")
    ax_rec.set_title("Is the nearest crop in the batch the same SKU? (higher is better)")
    ax_rec.grid(alpha=0.3)
    ax_rec.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA unavailable; falling back to cpu (this will be slow)")
        device = "cpu"
    use_amp = device.startswith("cuda") and not args.no_amp

    from transformers import AutoImageProcessor, AutoModel

    model_id = resolve_model_id(args.model)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    width, height = canvas_for_patch(patch_size_of(model))

    trainable = freeze_all_but_last_blocks(model, args.unfreeze_blocks)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Trainable: %d tensors, %.1fM params (last %d blocks + norm)",
                 len(trainable), n_train / 1e6, args.unfreeze_blocks)

    items = collect_train([args.source, *args.extra], args.eval_dir)
    labels = [lab for _, lab in items]
    # Rebuild readable keys in label order for family grouping
    keys: dict[int, tuple[str, str]] = {}
    for path, lab in items:
        keys[lab] = (path.parent.name, path.parent.parent.name)
    key_list = [keys[i] for i in range(len(keys))]
    sizes = defaultdict(int)
    for lab in labels:
        sizes[lab] += 1
    logging.info("Train: %d images, %d SKUs (%d with a single image)",
                 len(items), len(keys), sum(1 for v in sizes.values() if v == 1))

    dataset = SkuDataset(items, width, height, processor.image_mean, processor.image_std)
    sampler = SkuBatchSampler(labels, key_list, args.skus_per_batch, args.imgs_per_sku,
                              args.steps, args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                        pin_memory=device.startswith("cuda"))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + np.cos(np.pi * prog))

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    ema = None
    args.out.mkdir(parents=True, exist_ok=True)
    log_rows: list[dict] = []
    log_fields = ["step", "loss", "ema_loss", "lr", "batch_recall1"]
    csv_file = (args.out / "loss.csv").open("w", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=log_fields)
    csv_writer.writeheader()
    pbar = tqdm(enumerate(loader), total=args.steps, desc="Training")
    for step, (x, y) in pbar:
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)
        x, y = x.to(device, non_blocking=True), y.to(device)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                            dtype=torch.float16, enabled=use_amp):
            cls = model(pixel_values=x).last_hidden_state[:, 0]
        z = F.normalize(cls.float(), dim=-1)
        loss = supcon_loss(z, y, args.temperature)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        loss_val = float(loss.item())
        ema = loss_val if ema is None else 0.95 * ema + 0.05 * loss_val
        recall = batch_recall_at_1(z.detach(), y)
        row = {
            "step": step + 1,
            "loss": round(loss_val, 6),
            "ema_loss": round(ema, 6),
            "lr": lr_at(step),
            "batch_recall1": round(recall, 4),
        }
        log_rows.append(row)
        csv_writer.writerow(row)
        pbar.set_postfix(loss=f"{ema:.3f}", rec=f"{recall:.2f}", lr=f"{lr_at(step):.1e}")
        if (step + 1) % 25 == 0:
            csv_file.flush()
            plot_training(log_rows, args.out / "loss.png")  # refresh while training

    csv_file.close()
    plot_training(log_rows, args.out / "loss.png")
    logging.info("Wrote %s and %s", args.out / "loss.csv", args.out / "loss.png")

    model.eval()
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    (args.out / "train_info.json").write_text(
        json.dumps(
            {
                "base_model": model_id,
                "steps": args.steps,
                "skus_per_batch": args.skus_per_batch,
                "imgs_per_sku": args.imgs_per_sku,
                "unfreeze_blocks": args.unfreeze_blocks,
                "lr": args.lr,
                "temperature": args.temperature,
                "train_images": len(items),
                "train_skus": len(keys),
                "final_ema_loss": None if not log_rows else log_rows[-1]["ema_loss"],
                "final_batch_recall1": None
                if not log_rows
                else round(float(np.mean([r["batch_recall1"] for r in log_rows[-50:]])), 4),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logging.info("Saved fine-tuned model to %s", args.out)
    print()
    print("Next, evaluate it (re-embeds the gallery with the new model):")
    print(f"  python eval_similarity.py --dinov3-model {args.out.as_posix()}")


if __name__ == "__main__":
    main()
