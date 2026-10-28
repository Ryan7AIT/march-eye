"""Prepare SKU-110K image lists and train a 1-class YOLO detector.

Reads the official pre-split layout under data/SKU-110K/ (images/{train,val,test}
paired with labels/{train,val,test}) and writes Ultralytics list files under
yolo_data/ without copying the dataset.

Examples:
    python train_yolo.py --max-train-images 100 --max-val-images 20 --epochs 10 --imgsz 640 --batch 4 --device 0
    python train_yolo.py --epochs 50 --imgsz 640 --batch 4 --device 0
    python train_yolo.py --prepare-only --max-train-images 100 --max-val-images 20 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {0: "object"}
SPLITS = ("train", "val", "test")
MAX_DET = 700


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 1-class YOLO detector on SKU-110K.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/SKU-110K"),
        help="Dataset root with images/{train,val,test} and labels/{train,val,test}",
    )
    parser.add_argument(
        "--yolo-dir",
        type=Path,
        default=Path("yolo_data"),
        help="Output folder for data.yaml and train/val/test image lists",
    )
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=None,
        help="Cap training images after a seeded shuffle (default: use all)",
    )
    parser.add_argument(
        "--max-val-images",
        type=int,
        default=None,
        help="Cap validation images after a seeded shuffle (default: use all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image caps")
    parser.add_argument("--model", default="yolo26s.pt", help="Ultralytics model checkpoint")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--imgsz", type=int, default=980)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--device", default="cpu", help="cpu, 0, or cuda:0")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows)")
    parser.add_argument("--project", type=Path, default=Path("runs"), help="Ultralytics project dir")
    parser.add_argument("--name", default="sku110k", help="Run name under --project")
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Freeze backbone modules 0..N-1",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (epochs without fitness improvement)",
    )
    parser.add_argument(
        "--nbs",
        type=int,
        default=None,
        help="Nominal batch size for loss scaling (default: same as --batch)",
    )
    parser.add_argument(
        "--cls-pw",
        type=float,
        default=0.4,
        help="Class weight power for inverse-frequency weighting (0 disables)",
    )
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache train images in RAM (off by default; SKU-110K images are large)",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=20,
        help="Disable mosaic augmentation for the final N epochs",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Only write yolo_data/ lists, skip training")
    return parser.parse_args()


def has_presplit_layout(source_dir: Path) -> bool:
    return (source_dir / "images" / "train").is_dir() and (source_dir / "labels" / "train").is_dir()


def paired_samples_in_dir(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Return (image, label) pairs, matching labels by path relative to images_dir."""
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / image_path.relative_to(images_dir).with_suffix(".txt")
        if not label_path.is_file():
            logging.warning("Skipping %s: missing label %s", image_path.name, label_path)
            continue
        if label_path.stat().st_size == 0:
            logging.warning("Skipping %s: empty label file", image_path.name)
            continue
        pairs.append((image_path, label_path))
    return pairs


def collect_presplit_pairs(source_dir: Path) -> dict[str, list[tuple[Path, Path]]]:
    """Load official train/val/test pairs from images/<split> + labels/<split>."""
    if not has_presplit_layout(source_dir):
        raise FileNotFoundError(
            f"Expected pre-split layout {source_dir / 'images' / 'train'} and "
            f"{source_dir / 'labels' / 'train'}"
        )

    splits: dict[str, list[tuple[Path, Path]]] = {}
    for split in SPLITS:
        images_dir = source_dir / "images" / split
        labels_dir = source_dir / "labels" / split
        if not images_dir.is_dir():
            splits[split] = []
            continue
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"Expected labels folder {labels_dir}")
        splits[split] = paired_samples_in_dir(images_dir, labels_dir)
        logging.info("Found %d labeled %s image(s) under %s", len(splits[split]), split, images_dir)

    if not splits["train"]:
        raise FileNotFoundError(f"No labeled train image/label pairs found under {source_dir}")
    if not splits["val"]:
        raise FileNotFoundError(f"No labeled val image/label pairs found under {source_dir}")
    return splits


def cap_pairs(
    pairs: list[tuple[Path, Path]],
    max_images: int | None,
    seed: int,
    split: str,
) -> list[tuple[Path, Path]]:
    """Keep all pairs, or a seeded subset when max_images is set."""
    total = len(pairs)
    if max_images is None:
        logging.info("%s: %d / %d (all)", split, total, total)
        return pairs
    if max_images < 1:
        raise ValueError(f"--max-{split}-images must be >= 1 (got {max_images})")
    if max_images >= total:
        logging.warning(
            "%s: requested %d image(s) but only %d available; using all",
            split,
            max_images,
            total,
        )
        logging.info("%s: %d / %d (all)", split, total, total)
        return pairs

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    selected = shuffled[:max_images]
    logging.info("%s: %d / %d", split, len(selected), total)
    return selected


def write_image_list(list_path: Path, pairs: list[tuple[Path, Path]]) -> None:
    lines = [image_path.resolve().as_posix() for image_path, _ in pairs]
    list_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_data_yaml(yolo_dir: Path, has_test: bool) -> Path:
    """Write Ultralytics data.yaml pointing at image-list files."""
    yaml_path = yolo_dir / "data.yaml"
    root = yolo_dir.resolve().as_posix()
    lines = [
        f"path: {root}",
        "train: train.txt",
        "val: val.txt",
    ]
    if has_test:
        lines.append("test: test.txt")
    lines.extend(
        [
            f"nc: {len(CLASS_NAMES)}",
            "names:",
        ]
    )
    for class_id, name in CLASS_NAMES.items():
        lines.append(f"  {class_id}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def count_list_images(yolo_dir: Path, split: str) -> int:
    """Count image paths listed in yolo_dir/<split>.txt."""
    list_path = yolo_dir / f"{split}.txt"
    if not list_path.is_file():
        return 0
    return sum(1 for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip())


def log_accumulation_guard(train_count: int, batch: int, nbs: int) -> None:
    """Log optimizer step frequency and warn when gradient accumulation is too high."""
    batches_per_epoch = max(1, math.ceil(train_count / batch))
    accumulate = max(round(nbs / batch), 1)
    optimizer_steps_per_epoch = max(1, batches_per_epoch // accumulate)
    logging.info(
        "Train images: %d, batches/epoch: %d, nbs=%d, accumulate=%d, optimizer steps/epoch: %d",
        train_count,
        batches_per_epoch,
        nbs,
        accumulate,
        optimizer_steps_per_epoch,
    )
    if accumulate > 4:
        logging.warning(
            "Gradient accumulation is %d (nbs=%d, batch=%d). The optimizer may step only once "
            "every %d batches, starving training on small datasets. Set --nbs to match --batch.",
            accumulate,
            nbs,
            batch,
            accumulate,
        )


def prepare_dataset(
    source_dir: Path,
    yolo_dir: Path,
    max_train_images: int | None,
    max_val_images: int | None,
    seed: int,
) -> Path:
    """Write image-list files and data.yaml; keep official train/val/test splits."""
    splits = collect_presplit_pairs(source_dir)
    train_pairs = cap_pairs(splits["train"], max_train_images, seed, "train")
    val_pairs = cap_pairs(splits["val"], max_val_images, seed, "val")
    test_pairs = splits["test"]
    logging.info("test: %d / %d (all)", len(test_pairs), len(test_pairs))

    if len(val_pairs) < 3:
        logging.warning(
            "Validation split has only %d image(s). Early stopping and best.pt selection will be noisy.",
            len(val_pairs),
        )

    yolo_dir.mkdir(parents=True, exist_ok=True)
    write_image_list(yolo_dir / "train.txt", train_pairs)
    write_image_list(yolo_dir / "val.txt", val_pairs)
    has_test = bool(test_pairs)
    if has_test:
        write_image_list(yolo_dir / "test.txt", test_pairs)
    elif (yolo_dir / "test.txt").exists():
        (yolo_dir / "test.txt").unlink()

    yaml_path = write_data_yaml(yolo_dir, has_test=has_test)
    logging.info("Wrote %s", yaml_path)
    return yaml_path


def evaluate_test_split(
    weights_path: Path,
    yaml_path: Path,
    args: argparse.Namespace,
) -> None:
    """Run held-out evaluation on the test split."""
    from ultralytics import YOLO

    test_count = count_list_images(args.yolo_dir, "test")
    if test_count == 0:
        logging.info("No test images; skipping held-out evaluation.")
        return

    logging.info("Evaluating test split (%d image(s)) with %s", test_count, weights_path)
    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(yaml_path),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        max_det=MAX_DET,
        plots=True,
        verbose=True,
    )
    if metrics is not None and hasattr(metrics, "box"):
        box = metrics.box
        logging.info(
            "Test metrics: precision=%.4f recall=%.4f mAP50=%.4f mAP50-95=%.4f",
            float(box.mp),
            float(box.mr),
            float(box.map50),
            float(box.map),
        )


def train(yaml_path: Path, args: argparse.Namespace) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("ultralytics is not installed. Run: pip install ultralytics") from error

    nbs = args.nbs if args.nbs is not None else args.batch
    train_count = count_list_images(args.yolo_dir, "train")
    log_accumulation_guard(train_count, args.batch, nbs)

    model = YOLO(args.model)
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        freeze=args.freeze,
        patience=args.patience,
        nbs=nbs,
        cls_pw=args.cls_pw,
        cache=args.cache,
        close_mosaic=args.close_mosaic,
        seed=args.seed,
        cos_lr=True,
        max_det=MAX_DET,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=5.0,
        shear=2.0,
        perspective=0.0005,
        flipud=0.0,
        hsv_v=0.5,
        hsv_s=0.7,
        scale=0.35,
        plots=True,
        verbose=True,
    )
    save_dir = Path(getattr(results, "save_dir", args.project / args.name))
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    weights_path = best if best.is_file() else last
    logging.info("Training finished. best=%s last=%s", best, last)

    evaluate_test_split(weights_path, yaml_path, args)
    return weights_path


def main() -> int:
    args = parse_args()
    args.project = args.project.resolve()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        yaml_path = prepare_dataset(
            args.source_dir,
            args.yolo_dir,
            args.max_train_images,
            args.max_val_images,
            args.seed,
        )
    except (FileNotFoundError, ValueError) as error:
        logging.error("%s", error)
        return 2

    if args.prepare_only:
        logging.info("Prepare-only mode; skipping training.")
        return 0

    try:
        weights = train(yaml_path, args)
    except Exception:
        logging.exception("Training failed")
        return 1

    logging.info("Done. Use weights: %s", weights)
    return 0


if __name__ == "__main__":
    sys.exit(main())
