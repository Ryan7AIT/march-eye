"""Fine-tune the 1-class shelf detector on local labeled photos.

Reads flat pairs from dataset/images and dataset/labels. Older labels may be
3-class boxes (bottle/can/carton) or polygons. Every object is rewritten as a
class-0 box so it matches the production detector. Original files are not
changed. SKU-110K lists under yolo_data/ are not replaced.

The run starts from runs/sku110k/weights/best.pt and writes
runs/finetune_shelf/, which the web API loads for detection.

Example:
    python fine_tune_yolo.py --device 0
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VAL_COUNT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the shelf YOLO detector on dataset/.")
    parser.add_argument("--images", type=Path, default=Path("dataset/images"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("yolo_data/finetune"),
        help="Converted images, labels, lists, and data.yaml",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/sku110k/weights/best.pt"),
        help="Checkpoint to fine-tune",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=940)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=700)
    parser.add_argument("--val-count", type=int, default=VAL_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0", help="0, cuda:0, or cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--name", default="finetune_shelf")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the converted split and stop before training",
    )
    return parser.parse_args()


def collect_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Pair each image with a same-stem label. Warn and skip orphans."""
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    images = {
        path.stem: path
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    labels = {
        path.stem: path
        for path in sorted(labels_dir.iterdir())
        if path.is_file() and path.suffix.lower() == ".txt"
    }

    for stem in sorted(set(images) - set(labels)):
        logging.warning("Skipping %s: no label file", images[stem].name)
    for stem in sorted(set(labels) - set(images)):
        logging.warning("Skipping %s: no image file", labels[stem].name)

    pairs = [(images[stem], labels[stem]) for stem in sorted(set(images) & set(labels))]
    if not pairs:
        raise FileNotFoundError(f"No image/label pairs under {images_dir} and {labels_dir}")
    return pairs


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def line_to_box(line: str, source: Path) -> str | None:
    """Return one class-0 YOLO box, or None when the line is empty or invalid."""
    parts = line.split()
    if not parts:
        return None
    try:
        numbers = [float(part) for part in parts[1:]]
    except ValueError:
        logging.warning("Skipping a non-numeric line in %s", source.name)
        return None

    if len(numbers) == 4:
        cx, cy, width, height = numbers
    elif len(numbers) >= 6 and len(numbers) % 2 == 0:
        xs = numbers[0::2]
        ys = numbers[1::2]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
    else:
        logging.warning("Skipping a label line with %d numbers in %s", len(numbers), source.name)
        return None

    cx, cy = _clamp01(cx), _clamp01(cy)
    width, height = _clamp01(width), _clamp01(height)
    if width <= 0 or height <= 0:
        logging.warning("Skipping a zero-area box in %s", source.name)
        return None
    return f"0 {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"


def convert_label(label_path: Path) -> list[str]:
    boxes: list[str] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        box = line_to_box(line, label_path)
        if box is not None:
            boxes.append(box)
    return boxes


def link_or_copy(source: Path, destination: Path) -> None:
    """Reuse the image bytes when a hard link works, otherwise copy."""
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_split(
    pairs: list[tuple[Path, Path]],
    out_dir: Path,
    val_count: int,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Write converted train/val images and class-0 labels. Return image paths."""
    if val_count < 1:
        raise ValueError("--val-count must be at least 1")
    if len(pairs) <= val_count:
        raise ValueError(
            f"Need more than {val_count} labeled images to hold out a val split (found {len(pairs)})"
        )

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    val_pairs = shuffled[:val_count]
    train_pairs = shuffled[val_count:]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    train_images: list[Path] = []
    val_images: list[Path] = []
    for split, split_pairs, bucket in (
        ("train", train_pairs, train_images),
        ("val", val_pairs, val_images),
    ):
        image_dir = out_dir / "images" / split
        label_dir = out_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_path, label_path in split_pairs:
            boxes = convert_label(label_path)
            if not boxes:
                logging.warning("Skipping %s: no boxes after conversion", image_path.name)
                continue
            destination = image_dir / image_path.name
            link_or_copy(image_path, destination)
            (label_dir / f"{image_path.stem}.txt").write_text(
                "\n".join(boxes) + "\n",
                encoding="utf-8",
            )
            bucket.append(destination.resolve())

    if len(train_images) < 1 or len(val_images) < 1:
        raise ValueError("Conversion left an empty train or val split")

    (out_dir / "train.txt").write_text(
        "\n".join(path.as_posix() for path in train_images) + "\n",
        encoding="utf-8",
    )
    (out_dir / "val.txt").write_text(
        "\n".join(path.as_posix() for path in val_images) + "\n",
        encoding="utf-8",
    )
    yaml_path = out_dir / "data.yaml"
    root = out_dir.resolve().as_posix()
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names:",
                "  0: object",
                "",
            ]
        ),
        encoding="utf-8",
    )
    logging.info("Train images: %d", len(train_images))
    logging.info("Val images: %d", len(val_images))
    for path in val_images:
        logging.info("  val %s", path.name)
    logging.info("Wrote %s", yaml_path)
    return train_images, val_images


def train(args: argparse.Namespace, yaml_path: Path) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("ultralytics is not installed. Run: pip install ultralytics") from error

    if not args.model.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.model}")

    # Absolute project path so a loaded checkpoint cannot nest the run under runs/detect.
    # AdamW is set explicitly because optimizer='auto' ignores lr0.
    args.project = args.project.resolve()
    model = YOLO(str(args.model))
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
        nbs=args.batch,
        optimizer="AdamW",
        lr0=args.lr0,
        seed=args.seed,
        cos_lr=True,
        max_det=args.max_det,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=0,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,
        plots=True,
        verbose=True,
    )
    save_dir = Path(getattr(results, "save_dir", args.project / args.name))
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    weights_path = best if best.is_file() else last
    logging.info("Training finished. best=%s last=%s", best, last)
    logging.info("API weights were not changed. Compare before pointing the pipeline at %s", weights_path)
    return weights_path


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        pairs = collect_pairs(args.images, args.labels)
        write_split(pairs, args.out, args.val_count, args.seed)
    except (FileNotFoundError, ValueError) as error:
        logging.error("%s", error)
        return 2

    if args.prepare_only:
        logging.info("Prepare-only mode; skipping training.")
        return 0

    try:
        weights = train(args, args.out / "data.yaml")
    except Exception:
        logging.exception("Training failed")
        return 1

    logging.info("Done. New weights: %s", weights)
    return 0


if __name__ == "__main__":
    sys.exit(main())
