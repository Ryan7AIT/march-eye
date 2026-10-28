"""Create YOLO pseudo-labels for retail-shelf product images with OWLv2.

Example:
    python auto_label.py --input-dir data/shelf_images --device cuda

Sliced inference is intentional: on a large shelf photograph, a single
full-image pass makes individual products too small for a zero-shot detector.
Extra tile scales and a full-image pass are available via --slice-sizes and
--full-image-pass, but the defaults stay conservative: low thresholds and
whole-image passes flood dense shelves with false positives.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from tqdm import tqdm
from transformers import Owlv2ForObjectDetection, Owlv2Processor


# Query order is important: OWLv2 returns the index of the matching query.
CLASS_QUERIES: dict[int, list[str]] = {
    0: ["a beverage bottle", "a plastic bottle", "a glass bottle"],
    1: ["a soda can", "an aluminum beverage can"],
    2: ["a juice carton", "a milk carton", "a cardboard drink box"],
}
CLASS_NAMES = {0: "bottle", 1: "can", 2: "carton"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Detection:
    """One detection in original-image pixel coordinates."""

    class_id: int
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


def build_queries() -> tuple[list[str], list[int]]:
    """Flatten class prompts and retain the class represented by each prompt."""
    queries: list[str] = []
    query_to_class: list[int] = []
    for class_id, prompts in CLASS_QUERIES.items():
        queries.extend(prompts)
        query_to_class.extend([class_id] * len(prompts))
    return queries, query_to_class


def slice_positions(length: int, slice_size: int, overlap_ratio: float) -> list[int]:
    """Return starts that cover an axis, including an edge-aligned final slice."""
    if length <= slice_size:
        return [0]
    stride = max(1, int(slice_size * (1.0 - overlap_ratio)))
    positions = list(range(0, length - slice_size, stride))
    last_start = length - slice_size
    if not positions or positions[-1] != last_start:
        positions.append(last_start)
    return positions


def image_slices(
    image: Image.Image,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
) -> Iterable[tuple[Image.Image, int, int]]:
    """Yield edge-aligned RGB slices together with their top-left coordinates."""
    width, height = image.size
    top_positions = slice_positions(height, slice_height, overlap_height_ratio)
    left_positions = slice_positions(width, slice_width, overlap_width_ratio)
    for top in top_positions:
        for left in left_positions:
            yield image.crop((left, top, min(left + slice_width, width), min(top + slice_height, height))), left, top


def collect_inference_tiles(
    image: Image.Image,
    slice_sizes: list[int],
    overlap_height_ratio: float,
    overlap_width_ratio: float,
    full_image_pass: bool,
) -> list[tuple[Image.Image, int, int]]:
    """Build multi-scale tiles, optionally including one full-image pass."""
    tiles: list[tuple[Image.Image, int, int]] = []
    for size in slice_sizes:
        tiles.extend(
            list(
                image_slices(
                    image,
                    size,
                    size,
                    overlap_height_ratio,
                    overlap_width_ratio,
                )
            )
        )
    if full_image_pass:
        tiles.append((image, 0, 0))
    return tiles


def translate_and_clip_box(
    box: Iterable[float],
    offset_x: int,
    offset_y: int,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Translate a slice-local box to global coordinates and clip it to the image."""
    x1, y1, x2, y2 = (float(value) for value in box)
    x1 = max(0.0, min(float(image_width), x1 + offset_x))
    y1 = max(0.0, min(float(image_height), y1 + offset_y))
    x2 = max(0.0, min(float(image_width), x2 + offset_x))
    y2 = max(0.0, min(float(image_height), y2 + offset_y))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available to PyTorch.")
    return device


def infer_tile(
    model: Owlv2ForObjectDetection,
    processor: Owlv2Processor,
    tile: Image.Image,
    queries: list[str],
    query_to_class: list[int],
    device: torch.device,
    use_fp16: bool,
    score_threshold: float,
    offset_x: int,
    offset_y: int,
    image_width: int,
    image_height: int,
) -> list[Detection]:
    """Run one OWLv2 image/query pass and translate boxes to source coordinates."""
    encoded = processor(text=[queries], images=tile, return_tensors="pt")
    encoded = {name: value.to(device) for name, value in encoded.items()}
    if use_fp16:
        encoded["pixel_values"] = encoded["pixel_values"].half()

    with torch.inference_mode():
        outputs = model(**encoded)

    # target_sizes is (height, width), not (width, height).
    target_sizes = torch.tensor([tile.size[::-1]], device=device)
    # Transformers renamed this processor method in newer releases (including
    # 5.x). The grounded method delegates to the same OWLv2 box conversion.
    post_process = getattr(processor, "post_process_object_detection", None)
    if post_process is None:
        post_process = processor.post_process_grounded_object_detection
    result = post_process(outputs=outputs, target_sizes=target_sizes, threshold=score_threshold)[0]

    boxes = result["boxes"].detach().float().cpu().numpy()
    scores = result["scores"].detach().float().cpu().numpy()
    labels = result["labels"].detach().cpu().numpy()
    detections: list[Detection] = []
    for box, score, query_index in zip(boxes, scores, labels):
        query_index = int(query_index)
        if query_index < 0 or query_index >= len(query_to_class):
            continue
        global_box = translate_and_clip_box(box, offset_x, offset_y, image_width, image_height)
        if global_box is not None:
            detections.append(Detection(query_to_class[query_index], float(score), *global_box))
    return detections


def filter_box_geometry(
    detections: list[Detection],
    image_width: int,
    image_height: int,
    max_area_ratio: float,
) -> list[Detection]:
    """Drop implausible product boxes.

    On a shelf photo a single facing never covers a large fraction of the
    image, so oversized boxes are background/shelf false positives. Extreme
    aspect ratios (long thin strips along shelf edges) are also rejected.
    """
    image_area = float(image_width * image_height)
    kept: list[Detection] = []
    for detection in detections:
        box_width = detection.x2 - detection.x1
        box_height = detection.y2 - detection.y1
        if box_width <= 1.0 or box_height <= 1.0:
            continue
        if (box_width * box_height) / image_area > max_area_ratio:
            continue
        aspect = max(box_width / box_height, box_height / box_width)
        if aspect > 6.0:
            continue
        kept.append(detection)
    return kept


def global_nms(detections: list[Detection], score_threshold: float, nms_iou: float) -> list[Detection]:
    """Merge every overlapping slice detection, including conflicting class labels."""
    if not detections:
        return []
    boxes_xywh = [
        [detection.x1, detection.y1, detection.x2 - detection.x1, detection.y2 - detection.y1]
        for detection in detections
    ]
    scores = [detection.score for detection in detections]
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold, nms_iou)
    kept = [detections[int(index)] for index in np.asarray(indices).reshape(-1)] if len(indices) else []
    return sorted(kept, key=lambda detection: (detection.class_id, -detection.score))


def yolo_lines(detections: list[Detection], width: int, height: int) -> str:
    """Convert pixel boxes to normalized YOLO label rows."""
    rows: list[str] = []
    for detection in detections:
        x_center = (detection.x1 + detection.x2) / (2.0 * width)
        y_center = (detection.y1 + detection.y2) / (2.0 * height)
        box_width = (detection.x2 - detection.x1) / width
        box_height = (detection.y2 - detection.y1) / height
        values = (x_center, y_center, box_width, box_height)
        # Clipping in infer_tile should guarantee this, but it also protects
        # output if a detector returns slightly out-of-bounds coordinates.
        values = tuple(max(0.0, min(1.0, value)) for value in values)
        if values[2] > 0.0 and values[3] > 0.0:
            rows.append(f"{detection.class_id} " + " ".join(f"{value:.6f}" for value in values))
    return "\n".join(rows) + ("\n" if rows else "")


def save_verification_image(image: Image.Image, detections: list[Detection], destination: Path) -> None:
    """Write a readable visual check image, retaining only successful samples."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    colors = {0: (0, 220, 0), 1: (255, 180, 0), 2: (0, 190, 255)}
    for detection in detections:
        color = colors[detection.class_id]
        box = (detection.x1, detection.y1, detection.x2, detection.y2)
        draw.rectangle(box, outline=color, width=3)
        text = f"{CLASS_NAMES[detection.class_id]} {detection.score:.2f}"
        text_box = draw.textbbox((detection.x1, detection.y1), text, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((detection.x1, detection.y1), text, fill=(0, 0, 0), font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=95)


def load_image(path: Path) -> Image.Image:
    """Load RGB pixels with EXIF orientation applied, and strip EXIF metadata.

    YOLO trainers typically honor EXIF Orientation. Labels are computed on the
    transposed pixel grid, so the training image must be the same pixels (without
    a leftover Orientation tag that would rotate them again at train time).
    """
    with Image.open(path) as opened:
        transposed = ImageOps.exif_transpose(opened)
        image = transposed.convert("RGB")
        image.info = {}
        return image


def save_training_image(image: Image.Image, destination: Path) -> None:
    """Persist the labeled pixel grid (already EXIF-transposed, no Orientation tag)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(destination, format="JPEG", quality=95)
    elif suffix == ".png":
        image.save(destination, format="PNG")
    else:
        image.save(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pseudo-label shelf products with OWLv2 and export YOLO labels.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/shelf_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--verification-dir", type=Path, default=Path("data/verification_samples"))
    parser.add_argument("--model-name", default="google/owlv2-base-patch16-ensemble")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a PyTorch device string such as cuda:0")
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--nms-iou", type=float, default=0.45, help="IoU threshold for global OpenCV NMS")
    parser.add_argument(
        "--slice-sizes",
        type=int,
        nargs="+",
        default=[640],
        help="Square tile edge lengths in pixels. Multi-scale example: --slice-sizes 640 1280",
    )
    parser.add_argument("--overlap-height-ratio", type=float, default=0.20, help="Vertical slice-overlap fraction")
    parser.add_argument("--overlap-width-ratio", type=float, default=0.20, help="Horizontal slice-overlap fraction")
    parser.add_argument(
        "--full-image-pass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run OWLv2 on the full (EXIF-corrected) image before NMS (default: off)",
    )
    parser.add_argument(
        "--max-box-area-ratio",
        type=float,
        default=0.05,
        help="Reject boxes covering more than this fraction of the image (background false positives)",
    )
    parser.add_argument("--fp16", action="store_true", help="Use float16 model inference on CUDA GPUs")
    parser.add_argument("--max-images", type=int, default=None, help="Optional limit useful for a smoke test")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be between 0 and 1.")
    if not 0.0 < args.nms_iou <= 1.0:
        raise ValueError("--nms-iou must be in (0, 1].")
    if not args.slice_sizes:
        raise ValueError("--slice-sizes must contain at least one positive integer.")
    if any(size <= 0 for size in args.slice_sizes):
        raise ValueError("--slice-sizes values must be positive integers.")
    if not 0.0 <= args.overlap_height_ratio < 1.0:
        raise ValueError("--overlap-height-ratio must be in [0, 1).")
    if not 0.0 <= args.overlap_width_ratio < 1.0:
        raise ValueError("--overlap-width-ratio must be in [0, 1).")
    if not 0.0 < args.max_box_area_ratio <= 1.0:
        raise ValueError("--max-box-area-ratio must be in (0, 1].")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive.")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        device = choose_device(args.device)
    except (ValueError, RuntimeError) as error:
        logging.error("Configuration error: %s", error)
        return 2

    use_fp16 = args.fp16 and device.type == "cuda"
    if args.fp16 and not use_fp16:
        logging.warning("--fp16 is ignored because the selected device is %s.", device)

    image_paths = sorted(
        path for path in args.input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        logging.error("No supported image files found under %s", args.input_dir)
        return 2

    images_dir = args.output_dir / "images"
    labels_dir = args.output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    args.verification_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading %s on %s%s", args.model_name, device, " (float16)" if use_fp16 else "")
    logging.info(
        "Inference config: score=%.2f nms_iou=%.2f slice_sizes=%s overlap=(%.2f, %.2f) full_image_pass=%s max_box_area=%.2f",
        args.score_threshold,
        args.nms_iou,
        args.slice_sizes,
        args.overlap_height_ratio,
        args.overlap_width_ratio,
        args.full_image_pass,
        args.max_box_area_ratio,
    )
    try:
        processor = Owlv2Processor.from_pretrained(args.model_name)
        model = Owlv2ForObjectDetection.from_pretrained(args.model_name).to(device).eval()
        if use_fp16:
            model = model.half()
    except Exception as error:  # Model download/cache/device issues should be actionable.
        logging.exception("Could not load OWLv2 model '%s': %s", args.model_name, error)
        return 1

    queries, query_to_class = build_queries()
    failures: list[tuple[str, str]] = []
    successful_samples = 0
    total_detections = 0
    for image_path in tqdm(image_paths, desc="Pseudo-labeling", unit="image", position=0):
        relative_path = image_path.relative_to(args.input_dir)
        try:
            image = load_image(image_path)
            width, height = image.size
            tiles = collect_inference_tiles(
                image,
                args.slice_sizes,
                args.overlap_height_ratio,
                args.overlap_width_ratio,
                args.full_image_pass,
            )
            logging.info(
                "%s: processing %d tiles (sizes=%s, full_image=%s) on %dx%d",
                relative_path,
                len(tiles),
                args.slice_sizes,
                args.full_image_pass,
                width,
                height,
            )
            raw_detections: list[Detection] = []
            for image_slice, offset_x, offset_y in tqdm(
                tiles,
                desc=str(relative_path),
                unit="patch",
                leave=False,
                position=1,
            ):
                raw_detections.extend(
                    infer_tile(
                        model, processor, image_slice, queries, query_to_class, device, use_fp16,
                        args.score_threshold, offset_x, offset_y, width, height,
                    )
                )
            raw_detections = filter_box_geometry(raw_detections, width, height, args.max_box_area_ratio)
            detections = global_nms(raw_detections, args.score_threshold, args.nms_iou)

            output_image = images_dir / relative_path
            output_label = (labels_dir / relative_path).with_suffix(".txt")
            output_label.parent.mkdir(parents=True, exist_ok=True)
            # Save the same EXIF-corrected pixels the boxes were computed on.
            save_training_image(image, output_image)
            output_label.write_text(yolo_lines(detections, width, height), encoding="utf-8")

            if successful_samples < 10:
                sample_name = f"{successful_samples + 1:02d}_{image_path.stem}.jpg"
                save_verification_image(image, detections, args.verification_dir / sample_name)
                successful_samples += 1
            total_detections += len(detections)
        except (UnidentifiedImageError, OSError, ValueError, RuntimeError, cv2.error) as error:
            failures.append((str(relative_path), f"{type(error).__name__}: {error}"))
            logging.warning("Skipping %s: %s", relative_path, error)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as error:  # Keep a 10,000-image run alive on unexpected per-file problems.
            failures.append((str(relative_path), f"Unexpected {type(error).__name__}: {error}"))
            logging.exception("Skipping %s due to an unexpected error", relative_path)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    failure_log = args.output_dir / "failed_images.csv"
    if failures:
        with failure_log.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["image", "error"])
            writer.writerows(failures)
        logging.warning("Skipped %d image(s); details saved to %s", len(failures), failure_log)
    else:
        # Do not leave errors from an earlier run beside a clean dataset.
        failure_log.unlink(missing_ok=True)
    logging.info(
        "Done. Processed %d/%d images, exported %d boxes, and saved %d verification sample(s).",
        len(image_paths) - len(failures), len(image_paths), total_detections, successful_samples,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
