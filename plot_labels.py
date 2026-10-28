"""Draw YOLO bounding boxes onto a dataset image for quick visual QA.

Loads images the same way as auto_label.py (PIL + EXIF transpose) so
normalized label coordinates match the pixel grid they were written against.
Drawing follows save_verification_image.

Example:
    python plot_labels.py --image dataset/images/20260506_121410.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

CLASS_NAMES = {0: "bottle", 1: "can", 2: "carton"}
# Same RGB palette as auto_label.save_verification_image.
CLASS_COLORS = {0: (0, 220, 0), 1: (255, 180, 0), 2: (0, 190, 255)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay YOLO labels on an image and save a debug plot."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the original image (label is inferred from the stem).",
    )
    parser.add_argument(
        "--label",
        type=Path,
        default=None,
        help="Optional YOLO .txt path. Defaults to dataset/labels/<stem>.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/debug_plots"),
        help="Folder where the annotated image is written.",
    )
    return parser.parse_args()


def resolve_label_path(image_path: Path, label_path: Path | None) -> Path:
    if label_path is not None:
        return label_path
    return Path("dataset") / "labels" / f"{image_path.stem}.txt"


def load_image(path: Path) -> Image.Image:
    """Match auto_label.load_image: RGB pixels after EXIF transpose."""
    with Image.open(path) as opened:
        transposed = ImageOps.exif_transpose(opened)
        image = transposed.convert("RGB")
        image.info = {}
        return image


def load_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse YOLO lines: class_id x_center y_center width height (normalized)."""
    labels: list[tuple[int, float, float, float, float]] = []
    with label_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"{label_path}:{line_number}: expected 5 values, got {len(parts)}"
                )
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = (float(v) for v in parts[1:])
            labels.append((class_id, x_center, y_center, width, height))
    return labels


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Inverse of auto_label.yolo_lines: normalized YOLO -> absolute corners."""
    x1 = (x_center - box_width / 2.0) * image_width
    y1 = (y_center - box_height / 2.0) * image_height
    x2 = (x_center + box_width / 2.0) * image_width
    y2 = (y_center + box_height / 2.0) * image_height
    return x1, y1, x2, y2


def load_label_font(image_width: int, image_height: int) -> ImageFont.ImageFont:
    """Pick a readable TrueType size for large shelf photos; fall back to default."""
    # ~32px on a 4K edge; keeps labels visible without covering whole products.
    size = max(18, min(image_width, image_height) // 100)
    for font_path in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    ):
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def draw_labels(
    image: Image.Image,
    labels: list[tuple[int, float, float, float, float]],
) -> Image.Image:
    """Mirror auto_label.save_verification_image drawing style, with readable labels."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width, height = image.size
    font = load_label_font(width, height)

    for class_id, x_center, y_center, box_width, box_height in labels:
        color = CLASS_COLORS.get(class_id, (255, 255, 0))
        name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        x1, y1, x2, y2 = yolo_to_xyxy(
            x_center, y_center, box_width, box_height, width, height
        )
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text = name
        text_box = draw.textbbox((x1, y1), text, font=font)
        # Pad the background so text does not sit on the border.
        padded = (text_box[0] - 2, text_box[1] - 2, text_box[2] + 2, text_box[3] + 2)
        draw.rectangle(padded, fill=color)
        draw.text((x1, y1), text, fill=(0, 0, 0), font=font)

    return canvas


def main() -> int:
    args = parse_args()
    image_path = args.image
    if not image_path.is_file():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        print(f"Unsupported image type: {image_path.suffix}", file=sys.stderr)
        return 1

    label_path = resolve_label_path(image_path, args.label)
    if not label_path.is_file():
        print(f"Label not found: {label_path}", file=sys.stderr)
        return 1

    try:
        image = load_image(image_path)
    except (UnidentifiedImageError, OSError) as error:
        print(f"Failed to read image: {image_path} ({error})", file=sys.stderr)
        return 1

    labels = load_yolo_labels(label_path)
    annotated = draw_labels(image, labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{image_path.stem}_annotated.jpg"
    annotated.save(output_path, format="JPEG", quality=95)

    print(f"Drew {len(labels)} boxes -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
