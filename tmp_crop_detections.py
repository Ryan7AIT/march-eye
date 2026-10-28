"""Temp smoke test: run YOLO detect, then save clean product crops (no boxes).

Crops are rectangular cutouts from the original shelf photo — the input Stage 2
needs for similarity search. No rembg; box overlays are not drawn on crops.
The detector is 1-class (object); packaging is not written into crop names.

Example:
    python tmp_crop_detections.py
    python tmp_crop_detections.py --device cpu
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect products and save clean crops to a temp folder.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/sku110k/weights/best.pt"),
        help="Trained 1-class YOLO weights",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/shelf_images/17.jpg"),
        help="Shelf image to run predict on",
    )
    parser.add_argument("--imgsz", type=int, default=940)
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=700, help="Max detections per image")
    parser.add_argument("--device", default="cpu", help="cpu, 0, or cuda:0")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/crops"),
        help="Directory for clean product crops",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=0.05,
        help="Relative padding around each box (0.05 = 5%% of box size)",
    )
    parser.add_argument(
        "--clear",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear --out-dir before writing (default: on)",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Also save the YOLO annotated predict image under runs/",
    )
    return parser.parse_args()


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return x1, y1, x2, y2


def expand_box(
    x1: float, y1: float, x2: float, y2: float, pad: float, w: int, h: int
) -> tuple[int, int, int, int]:
    bw = x2 - x1
    bh = y2 - y1
    dx = bw * pad
    dy = bh * pad
    return clamp_box(int(x1 - dx), int(y1 - dy), int(x2 + dx), int(y2 + dy), w, h)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.source.is_file():
        raise FileNotFoundError(f"Source image not found: {args.source}")

    image = cv2.imread(str(args.source))
    if image is None:
        raise RuntimeError(f"Failed to read image: {args.source}")
    height, width = image.shape[:2]

    if args.clear and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    results = model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        save=args.save_annotated,
        verbose=False,
    )

    stem = args.source.stem
    saved = 0
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            logging.warning("No detections for %s", args.source.name)
            continue

        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())

            x1, y1, x2, y2 = expand_box(*xyxy, pad=args.pad, w=width, h=height)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            out_name = f"{stem}_{i:04d}_{conf:.2f}.jpg"
            out_path = args.out_dir / out_name
            cv2.imwrite(str(out_path), crop)
            saved += 1

    logging.info("Saved %d clean crops to %s", saved, args.out_dir.resolve())


if __name__ == "__main__":
    main()
