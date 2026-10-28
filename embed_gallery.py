"""Build a FAISS gallery index from data/single_products/{class}/{SKU}/*.

Expects layout:
    data/single_products/bottle/RAMY_ORANGE_1L/01.jpg
    data/single_products/carton/CARTON_001/01.png

Writes:
    gallery/faiss.index
    gallery/metadata.json

Embeddings are DINOv3 CLS tokens (L2-normalized) on a tall letterbox canvas.
Query code must use the same model and the same canvas.

Example:
    python embed_gallery.py
    python embed_gallery.py --source data/single_products --out gallery --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PACKAGING_CLASSES = ("bottle", "can", "carton")

# ViT-B/16, 768-d. Production embedder. Weights are gated: accept the license
# at https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m before download.
DINOV3_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
EMBEDDER_MODELS = {
    "dinov3": DINOV3_MODEL,
}
# Target portrait canvas. Snapped down to a multiple of the model patch size
# (DINOv3 patch 16 -> 224x640).
CANVAS_WIDTH_TARGET = 224
CANVAS_HEIGHT_TARGET = 644
# Eval-only comparison model. Larger option: google/siglip2-so400m-patch14-384.
SIGLIP_MODEL = "google/siglip2-base-patch16-224"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed single-product gallery into FAISS.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/single_products"),
        help="Root with bottle/can/carton SKU folders",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("gallery"),
        help="Output directory for faiss.index + metadata.json",
    )
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    parser.add_argument(
        "--model",
        default=DINOV3_MODEL,
        help=(
            "Hugging Face DINO id, or alias dinov3 "
            f"(default: {DINOV3_MODEL})"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def collect_gallery_images(source: Path) -> list[dict]:
    """Return records for images under source/{class}/{sku}/file."""
    if not source.is_dir():
        raise FileNotFoundError(f"Gallery source not found: {source}")

    records: list[dict] = []
    for packaging in PACKAGING_CLASSES:
        class_dir = source / packaging
        if not class_dir.is_dir():
            logging.warning("Missing class folder: %s", class_dir)
            continue

        # Warn about loose files sitting directly under the class folder.
        for loose in sorted(class_dir.iterdir()):
            if loose.is_file() and loose.suffix.lower() in IMAGE_EXTENSIONS:
                logging.warning(
                    "Ignoring loose image (put it under a SKU folder): %s",
                    loose,
                )

        for sku_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            sku_id = sku_dir.name
            images = sorted(
                p
                for p in sku_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not images:
                logging.warning("SKU folder has no images: %s", sku_dir)
                continue
            for image_path in images:
                records.append(
                    {
                        "sku_id": sku_id,
                        "packaging_type": packaging,
                        "image_path": str(image_path.as_posix()),
                    }
                )

    if not records:
        raise FileNotFoundError(
            f"No gallery images found under {source}/{{bottle,can,carton}}/{{SKU}}/"
        )
    return records


def resolve_model_id(model: str) -> str:
    """Map a short alias (dinov3) to a Hugging Face id."""
    return EMBEDDER_MODELS.get(model, model)


def embedder_family(model_id: str) -> str:
    """Label stored in gallery metadata: dinov3, or dino."""
    name = model_id.lower()
    if "dinov3" in name:
        return "dinov3"
    return "dino"


def patch_size_of(model) -> int:
    patch = getattr(model.config, "patch_size", 14)
    if isinstance(patch, (list, tuple)):
        patch = patch[0]
    return int(patch)


def canvas_for_patch(
    patch_size: int,
    width_target: int = CANVAS_WIDTH_TARGET,
    height_target: int = CANVAS_HEIGHT_TARGET,
) -> tuple[int, int]:
    """Snap a portrait canvas down so both sides divide evenly by the patch size."""
    if patch_size <= 0:
        raise ValueError(f"patch size must be positive, got {patch_size}")
    width = width_target - (width_target % patch_size)
    height = height_target - (height_target % patch_size)
    if width < patch_size or height < patch_size:
        raise ValueError(
            f"Canvas target {width_target}x{height_target} is smaller than patch size {patch_size}"
        )
    return width, height


def letterbox_rect(
    image: Image.Image,
    width: int = CANVAS_WIDTH_TARGET,
    height: int = CANVAS_HEIGHT_TARGET,
) -> Image.Image:
    """Fit image inside width×height, preserve aspect ratio, pad with black."""
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    new_w = max(1, int(round(image.width * scale)))
    new_h = max(1, int(round(image.height * scale)))
    resized = image.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
    return canvas


def letterbox(image: Image.Image, size: int) -> Image.Image:
    """Scale the long side to `size` and pad so the whole bottle stays in frame."""
    image = image.convert("RGB")
    width, height = image.size
    scale = size / max(width, height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


class DinoV3Embedder:
    """CLS-token embeddings from a DINOv3 checkpoint, L2-normalized.

    The portrait canvas is snapped to the checkpoint patch size so the model
    sees the full letterbox instead of cropping a non-multiple height.
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        canvas_width: int = CANVAS_WIDTH_TARGET,
        canvas_height: int = CANVAS_HEIGHT_TARGET,
    ):
        from transformers import AutoImageProcessor, AutoModel

        self.model_id = resolve_model_id(model_id)
        self.device = device
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = AutoModel.from_pretrained(self.model_id).to(device).eval()
        except OSError as exc:
            if "gated repo" in str(exc).lower():
                raise RuntimeError(
                    f"Cannot download {self.model_id}. Open "
                    f"https://huggingface.co/{self.model_id} while logged in, submit the "
                    "access form, then rerun: "
                    "python embed_gallery.py --source data/single_products --out gallery"
                ) from exc
            raise
        self.patch_size = patch_size_of(self.model)
        self.canvas_width, self.canvas_height = canvas_for_patch(
            self.patch_size, canvas_width, canvas_height
        )
        if (self.canvas_width, self.canvas_height) != (canvas_width, canvas_height):
            logging.info(
                "Canvas %dx%d snapped to %dx%d for patch size %d",
                canvas_width,
                canvas_height,
                self.canvas_width,
                self.canvas_height,
                self.patch_size,
            )

    @torch.inference_mode()
    def embed_pils(self, images: list[Image.Image]) -> np.ndarray:
        canvases = [
            letterbox_rect(image, self.canvas_width, self.canvas_height) for image in images
        ]
        inputs = self.processor(
            images=canvases,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        cls = self.model(**inputs).last_hidden_state[:, 0].float()
        cls = cls / cls.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return cls.cpu().numpy().astype(np.float32)

    def embed_paths(self, paths: list[Path], batch_size: int) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in tqdm(range(0, len(paths), batch_size), desc="Embedding DINO"):
            batch = paths[start : start + batch_size]
            images = [Image.open(path) for path in batch]
            vectors.append(self.embed_pils(images))
        if not vectors:
            raise ValueError("No images to embed.")
        return np.vstack(vectors)


def _processor_image_size(processor) -> int:
    size = getattr(processor, "size", None) or {}
    if isinstance(size, dict):
        for key in ("height", "shortest_edge", "width"):
            if key in size:
                return int(size[key])
    return 224


class SiglipEmbedder:
    """Image-encoder embeddings from a SigLIP checkpoint, L2-normalized."""

    def __init__(self, model_id: str, device: str):
        from transformers import AutoModel, AutoProcessor

        self.model_id = model_id
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.image_size = _processor_image_size(self.processor)

    @torch.inference_mode()
    def embed_pils(self, images: list[Image.Image]) -> np.ndarray:
        canvases = [letterbox(image, self.image_size) for image in images]
        inputs = self.processor(
            images=canvases,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        pixel_values = inputs["pixel_values"].to(self.device)
        feats = self.model.get_image_features(pixel_values=pixel_values)
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output
        feats = feats.float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.cpu().numpy().astype(np.float32)

    def embed_paths(self, paths: list[Path], batch_size: int) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in tqdm(range(0, len(paths), batch_size), desc="Embedding SigLIP"):
            batch = paths[start : start + batch_size]
            images = [Image.open(path) for path in batch]
            vectors.append(self.embed_pils(images))
        if not vectors:
            raise ValueError("No images to embed.")
        return np.vstack(vectors)


def load_open_clip(model_name: str, pretrained: str, device: str):
    # "longest" letterboxes: scale so the long side fits image_size, then pad.
    # Default "shortest" center-crops tall bottles and drops cap / volume text.
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
        image_resize_mode="longest",
    )
    model.eval()
    return model, preprocess


@torch.inference_mode()
def embed_images(
    records: list[dict],
    model,
    preprocess,
    device: str,
    batch_size: int,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for start in tqdm(range(0, len(records), batch_size), desc="Embedding"):
        batch = records[start : start + batch_size]
        tensors = []
        for rec in batch:
            image = Image.open(rec["image_path"]).convert("RGB")
            tensors.append(preprocess(image))
        pixel_values = torch.stack(tensors).to(device)
        feats = model.encode_image(pixel_values)
        feats = feats.float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        vectors.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(vectors)


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def write_outputs(
    out_dir: Path,
    index: faiss.Index,
    records: list[dict],
    vectors: np.ndarray,
    model_id: str,
    canvas_width: int,
    canvas_height: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "faiss.index"
    meta_path = out_dir / "metadata.json"

    faiss.write_index(index, str(index_path))

    payload = {
        "embedder": embedder_family(model_id),
        "model": model_id,
        "image_canvas": {
            "width": canvas_width,
            "height": canvas_height,
        },
        "embedding_dim": int(vectors.shape[1]),
        "num_vectors": int(vectors.shape[0]),
        "entries": [
            {
                "id": i,
                "sku_id": rec["sku_id"],
                "packaging_type": rec["packaging_type"],
                "image_path": rec["image_path"],
            }
            for i, rec in enumerate(records)
        ],
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Wrote %s (%d vectors)", index_path, index.ntotal)
    logging.info("Wrote %s", meta_path)


def summarize(records: list[dict]) -> None:
    by_class: dict[str, set[str]] = {c: set() for c in PACKAGING_CLASSES}
    for rec in records:
        by_class[rec["packaging_type"]].add(rec["sku_id"])
    sku_total = len({rec["sku_id"] for rec in records})
    logging.info("Gallery: %d images, %d SKUs", len(records), sku_total)
    for packaging in PACKAGING_CLASSES:
        logging.info(
            "  %s: %d SKUs, %d images",
            packaging,
            len(by_class[packaging]),
            sum(1 for r in records if r["packaging_type"] == packaging),
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable; falling back to cpu")
        device = "cpu"

    records = collect_gallery_images(args.source)
    summarize(records)

    model_id = resolve_model_id(args.model)
    logging.info("Loading %s on %s", model_id, device)
    embedder = DinoV3Embedder(model_id, device)
    logging.info(
        "Embedding on a %dx%d canvas (patch %d)",
        embedder.canvas_width,
        embedder.canvas_height,
        embedder.patch_size,
    )
    vectors = embedder.embed_paths(
        [Path(rec["image_path"]) for rec in records],
        args.batch_size,
    )
    index = build_faiss_index(vectors)
    write_outputs(
        args.out,
        index,
        records,
        vectors,
        embedder.model_id,
        embedder.canvas_width,
        embedder.canvas_height,
    )


if __name__ == "__main__":
    main()
