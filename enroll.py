"""Copy crops and manage SKU folders under data/single_products/{packaging}/{sku}/.

Used by the web correction UI and the catalog manager. Disk writes do not
update the FAISS gallery by themselves — the catalog API does that when the
pipeline is loaded; otherwise re-run embed_gallery.py.
"""

from __future__ import annotations

import hashlib
import io
import re
import shutil
import uuid
from pathlib import Path

from PIL import Image

PACKAGING_CLASSES = ("bottle", "can", "carton")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SKU_ALLOWED = re.compile(r"[^a-z0-9._-]+")
FILENAME_ALLOWED = re.compile(r"[^a-zA-Z0-9._-]+")
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")
CROP_NAME_RE = re.compile(
    r"^[A-Za-z0-9._-]+\.(jpg|jpeg|png|webp|bmp)$", re.IGNORECASE
)


class EnrollError(ValueError):
    """Invalid correction payload or missing crop file."""


def list_skus(root: Path) -> list[dict]:
    """Return SKU folders on disk: {sku_id, packaging_type}."""
    items: list[dict] = []
    for packaging in PACKAGING_CLASSES:
        class_dir = root / packaging
        if not class_dir.is_dir():
            continue
        for sku_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            items.append({"sku_id": sku_dir.name, "packaging_type": packaging})
    return items


def sanitize_sku_id(raw: str) -> str:
    sku = (raw or "").strip().lower().replace(" ", "_")
    sku = SKU_ALLOWED.sub("", sku).strip("._-")
    if not sku or ".." in sku:
        raise EnrollError("Invalid sku_id.")
    return sku


def _safe_request_id(request_id: str) -> str:
    value = (request_id or "").strip()
    if not REQUEST_ID_RE.fullmatch(value):
        raise EnrollError("Invalid request_id.")
    return value


def _safe_crop_name(crop_name: str) -> str:
    value = Path((crop_name or "").strip()).name
    if not CROP_NAME_RE.fullmatch(value):
        raise EnrollError("Invalid crop_name.")
    return value


def resolve_crop(results_root: Path, request_id: str, crop_name: str) -> Path:
    request_id = _safe_request_id(request_id)
    crop_name = _safe_crop_name(crop_name)
    crops_root = (results_root / request_id / "crops").resolve()
    crop_path = (crops_root / crop_name).resolve()
    if not crop_path.is_relative_to(crops_root) or not crop_path.is_file():
        raise EnrollError("Crop not found.")
    return crop_path


def save_crop(
    *,
    results_root: Path,
    gallery_root: Path,
    request_id: str,
    crop_name: str,
    sku_id: str,
    packaging_type: str,
) -> dict:
    packaging = (packaging_type or "").strip().lower()
    if packaging not in PACKAGING_CLASSES:
        raise EnrollError("packaging_type must be bottle, can, or carton.")

    sku_id = sanitize_sku_id(sku_id)
    crop_path = resolve_crop(results_root, request_id, crop_name)

    sku_dir = (gallery_root / packaging / sku_id).resolve()
    gallery_root = gallery_root.resolve()
    if not sku_dir.is_relative_to(gallery_root):
        raise EnrollError("Invalid destination folder.")

    created_sku = not sku_dir.is_dir()
    sku_dir.mkdir(parents=True, exist_ok=True)

    dest_name = f"{_safe_request_id(request_id)}_{_safe_crop_name(crop_name)}"
    dest = sku_dir / dest_name
    already = dest.is_file()
    if not already:
        shutil.copy2(crop_path, dest)

    rel = dest.relative_to(gallery_root).as_posix()
    return {
        "saved": True,
        "path": rel,
        "sku_id": sku_id,
        "packaging_type": packaging,
        "created_sku": created_sku,
        "already_existed": already,
        "image_path": posix_from_cwd(dest),
        "filename": dest.name,
    }


# ---------------------------------------------------------------------- #
# Catalog manager (create / add / delete / move SKUs and images)


def sanitize_packaging(raw: str) -> str:
    packaging = (raw or "").strip().lower()
    if packaging not in PACKAGING_CLASSES:
        raise EnrollError("packaging_type must be bottle, can, or carton.")
    return packaging


def posix_from_cwd(path: Path) -> str:
    """Path as posix relative to the process cwd, matching gallery metadata."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def sku_dir_of(gallery_root: Path, packaging: str, sku_id: str) -> Path:
    gallery_root = gallery_root.resolve()
    sku_dir = (gallery_root / packaging / sku_id).resolve()
    if not sku_dir.is_relative_to(gallery_root):
        raise EnrollError("Invalid destination folder.")
    return sku_dir


def list_sku_images(sku_dir: Path) -> list[Path]:
    if not sku_dir.is_dir():
        return []
    return sorted(
        p for p in sku_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def sanitize_filename(raw: str, default_ext: str = ".jpg") -> str:
    name = Path((raw or "image").strip()).name
    stem = Path(name).stem
    ext = Path(name).suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        ext = default_ext if default_ext in IMAGE_EXTENSIONS else ".jpg"
    stem = FILENAME_ALLOWED.sub("_", stem).strip("._-") or "image"
    if ".." in stem:
        raise EnrollError("Invalid filename.")
    return f"{stem}{ext}"


def decode_image(data: bytes) -> Image.Image:
    if not data:
        raise EnrollError("Empty file.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image.convert("RGB")
    except Exception as exc:
        raise EnrollError("Not a valid image.") from exc


def unique_image_path(sku_dir: Path, filename: str) -> Path:
    dest = sku_dir / filename
    if not dest.exists():
        return dest
    stem = dest.stem
    ext = dest.suffix
    return sku_dir / f"{stem}_{uuid.uuid4().hex[:6]}{ext}"


def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_sku(gallery_root: Path, sku_id: str, packaging_type: str) -> dict:
    packaging = sanitize_packaging(packaging_type)
    sku_id = sanitize_sku_id(sku_id)
    sku_dir = sku_dir_of(gallery_root, packaging, sku_id)
    if sku_dir.is_dir():
        raise EnrollError(f"SKU already exists: {packaging}/{sku_id}")
    sku_dir.mkdir(parents=True, exist_ok=False)
    return {
        "sku_id": sku_id,
        "packaging_type": packaging,
        "created": True,
        "n_images": 0,
    }


def add_image_bytes(
    gallery_root: Path,
    sku_id: str,
    packaging_type: str,
    filename: str,
    data: bytes,
) -> dict:
    packaging = sanitize_packaging(packaging_type)
    sku_id = sanitize_sku_id(sku_id)
    decode_image(data)
    sku_dir = sku_dir_of(gallery_root, packaging, sku_id)
    created_sku = not sku_dir.is_dir()
    sku_dir.mkdir(parents=True, exist_ok=True)

    incoming = _md5_bytes(data)
    for existing in list_sku_images(sku_dir):
        if _md5_file(existing) == incoming:
            return {
                "saved": True,
                "already_existed": True,
                "created_sku": False,
                "sku_id": sku_id,
                "packaging_type": packaging,
                "filename": existing.name,
                "path": existing.relative_to(gallery_root.resolve()).as_posix(),
                "image_path": posix_from_cwd(existing),
            }

    dest = unique_image_path(sku_dir, sanitize_filename(filename))
    dest.write_bytes(data)
    return {
        "saved": True,
        "already_existed": False,
        "created_sku": created_sku,
        "sku_id": sku_id,
        "packaging_type": packaging,
        "filename": dest.name,
        "path": dest.relative_to(gallery_root.resolve()).as_posix(),
        "image_path": posix_from_cwd(dest),
    }


def delete_image(
    gallery_root: Path,
    sku_id: str,
    packaging_type: str,
    filename: str,
) -> dict:
    packaging = sanitize_packaging(packaging_type)
    sku_id = sanitize_sku_id(sku_id)
    sku_dir = sku_dir_of(gallery_root, packaging, sku_id)
    safe_name = Path(filename).name
    if safe_name != filename or ".." in safe_name:
        raise EnrollError("Invalid filename.")
    target = (sku_dir / safe_name).resolve()
    if not target.is_relative_to(sku_dir) or not target.is_file():
        raise EnrollError("Image not found.")
    if target.suffix.lower() not in IMAGE_EXTENSIONS:
        raise EnrollError("Not a gallery image.")
    image_path = posix_from_cwd(target)
    target.unlink()
    return {
        "deleted": True,
        "sku_id": sku_id,
        "packaging_type": packaging,
        "filename": safe_name,
        "image_path": image_path,
        "n_images": len(list_sku_images(sku_dir)),
    }


def delete_sku(
    gallery_root: Path,
    sku_id: str,
    packaging_type: str,
    *,
    force: bool = False,
) -> dict:
    packaging = sanitize_packaging(packaging_type)
    sku_id = sanitize_sku_id(sku_id)
    sku_dir = sku_dir_of(gallery_root, packaging, sku_id)
    if not sku_dir.is_dir():
        raise EnrollError("SKU not found.")
    images = list_sku_images(sku_dir)
    if images and not force:
        raise EnrollError(f"SKU has {len(images)} images. Delete them first, or use force.")
    removed_paths = [posix_from_cwd(path) for path in images]
    for path in images:
        path.unlink()
    shutil.rmtree(sku_dir)
    return {
        "deleted": True,
        "sku_id": sku_id,
        "packaging_type": packaging,
        "removed_images": removed_paths,
    }


def move_sku(
    gallery_root: Path,
    sku_id: str,
    packaging_type: str,
    new_sku_id: str | None = None,
    new_packaging_type: str | None = None,
) -> dict:
    packaging = sanitize_packaging(packaging_type)
    sku_id = sanitize_sku_id(sku_id)
    dest_sku = sanitize_sku_id(new_sku_id) if new_sku_id else sku_id
    dest_pack = sanitize_packaging(new_packaging_type) if new_packaging_type else packaging
    if dest_sku == sku_id and dest_pack == packaging:
        return {
            "moved": False,
            "sku_id": sku_id,
            "packaging_type": packaging,
            "path_map": {},
        }

    src = sku_dir_of(gallery_root, packaging, sku_id)
    if not src.is_dir():
        raise EnrollError("SKU not found.")
    dest = sku_dir_of(gallery_root, dest_pack, dest_sku)
    if dest.exists():
        raise EnrollError(f"Target SKU already exists: {dest_pack}/{dest_sku}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    path_map: dict[str, str] = {}
    for path in list_sku_images(src):
        path_map[posix_from_cwd(path)] = posix_from_cwd(dest / path.name)
    shutil.move(str(src), str(dest))
    return {
        "moved": True,
        "sku_id": dest_sku,
        "packaging_type": dest_pack,
        "from_sku_id": sku_id,
        "from_packaging_type": packaging,
        "path_map": path_map,
    }
