"""SKU catalog manager: list, create, rename, add/delete images, visual search.

Mounted in the main app (models loaded) and the standalone dashboard (disk only).
Visual search and live FAISS updates run only when a ShelfPipeline is on app.state.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import anyio
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from enroll import (
    IMAGE_EXTENSIONS,
    PACKAGING_CLASSES,
    EnrollError,
    add_image_bytes,
    create_sku,
    decode_image,
    delete_image,
    delete_sku,
    list_sku_images,
    move_sku,
    posix_from_cwd,
    sanitize_packaging,
    sanitize_sku_id,
    sku_dir_of,
)

GALLERY_ROOT = Path("data/single_products")
GALLERY_DIR = Path("gallery")
MIN_GOOD_IMAGES = 3
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_FILES_PER_REQUEST = 30

router = APIRouter(prefix="/api/catalog", tags=["catalog"])


class CreateSkuBody(BaseModel):
    sku_id: str = Field(..., min_length=1)
    packaging_type: str = Field(..., min_length=1)


class PatchSkuBody(BaseModel):
    sku_id: str | None = None
    packaging_type: str | None = None


def pipeline_of(request: Request):
    return getattr(request.app.state, "pipeline", None)


def media_url(packaging: str, sku_id: str, filename: str) -> str:
    return f"/api/catalog/media/{quote(packaging)}/{quote(sku_id)}/{quote(filename)}"


def url_from_entry_path(image_path: str | None) -> str | None:
    if not image_path:
        return None
    parts = Path(image_path).as_posix().split("/")
    for i, part in enumerate(parts):
        if part in PACKAGING_CLASSES and i + 2 < len(parts):
            return media_url(part, parts[i + 1], parts[i + 2])
    return None


def image_info(path: Path, packaging: str, sku_id: str) -> dict:
    width = height = 0
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
    except Exception:
        pass
    st = path.stat()
    return {
        "name": path.name,
        "url": media_url(packaging, sku_id, path.name),
        "w": width,
        "h": height,
        "bytes": st.st_size,
    }


def flags_for(n_images: int) -> list[str]:
    if n_images <= 0:
        return ["empty"]
    if n_images == 1:
        return ["single"]
    if n_images < MIN_GOOD_IMAGES:
        return ["few"]
    return []


def index_status(gallery_root: Path) -> dict:
    meta_path = GALLERY_DIR / "metadata.json"
    index_path = GALLERY_DIR / "faiss.index"
    if not meta_path.is_file() or not index_path.is_file():
        return {"exists": False, "stale": True}
    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    entry_paths = {entry["image_path"] for entry in meta.get("entries", [])}
    disk_paths: set[str] = set()
    for packaging in PACKAGING_CLASSES:
        class_dir = gallery_root / packaging
        if not class_dir.is_dir():
            continue
        for sku_dir in class_dir.iterdir():
            if not sku_dir.is_dir():
                continue
            for image in list_sku_images(sku_dir):
                disk_paths.add(posix_from_cwd(image))
    new_on_disk = disk_paths - entry_paths
    gone = entry_paths - disk_paths
    return {
        "exists": True,
        "model": meta.get("model"),
        "num_vectors": meta.get("num_vectors"),
        "stale": bool(new_on_disk or gone),
        "new_on_disk": len(new_on_disk),
        "missing_on_disk": len(gone),
    }


def list_catalog_skus(gallery_root: Path) -> list[dict]:
    rows: list[dict] = []
    for packaging in PACKAGING_CLASSES:
        class_dir = gallery_root / packaging
        if not class_dir.is_dir():
            continue
        for sku_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            images = list_sku_images(sku_dir)
            thumb = images[0] if images else None
            rows.append(
                {
                    "sku_id": sku_dir.name,
                    "packaging_type": packaging,
                    "n_images": len(images),
                    "thumb_url": media_url(packaging, sku_dir.name, thumb.name) if thumb else None,
                    "flags": flags_for(len(images)),
                }
            )
    return rows


def http_enroll(exc: EnrollError) -> HTTPException:
    return HTTPException(400, str(exc))


async def sync_add(request: Request, saved: list[dict]) -> bool:
    pipeline = pipeline_of(request)
    if pipeline is None:
        return False
    fresh = [row for row in saved if not row.get("already_existed")]
    if not fresh:
        return True
    from PIL import Image

    records = []
    images = []
    for row in fresh:
        path = GALLERY_ROOT / row["path"]
        records.append(
            {
                "sku_id": row["sku_id"],
                "packaging_type": row["packaging_type"],
                "image_path": row["image_path"],
            }
        )
        images.append(Image.open(path).convert("RGB"))
    await anyio.to_thread.run_sync(pipeline.add_gallery_images, records, images)
    return True


async def sync_remove(request: Request, image_paths: list[str]) -> bool:
    pipeline = pipeline_of(request)
    if pipeline is None or not image_paths:
        return pipeline is not None
    await anyio.to_thread.run_sync(pipeline.remove_gallery_paths, image_paths)
    return True


async def sync_rename(request: Request, result: dict) -> bool:
    pipeline = pipeline_of(request)
    if pipeline is None or not result.get("moved"):
        return pipeline is not None
    await anyio.to_thread.run_sync(
        pipeline.rename_gallery_sku,
        result["from_sku_id"],
        result["from_packaging_type"],
        result["sku_id"],
        result["packaging_type"],
        result["path_map"],
    )
    return True


@router.get("/skus")
def list_skus_api(request: Request) -> dict:
    rows = list_catalog_skus(GALLERY_ROOT)
    n_images = sum(row["n_images"] for row in rows)
    return {
        "skus": rows,
        "min_good_images": MIN_GOOD_IMAGES,
        "summary": {
            "n_skus": len(rows),
            "n_images": n_images,
            "n_need_photos": sum(1 for row in rows if row["n_images"] < MIN_GOOD_IMAGES),
            "n_empty": sum(1 for row in rows if row["n_images"] == 0),
            "by_packaging": {
                pack: {
                    "skus": sum(1 for row in rows if row["packaging_type"] == pack),
                    "images": sum(row["n_images"] for row in rows if row["packaging_type"] == pack),
                }
                for pack in PACKAGING_CLASSES
            },
        },
        "index": index_status(GALLERY_ROOT),
        "visual_search": pipeline_of(request) is not None,
    }


@router.post("/skus")
def create_sku_api(body: CreateSkuBody) -> dict:
    try:
        return create_sku(GALLERY_ROOT, body.sku_id, body.packaging_type)
    except EnrollError as exc:
        raise http_enroll(exc)


@router.get("/skus/{packaging}/{sku_id}")
def sku_detail(request: Request, packaging: str, sku_id: str) -> dict:
    try:
        packaging = sanitize_packaging(packaging)
        sku_id = sanitize_sku_id(sku_id)
        sku_dir = sku_dir_of(GALLERY_ROOT, packaging, sku_id)
    except EnrollError as exc:
        raise http_enroll(exc)
    if not sku_dir.is_dir():
        raise HTTPException(404, "SKU not found.")
    images = [image_info(path, packaging, sku_id) for path in list_sku_images(sku_dir)]
    neighbors: list[dict] = []
    pipeline = pipeline_of(request)
    if pipeline is not None:
        neighbors = [
            {
                **row,
                "image_url": url_from_entry_path(row.get("match_path")),
            }
            for row in pipeline.neighbors_of(sku_id, packaging)
        ]
    return {
        "sku_id": sku_id,
        "packaging_type": packaging,
        "n_images": len(images),
        "flags": flags_for(len(images)),
        "images": images,
        "neighbors": neighbors,
        "min_good_images": MIN_GOOD_IMAGES,
    }


@router.patch("/skus/{packaging}/{sku_id}")
async def patch_sku(request: Request, packaging: str, sku_id: str, body: PatchSkuBody) -> dict:
    try:
        result = move_sku(
            GALLERY_ROOT,
            sku_id,
            packaging,
            new_sku_id=body.sku_id,
            new_packaging_type=body.packaging_type,
        )
    except EnrollError as exc:
        raise http_enroll(exc)
    indexed = await sync_rename(request, result)
    result["indexed"] = indexed
    return result


@router.delete("/skus/{packaging}/{sku_id}")
async def delete_sku_api(
    request: Request,
    packaging: str,
    sku_id: str,
    force: bool = Query(False),
) -> dict:
    try:
        result = delete_sku(GALLERY_ROOT, sku_id, packaging, force=force)
    except EnrollError as exc:
        raise http_enroll(exc)
    indexed = await sync_remove(request, result["removed_images"])
    result["indexed"] = indexed
    return result


@router.post("/skus/{packaging}/{sku_id}/images")
async def add_images_api(
    request: Request,
    packaging: str,
    sku_id: str,
    files: list[UploadFile] = File(...),
) -> dict:
    if not files:
        raise HTTPException(400, "No files uploaded.")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(400, f"Too many files (max {MAX_FILES_PER_REQUEST}).")
    saved: list[dict] = []
    errors: list[str] = []
    for upload in files:
        data = await upload.read()
        if len(data) > MAX_UPLOAD_BYTES:
            errors.append(f"{upload.filename}: file too large (max 25 MB)")
            continue
        try:
            saved.append(
                add_image_bytes(
                    GALLERY_ROOT,
                    sku_id,
                    packaging,
                    upload.filename or "image.jpg",
                    data,
                )
            )
        except EnrollError as exc:
            errors.append(f"{upload.filename}: {exc}")
    if not saved and errors:
        raise HTTPException(400, "; ".join(errors))
    indexed = await sync_add(request, saved)
    return {
        "saved": saved,
        "errors": errors,
        "indexed": indexed,
        "sku_id": saved[0]["sku_id"] if saved else sanitize_sku_id(sku_id),
        "packaging_type": saved[0]["packaging_type"] if saved else packaging,
    }


@router.delete("/skus/{packaging}/{sku_id}/images/{filename}")
async def delete_image_api(
    request: Request,
    packaging: str,
    sku_id: str,
    filename: str,
) -> dict:
    try:
        result = delete_image(GALLERY_ROOT, sku_id, packaging, filename)
    except EnrollError as exc:
        raise http_enroll(exc)
    indexed = await sync_remove(request, [result["image_path"]])
    result["indexed"] = indexed
    return result


@router.post("/search-image")
async def search_image_api(request: Request, file: UploadFile = File(...)):
    pipeline = pipeline_of(request)
    if pipeline is None:
        raise HTTPException(
            503,
            "Visual search needs the main app (models loaded). Run: python -m uvicorn app:app",
        )
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 25 MB)")
    try:
        image = decode_image(data)
    except EnrollError as exc:
        raise http_enroll(exc)

    result = await anyio.to_thread.run_sync(pipeline.search_similar, image)
    catalog = { (row["packaging_type"], row["sku_id"]): row for row in list_catalog_skus(GALLERY_ROOT) }
    skus = []
    for row in result["skus"]:
        info = catalog.get((row["packaging_type"], row["sku_id"]), {})
        skus.append(
            {
                **row,
                "n_images": info.get("n_images"),
                "thumb_url": info.get("thumb_url") or url_from_entry_path(row.get("match_path")),
                "image_url": url_from_entry_path(row.get("match_path")),
                "flags": info.get("flags", []),
            }
        )
    images = [
        {
            **row,
            "url": url_from_entry_path(row.get("image_path")),
        }
        for row in result["images"]
    ]
    return {"skus": skus, "images": images}


@router.get("/media/{packaging}/{sku_id}/{filename}")
def catalog_media(packaging: str, sku_id: str, filename: str):
    try:
        packaging = sanitize_packaging(packaging)
        sku_id = sanitize_sku_id(sku_id)
        sku_dir = sku_dir_of(GALLERY_ROOT, packaging, sku_id)
    except EnrollError:
        raise HTTPException(404)
    target = (sku_dir / Path(filename).name).resolve()
    if (
        not target.is_relative_to(sku_dir)
        or not target.is_file()
        or target.suffix.lower() not in IMAGE_EXTENSIONS
    ):
        raise HTTPException(404)
    return FileResponse(target)
