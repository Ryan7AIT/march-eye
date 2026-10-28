"""FastAPI server for shelf product detection & SKU recognition.

All models (YOLO detector, DINOv3 encoder, FAISS gallery) are loaded
once at startup so every upload is answered fast.

Run:
    uvicorn app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from catalog import router as catalog_router
from dashboard import router as dashboard_router
from enroll import EnrollError, list_skus, save_crop
from pipeline import PipelineConfig, ShelfPipeline

logger = logging.getLogger("shelf-app")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

WEB_DIR = Path("web")
GALLERY_MEDIA_ROOT = Path("data/single_products").resolve()

pipeline: ShelfPipeline | None = None


class SaveCropRequest(BaseModel):
    request_id: str = Field(..., min_length=1)
    crop_name: str = Field(..., min_length=1)
    sku_id: str = Field(..., min_length=1)
    packaging_type: str = Field(..., min_length=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("Loading models (YOLO + DINOv3 + FAISS gallery)...")
    config = PipelineConfig()
    config.results_root.mkdir(parents=True, exist_ok=True)
    pipeline = ShelfPipeline(config)
    app.state.pipeline = pipeline
    logger.info(
        "Ready: detector=%s, embedder=%s, %d gallery vectors, %d catalog products",
        config.yolo_weights,
        config.embedder_model,
        pipeline.index.ntotal,
        len(pipeline.sku_catalog),
    )
    yield
    app.state.pipeline = None


app = FastAPI(title="Shelf Product Recognition", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data dashboard: http://127.0.0.1:8000/dashboard.html (API under /api/dashboard)
# Catalog manager: http://127.0.0.1:8000/catalog.html (API under /api/catalog)
app.include_router(dashboard_router)
app.include_router(catalog_router)


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, f"Unsupported file type: {file.content_type}")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 25 MB)")
    if not contents:
        raise HTTPException(400, "Empty file")

    try:
        # Inference is CPU-bound; run in a worker thread to keep the loop free.
        result = await anyio.to_thread.run_sync(pipeline.analyze, contents)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    return JSONResponse(result)


@app.get("/api/gallery")
async def gallery():
    items = [
        {
            "sku_id": g.sku_id,
            "packaging_type": g.packaging_type,
            "image_url": "/gallery-media/" + g.image_path,
        }
        for g in pipeline.sku_catalog.values()
    ]
    return {"skus": items}


@app.get("/api/skus")
async def skus():
    return {"skus": list_skus(GALLERY_MEDIA_ROOT)}


@app.post("/api/save-crop")
async def save_crop_api(body: SaveCropRequest):
    results_root = pipeline.config.results_root if pipeline else Path("tmp/web")
    try:
        result = save_crop(
            results_root=results_root,
            gallery_root=GALLERY_MEDIA_ROOT,
            request_id=body.request_id,
            crop_name=body.crop_name,
            sku_id=body.sku_id,
            packaging_type=body.packaging_type,
        )
    except EnrollError as exc:
        raise HTTPException(400, str(exc))
    return result


@app.get("/gallery-media/{path:path}")
async def gallery_media(path: str):
    resolved = (Path(".") / path).resolve()
    if not str(resolved).startswith(str(GALLERY_MEDIA_ROOT)):
        raise HTTPException(404)
    if not resolved.is_file():
        raise HTTPException(404)
    return FileResponse(resolved)


app.mount("/results", StaticFiles(directory="tmp/web", check_dir=False), name="results")
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")