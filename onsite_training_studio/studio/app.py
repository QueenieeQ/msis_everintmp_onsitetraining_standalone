"""FastAPI app for the Onsite Training Studio (standalone tool)."""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import products, registration_io, train_jobs, train_options
from .upload_routes import register_upload_routes

STATIC_DIR = Path(__file__).resolve().parent / "static"


class SaveShotRequest(BaseModel):
    stage: str
    camera: str
    printer_id: str
    shot: str
    label: str
    boxes: List[List[float]] = []


class TrainRequest(BaseModel):
    stage: str
    camera: str
    printer_id: str
    run_training: bool = True
    output_folder: str | None = None
    shot: str | None = None  # None = all shots; else train just this one shot
    model: str | None = None  # None = all models; else screw_hole|patches|screw_head
    model_variant: str | None = None  # None = onsite YAML default; else e.g. yolo11m.pt


def _product(stage: str, camera: str, printer_id: str) -> products.ProductRef:
    try:
        return products.find_product(stage, camera, printer_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _shot_data(product: products.ProductRef, shot: str):
    try:
        yaml_path = product.registration_yaml()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    data = registration_io.load_yaml(yaml_path)
    shot_data = data.get(shot)
    if not isinstance(shot_data, dict):
        raise HTTPException(status_code=404, detail=f"shot not found: {shot}")
    return yaml_path, data, shot_data


def create_app() -> FastAPI:
    app = FastAPI(title="Onsite Training Studio", version="0.1.0")
    register_upload_routes(app)

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "service": "onsite_training_studio"}

    @app.get("/api/products")
    def list_products() -> dict:
        rows = [
            {"stage": p.stage, "camera": p.camera, "printer_id": p.printer_id}
            for p in products.discover_products()
        ]
        return {
            "products": rows,
            "shots": list(registration_io.SHOT_NAMES),
            "labels": list(registration_io.LABELS),
            "models": [{"value": v, "label": l} for v, l in train_options.MODEL_CHOICES],
            "yolo_variants": list(train_options.YOLO_VARIANTS),
        }

    @app.get("/api/shot")
    def get_shot(stage: str, camera: str, printer_id: str, shot: str, label: str) -> dict:
        product = _product(stage, camera, printer_id)
        _, _, shot_data = _shot_data(product, shot)
        image_path = registration_io.shot_image_path(product.root, shot_data, shot)
        boxes = registration_io.get_boxes(shot_data, label)
        boxes_out = [{"id": box_id, "bbox": bbox} for box_id, bbox in boxes]
        if not image_path.is_file():
            return {"has_image": False, "image_path": None, "width": 0, "height": 0, "boxes": boxes_out}
        img = cv2.imread(str(image_path))
        if img is None:
            raise HTTPException(status_code=400, detail=f"cannot read image: {image_path}")
        h, w = img.shape[:2]
        return {
            "has_image": True,
            "image_path": str(image_path),
            "width": w,
            "height": h,
            "boxes": boxes_out,
        }

    @app.get("/api/image-file")
    def image_file(path: str) -> FileResponse:
        p = Path(path)
        if not p.is_file():
            raise HTTPException(status_code=404, detail="image not found")
        return FileResponse(p)

    @app.post("/api/shot")
    def save_shot(body: SaveShotRequest) -> dict:
        product = _product(body.stage, body.camera, body.printer_id)
        yaml_path, data, shot_data = _shot_data(product, body.shot)
        registration_io.set_boxes(shot_data, body.label, body.boxes)
        registration_io.save_yaml(yaml_path, data)
        return {"ok": True, "yaml_path": str(yaml_path), "count": len(body.boxes)}

    @app.post("/api/train")
    def start_train(body: TrainRequest) -> dict:
        product = _product(body.stage, body.camera, body.printer_id)
        if body.shot and body.shot not in registration_io.SHOT_NAMES:
            raise HTTPException(status_code=400, detail=f"unknown shot: {body.shot!r}")
        if body.model and body.model not in train_options.model_values():
            raise HTTPException(status_code=400, detail=f"unknown model: {body.model!r}")
        if body.model_variant and body.model_variant not in train_options.YOLO_VARIANTS:
            raise HTTPException(status_code=400, detail=f"unknown YOLO version: {body.model_variant!r}")
        job_id = train_jobs.start_train(
            product.information_path,
            run_training=body.run_training,
            output_folder=body.output_folder,
            shot=body.shot,
            model=body.model,
            model_variant=body.model_variant,
        )
        return {"job_id": job_id, "information_path": product.information_path}

    @app.get("/api/train/{job_id}")
    def train_status(job_id: str) -> dict:
        job = train_jobs.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {
            "id": job.id,
            "status": job.status,
            "ok": job.ok,
            "comment": job.comment,
            "output_root": job.output_root,
            "error_code": job.error_code,
            "information_path": job.information_path,
            "output_folder": job.output_folder,
            "shot": job.shot,
        }

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
