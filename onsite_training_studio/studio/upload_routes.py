"""Create-product + upload-image routes (keeps app.py slim).

Upload uses a plain JSON body (base64 image data) rather than a multipart
form, so it works without the optional `python-multipart` dependency that
FastAPI's `File`/`Form`/`UploadFile` require.
"""

from __future__ import annotations

import base64
import binascii

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import products, registration_io

_IMAGE_EXT = ".png"


class CreateProductRequest(BaseModel):
    stage: str
    camera: str
    printer_id: str


class UploadImageRequest(BaseModel):
    stage: str
    camera: str
    printer_id: str
    shot: str
    image_base64: str  # raw base64, or a "data:image/...;base64,<data>" URL


def register_upload_routes(app: FastAPI) -> None:
    @app.post("/api/products")
    def create_product(body: CreateProductRequest) -> dict:
        try:
            product = products.create_product(body.stage, body.camera, body.printer_id)
        except products.InvalidProductError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "ok": True,
            "stage": product.stage,
            "camera": product.camera,
            "printer_id": product.printer_id,
            "information_path": product.information_path,
        }

    @app.post("/api/upload-image")
    def upload_image(body: UploadImageRequest) -> dict:
        if body.shot not in registration_io.SHOT_NAMES:
            raise HTTPException(status_code=400, detail=f"unknown shot: {body.shot!r}")
        try:
            product = products.find_product(body.stage, body.camera, body.printer_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        raw_b64 = body.image_base64.split(",", 1)[-1]
        try:
            raw = base64.b64decode(raw_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid base64 image data: {exc}") from exc

        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="uploaded file is not a readable image")

        try:
            yaml_path = product.registration_yaml()
        except FileNotFoundError:
            yaml_path = product.registration_yaml_target()
        data = registration_io.load_yaml(yaml_path) if yaml_path.exists() else {}
        shot_data = registration_io.ensure_shot(data, body.shot)

        image_path = product.root / "images" / f"{body.shot}{_IMAGE_EXT}"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), img)
        shot_data["image_path"] = f"images/{body.shot}{_IMAGE_EXT}"

        registration_io.save_yaml(yaml_path, data)
        h, w = img.shape[:2]
        return {"ok": True, "image_path": str(image_path), "width": w, "height": h}
