"""
FastAPI Backend Application for Mobile Coconut Disease Detection & Health Monitoring.

Exposes endpoints:
- GET  /api/v1/health     : Health check, active model metadata, thresholds.
- POST /api/v1/analyze    : Runs YOLOv8 inference & Phase 5 health analysis on an uploaded palm image.
- POST /api/v1/aggregate  : Aggregates multiple image records into a Plantation Health Report.
- GET  /                  : Serves the responsive mobile application interface.
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.health_monitoring.inference import HealthInferenceEngine
from src.health_monitoring.plantation_health import PlantationHealthAggregator
from src.health_monitoring.schemas import (
    CLASS_NAMES,
    ObservationStatus,
    DEFAULT_OPERATIONAL_THRESHOLD,
    DEFAULT_CANDIDATE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD
)
from src.segmentation.interface import FallbackBoundingBoxSegmentationProxy
from src.alerts.engine import ProgressionAlertEngine

# Initialize Phase 7 Extensions
segmentation_proxy = FallbackBoundingBoxSegmentationProxy(operational_threshold=DEFAULT_OPERATIONAL_THRESHOLD)
alert_engine = ProgressionAlertEngine()

# App Configuration
APP_DIR = PROJECT_ROOT / "app"
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Default model is EXP-002 (512x512) due to superior mobile precision (0.9044) and 32% lower latency (74.56ms)
MODEL_CHECKPOINT = PROJECT_ROOT / "outputs/training/EXP-002_imgsz512/weights/best.pt"
if not MODEL_CHECKPOINT.exists():
    # Fallback to EXP-001 if EXP-002 is unavailable
    MODEL_CHECKPOINT = PROJECT_ROOT / "outputs/training/EXP-001_baseline_yolov8n/weights/best.pt"

# Initialize Health Inference Engine
inference_engine = HealthInferenceEngine(
    model_path=MODEL_CHECKPOINT,
    imgsz=512 if "imgsz512" in str(MODEL_CHECKPOINT) else 640,
    operational_threshold=DEFAULT_OPERATIONAL_THRESHOLD,
    candidate_threshold=DEFAULT_CANDIDATE_THRESHOLD,
    iou_threshold=DEFAULT_IOU_THRESHOLD,
    device="cpu"
)

# Initialize Plantation Aggregator
plantation_aggregator = PlantationHealthAggregator(
    weight_incidence=0.85,
    weight_area_proxy=0.15
)

app = FastAPI(
    title="Coconut Tree Disease Detection & Plantation Health API",
    description="Backend inference and health monitoring API for mobile field applications.",
    version="1.0.0"
)

# Enable CORS for mobile clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

class AggregationRequest(BaseModel):
    plantation_id: Optional[str] = "PLANTATION_BLOCK_01"
    session_id: Optional[str] = None
    records: List[Dict[str, Any]]

@app.get("/api/v1/health")
def api_health():
    """Returns API readiness, active model metadata, and configured thresholds."""
    return {
        "status": "HEALTHY",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": inference_engine.model_id,
        "input_resolution": inference_engine.imgsz,
        "operational_confidence_threshold": inference_engine.operational_threshold,
        "candidate_confidence_threshold": inference_engine.candidate_threshold,
        "iou_threshold": inference_engine.iou_threshold,
        "classes": CLASS_NAMES,
        "hardware_device": inference_engine.device
    }

@app.post("/api/v1/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    plantation_id: Optional[str] = Form("FIELD_BLOCK_01"),
    gps_lat: Optional[float] = Form(None),
    gps_lng: Optional[float] = Form(None)
):
    """
    Receives image upload, decodes pixel buffer safely, executes YOLOv8 inference,
    and returns standardized detection records with Phase 5 health proxy calculations.
    """
    # 1. Validate file format
    valid_content_types = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    if file.content_type and file.content_type.lower() not in valid_content_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Please upload a JPEG or PNG image."
        )

    # 2. Read file bytes into memory
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image stream: {e}")

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty image file received.")

    # Size limit (max 15MB)
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image file exceeds maximum allowable size (15 MB).")

    # 3. Decode image buffer with OpenCV
    np_arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=422, detail="OpenCV failed to decode image pixel data. File may be corrupted.")

    # 4. Save to temporary scratch path for inference engine processing
    temp_dir = PROJECT_ROOT / "outputs/phase_6/temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / f"mobile_upload_{int(time.time() * 1000)}_{file.filename}"

    try:
        cv2.imwrite(str(temp_file), img_bgr)

        gps_dict = None
        if gps_lat is not None and gps_lng is not None:
            gps_dict = {"latitude": float(gps_lat), "longitude": float(gps_lng)}

        t_start = time.perf_counter()
        record = inference_engine.process_image(
            image_path=temp_file,
            plantation_id=plantation_id,
            gps_location=gps_dict,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        # Append single-image health index & tier computation for immediate feedback
        single_report = plantation_aggregator.generate_plantation_report([record], plantation_id=plantation_id)
        phi_info = single_report["plantation_health_index"]

        # Attach Phase 7 fine-grained segmentation status
        seg_result = segmentation_proxy.segment_image(
            image=img_bgr,
            detections=record.get("accepted_detections", [])
        )
        record["segmentation_analysis"] = seg_result

        # Attach Phase 7 monitoring alert evaluation
        primary_disease = record["detected_classes"][0] if record.get("detected_classes") else None
        current_sev = record.get("relative_affected_area_proxy", 0.0)
        # Using Phase 7 LOCF / conservative baseline for single-visit inspection
        alert_result = alert_engine.evaluate_progression(
            current_severity=current_sev,
            projected_severity=current_sev, # LOCF baseline
            primary_disease=primary_disease,
            phi_score=phi_info["phi_composite_score"]
        )
        record["progression_alert"] = alert_result

        record["health_assessment"] = {
            "phi_score": phi_info["phi_composite_score"],
            "health_tier": phi_info["health_tier"],
            "recommended_action": alert_result["recommended_action"],
            "disclaimer": (
                "Notice: Health scores and affected area proxies represent 2D image-space visual indicators "
                "computed by the research engine and do NOT constitute certified agronomic or clinical pathology diagnosis."
            )
        }
        record["inference_latency_ms"] = latency_ms

        return JSONResponse(content=record)

    finally:
        # Clean up temporary upload
        if temp_file.exists():
            try:
                os.remove(temp_file)
            except Exception:
                pass

@app.post("/api/v1/aggregate")
def aggregate_session(request: AggregationRequest):
    """
    Aggregates multiple tree observation records collected during a scouting walk
    into a structured Plantation Health Report using the Phase 5 engine.
    """
    if not request.records:
        raise HTTPException(status_code=400, detail="No observation records provided for aggregation.")

    report = plantation_aggregator.generate_plantation_report(
        image_records=request.records,
        plantation_id=request.plantation_id,
        observation_session_id=request.session_id or f"SCOUT_{int(time.time())}"
    )
    return JSONResponse(content=report)

@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serves the responsive mobile application user interface."""
    index_html_path = TEMPLATES_DIR / "index.html"
    if not index_html_path.exists():
        return HTMLResponse("<h1>Coconut Disease Mobile App</h1><p>Template loading...</p>", status_code=200)
    with open(index_html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)
