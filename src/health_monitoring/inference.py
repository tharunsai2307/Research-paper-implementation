"""
Health Monitoring Inference Engine.
Loads trained YOLOv8 checkpoints, runs inference on field images,
classifies observation states, and extracts standardized detection records.
"""

import os
import cv2
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from ultralytics import YOLO

from .schemas import (
    CLASS_NAMES,
    ObservationStatus,
    DEFAULT_OPERATIONAL_THRESHOLD,
    DEFAULT_CANDIDATE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD
)
from .severity_proxy import calculate_relative_affected_area_proxy

class HealthInferenceEngine:
    def __init__(
        self,
        model_path: Union[str, Path],
        imgsz: int = 640,
        operational_threshold: float = DEFAULT_OPERATIONAL_THRESHOLD,
        candidate_threshold: float = DEFAULT_CANDIDATE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        device: str = "cpu"
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {self.model_path}")

        self.imgsz = imgsz
        self.operational_threshold = operational_threshold
        self.candidate_threshold = candidate_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        self.model = YOLO(str(self.model_path))
        self.model_id = self.model_path.parent.parent.name if self.model_path.parent.name == "weights" else self.model_path.stem
        self.checkpoint_size_mb = round(self.model_path.stat().st_size / (1024 * 1024), 2)

    def process_image(
        self,
        image_path: Union[str, Path],
        plantation_id: Optional[str] = None,
        gps_location: Optional[Dict[str, float]] = None,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs standardized health monitoring inference on a single image."""
        img_p = Path(image_path)
        if not img_p.exists():
            return {
                "image_id": img_p.name,
                "image_path": str(img_p),
                "plantation_id": plantation_id,
                "timestamp": timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "gps_location": gps_location,
                "model_id": self.model_id,
                "model_input_size": self.imgsz,
                "image_width": None,
                "image_height": None,
                "observation_status": ObservationStatus.INVALID_INPUT,
                "error_message": "Image file does not exist",
                "accepted_detections": [],
                "candidate_detections": [],
                "relative_affected_area_proxy": 0.0,
                "detected_classes": []
            }

        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            return {
                "image_id": img_p.name,
                "image_path": str(img_p),
                "plantation_id": plantation_id,
                "timestamp": timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "gps_location": gps_location,
                "model_id": self.model_id,
                "model_input_size": self.imgsz,
                "image_width": None,
                "image_height": None,
                "observation_status": ObservationStatus.INVALID_INPUT,
                "error_message": "OpenCV failed to decode image pixel data",
                "accepted_detections": [],
                "candidate_detections": [],
                "relative_affected_area_proxy": 0.0,
                "detected_classes": []
            }

        h_img, w_img = img_bgr.shape[:2]

        try:
            # Predict sweeping down to candidate_threshold to separate confirmed from low-confidence candidates
            pred_res = self.model.predict(
                source=str(img_p),
                imgsz=self.imgsz,
                conf=self.candidate_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )[0]
        except Exception as e:
            return {
                "image_id": img_p.name,
                "image_path": str(img_p),
                "plantation_id": plantation_id,
                "timestamp": timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "gps_location": gps_location,
                "model_id": self.model_id,
                "model_input_size": self.imgsz,
                "image_width": w_img,
                "image_height": h_img,
                "observation_status": ObservationStatus.INFERENCE_ERROR,
                "error_message": str(e),
                "accepted_detections": [],
                "candidate_detections": [],
                "relative_affected_area_proxy": 0.0,
                "detected_classes": []
            }

        boxes = pred_res.boxes
        accepted_detections = []
        candidate_detections = []
        accepted_xyxy = []

        for i in range(len(boxes)):
            cid = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].cpu().numpy()]
            cname = CLASS_NAMES.get(cid, str(cid))

            # Normalized coords
            xc = round(((x1 + x2) / 2.0) / w_img, 6)
            yc = round(((y1 + y2) / 2.0) / h_img, 6)
            w = round((x2 - x1) / w_img, 6)
            h = round((y2 - y1) / h_img, 6)

            det_obj = {
                "class_id": cid,
                "disease_class": cname,
                "confidence": round(conf, 4),
                "bounding_box": {
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1)
                },
                "normalized_box": {
                    "xc": xc,
                    "yc": yc,
                    "w": w,
                    "h": h
                }
            }

            if conf >= self.operational_threshold:
                accepted_detections.append(det_obj)
                accepted_xyxy.append([x1, y1, x2, y2])
            else:
                candidate_detections.append(det_obj)

        # Determine observation status
        if len(accepted_detections) > 0:
            obs_status = ObservationStatus.DISEASE_DETECTED
        elif len(candidate_detections) > 0:
            obs_status = ObservationStatus.LOW_CONFIDENCE_CANDIDATE
        else:
            obs_status = ObservationStatus.NO_DISEASE_DETECTED

        # Calculate Relative Affected Area Proxy using union area (no double-counting)
        affected_area_proxy = calculate_relative_affected_area_proxy(accepted_xyxy, w_img, h_img)

        # Unique detected classes
        detected_classes = sorted(list(set(d["disease_class"] for d in accepted_detections)))

        # Confidence summary
        accepted_confs = [d["confidence"] for d in accepted_detections]
        conf_summary = {
            "mean": round(float(sum(accepted_confs) / len(accepted_confs)), 4) if accepted_confs else 0.0,
            "max": round(float(max(accepted_confs)), 4) if accepted_confs else 0.0,
            "min": round(float(min(accepted_confs)), 4) if accepted_confs else 0.0
        }

        return {
            "image_id": img_p.name,
            "image_path": str(img_p),
            "plantation_id": plantation_id,
            "timestamp": timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gps_location": gps_location,
            "model_id": self.model_id,
            "model_input_size": self.imgsz,
            "image_width": w_img,
            "image_height": h_img,
            "observation_status": obs_status,
            "accepted_detection_count": len(accepted_detections),
            "candidate_detection_count": len(candidate_detections),
            "accepted_detections": accepted_detections,
            "candidate_detections": candidate_detections,
            "detected_classes": detected_classes,
            "relative_affected_area_proxy": affected_area_proxy,
            "confidence_summary": conf_summary
        }

    def process_directory(
        self,
        image_dir: Union[str, Path],
        plantation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Processes all images in a target directory."""
        dir_p = Path(image_dir)
        if not dir_p.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_p}")

        image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
        images = sorted([p for p in dir_p.glob("*.*") if p.suffix.lower() in image_extensions])

        records = []
        for img_p in images:
            rec = self.process_image(img_p, plantation_id=plantation_id)
            records.append(rec)

        return records
