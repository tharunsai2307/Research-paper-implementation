"""
Schema constants and status definitions for the Plantation Health Monitoring Engine.
"""

from typing import Dict, Any

CLASS_NAMES: Dict[int, str] = {
    0: "bud root dropping",
    1: "bud rot",
    2: "gray leaf spot",
    3: "leaf rot",
    4: "stembleeding"
}

class ObservationStatus:
    DISEASE_DETECTED = "DISEASE_DETECTED"
    NO_DISEASE_DETECTED = "NO_DISEASE_DETECTED"
    LOW_CONFIDENCE_CANDIDATE = "LOW_CONFIDENCE_CANDIDATE"
    INVALID_INPUT = "INVALID_INPUT"
    INFERENCE_ERROR = "INFERENCE_ERROR"

# Operational Thresholds (Engineering configurations)
# BUG FIX 2026-09-20: Phase 12 threshold sensitivity study (validation set sweep)
# found optimal F1 at conf=0.10 for EXP-002 (deployed model).
# EXP-002 max prediction confidence on training images is ~0.16,
# which is below the previous DEFAULT_OPERATIONAL_THRESHOLD of 0.25.
# Fix: lower operational threshold to 0.10 (validated optimum).
# DEFAULT_CANDIDATE_THRESHOLD lowered to 0.05 to preserve two-tier screening.
# Phase 4 frozen test metrics (mAP50, precision, recall) are NOT affected —
# mAP50 is computed from the full PR curve and is threshold-independent.
DEFAULT_OPERATIONAL_THRESHOLD = 0.10  # was 0.25 — see research/bug_fix_detection_pipeline.md
DEFAULT_CANDIDATE_THRESHOLD = 0.05   # was 0.10 — lower screening floor
DEFAULT_IOU_THRESHOLD = 0.70

def validate_detection_record(det: Dict[str, Any]) -> bool:
    """Validates that an individual detection conforms to standardized contract."""
    required = ["class_id", "disease_class", "confidence", "bounding_box"]
    if not all(k in det for k in required):
        return False
    if det["class_id"] not in CLASS_NAMES:
        return False
    if not (0.0 <= det["confidence"] <= 1.0):
        return False
    bbox = det["bounding_box"]
    for coord in ["x1", "y1", "x2", "y2"]:
        if coord not in bbox or bbox[coord] < 0:
            return False
    if bbox["x2"] < bbox["x1"] or bbox["y2"] < bbox["y1"]:
        return False
    return True
