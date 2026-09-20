"""
Disease Distribution and Pathology Occurrence Statistics Module.
Calculates accepted lesion counts, class proportions, and unique affected-image counts.
Explicitly distinguishes detection instance count from unique affected-image count.
"""

from typing import List, Dict, Any
from collections import defaultdict
from .schemas import CLASS_NAMES, ObservationStatus

def calculate_disease_distribution(image_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes disease distribution and unique image infection statistics from image records.

    DISTINCTION MANDATE:
    - total_detections: Total number of accepted bounding boxes.
    - unique_affected_images: Number of distinct images exhibiting at least one accepted detection of that disease.
    """
    valid_records = [
        r for r in image_records 
        if r["observation_status"] not in [ObservationStatus.INVALID_INPUT, ObservationStatus.INFERENCE_ERROR]
    ]
    total_valid_images = len(valid_records)

    # Accumulate detection counts and image sets per class
    detections_per_class = {cname: 0 for cname in CLASS_NAMES.values()}
    images_per_class = {cname: set() for cname in CLASS_NAMES.values()}
    total_accepted_detections = 0

    for r in valid_records:
        img_id = r["image_id"]
        for det in r.get("accepted_detections", []):
            cname = det["disease_class"]
            if cname in detections_per_class:
                detections_per_class[cname] += 1
                images_per_class[cname].add(img_id)
                total_accepted_detections += 1

    distribution_per_class = {}
    for cid in sorted(CLASS_NAMES.keys()):
        cname = CLASS_NAMES[cid]
        det_count = detections_per_class[cname]
        img_count = len(images_per_class[cname])

        # Proportion of all accepted lesion detections
        pct_detections = (
            round((det_count / total_accepted_detections) * 100.0, 2)
            if total_accepted_detections > 0 else 0.0
        )

        # Prevalence among valid sampled images (Observation-level infection rate)
        pct_images = (
            round((img_count / total_valid_images) * 100.0, 2)
            if total_valid_images > 0 else 0.0
        )

        distribution_per_class[cname] = {
            "class_id": cid,
            "disease_class": cname,
            "detection_count": det_count,
            "percentage_of_total_detections": pct_detections,
            "unique_images_affected": img_count,
            "percentage_of_images_affected": pct_images
        }

    return {
        "total_sampled_images": len(image_records),
        "total_successfully_processed_images": total_valid_images,
        "total_accepted_detections": total_accepted_detections,
        "class_distributions": distribution_per_class,
        "methodological_note": (
            "Detection count reflects the total number of localized bounding boxes. "
            "Unique affected images reflects the number of tree photos where the pathology was confirmed. "
            "Because an infected palm can exhibit multiple lesion spots, detection count >= unique affected images."
        )
    }
