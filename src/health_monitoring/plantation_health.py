"""
Plantation Health Monitoring Aggregation and Index Engine.
Aggregates observation records into plantation-level indicators,
computes observation rates, disease distribution, and the Plantation Health Index (PHI).
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .schemas import ObservationStatus
from .disease_distribution import calculate_disease_distribution

class PlantationHealthAggregator:
    def __init__(
        self,
        weight_incidence: float = 0.85,
        weight_area_proxy: float = 0.15
    ):
        """
        Initializes aggregator with explicit, documented weighting parameters:
        - weight_incidence (w1): Influence of disease-positive tree proportion (default: 0.85)
        - weight_area_proxy (w2): Influence of mean visual affected area proxy (default: 0.15)
        Note: w1 + w2 = 1.0. Unweighted incidence-only index is also computed in parallel.
        """
        self.w1 = weight_incidence
        self.w2 = weight_area_proxy

    def generate_plantation_report(
        self,
        image_records: List[Dict[str, Any]],
        plantation_id: Optional[str] = "PLANTATION_BLOCK_01",
        observation_session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Aggregates image-level records into a structured plantation health assessment."""
        total_obs = len(image_records)
        successful_obs = [
            r for r in image_records
            if r["observation_status"] not in [ObservationStatus.INVALID_INPUT, ObservationStatus.INFERENCE_ERROR]
        ]
        num_successful = len(successful_obs)
        num_failed = total_obs - num_successful

        disease_pos_obs = [r for r in successful_obs if r["observation_status"] == ObservationStatus.DISEASE_DETECTED]
        disease_neg_obs = [r for r in successful_obs if r["observation_status"] == ObservationStatus.NO_DISEASE_DETECTED]
        low_conf_obs = [r for r in successful_obs if r["observation_status"] == ObservationStatus.LOW_CONFIDENCE_CANDIDATE]

        num_disease_pos = len(disease_pos_obs)
        num_disease_neg = len(disease_neg_obs)
        num_low_conf = len(low_conf_obs)

        # Observation-level rates (Observation-level statistics, NOT regional epidemiological prevalence)
        pos_rate = round((num_disease_pos / num_successful) * 100.0, 2) if num_successful > 0 else 0.0
        neg_rate = round((num_disease_neg / num_successful) * 100.0, 2) if num_successful > 0 else 0.0
        low_conf_rate = round((num_low_conf / num_successful) * 100.0, 2) if num_successful > 0 else 0.0

        # Disease distribution
        dist_report = calculate_disease_distribution(image_records)

        # Impact Indicators
        box_counts = [r["accepted_detection_count"] for r in successful_obs]
        area_proxies_all = [r["relative_affected_area_proxy"] for r in successful_obs]
        area_proxies_pos = [r["relative_affected_area_proxy"] for r in disease_pos_obs]

        mean_boxes = round(float(np.mean(box_counts)), 2) if box_counts else 0.0
        median_boxes = round(float(np.median(box_counts)), 2) if box_counts else 0.0
        max_boxes = int(max(box_counts)) if box_counts else 0

        mean_area_all = round(float(np.mean(area_proxies_all)), 4) if area_proxies_all else 0.0
        mean_area_pos = round(float(np.mean(area_proxies_pos)), 4) if area_proxies_pos else 0.0
        max_area = round(float(max(area_proxies_all)), 4) if area_proxies_all else 0.0

        # All accepted confidences
        all_confs = []
        for r in disease_pos_obs:
            for det in r.get("accepted_detections", []):
                all_confs.append(det["confidence"])

        conf_stats = {
            "mean": round(float(np.mean(all_confs)), 4) if all_confs else 0.0,
            "median": round(float(np.median(all_confs)), 4) if all_confs else 0.0,
            "min": round(float(min(all_confs)), 4) if all_confs else 0.0,
            "max": round(float(max(all_confs)), 4) if all_confs else 0.0
        }

        # Plantation Health Index (PHI) Calculation
        # Formulation 1: Transparent Incidence-Only PHI
        # PHI_incidence = 100 * (1 - num_disease_pos / num_successful)
        phi_incidence = round(100.0 * (1.0 - (num_disease_pos / num_successful)), 2) if num_successful > 0 else 100.0

        # Formulation 2: Composite PHI (Incidence + Visual Area Proxy)
        # PHI_composite = 100 * max(0.0, 1.0 - (w1 * (num_disease_pos / num_successful) + w2 * mean_area_all))
        raw_penalty = (self.w1 * (num_disease_pos / num_successful)) + (self.w2 * mean_area_all) if num_successful > 0 else 0.0
        phi_composite = round(max(0.0, min(100.0, 100.0 * (1.0 - raw_penalty))), 2)

        # Categorical Health Tier
        if phi_composite >= 85.0:
            health_tier = "EXCELLENT / MINIMAL_DISEASE_PRESSURE"
            recommendation = "Maintain routine prophylactic monitoring. No immediate systemic treatment required."
        elif phi_composite >= 70.0:
            health_tier = "GOOD / LOCALIZED_TREATMENT_RECOMMENDED"
            recommendation = "Deploy targeted agronomic inspection to positive trees; spot-treat lesions."
        elif phi_composite >= 50.0:
            health_tier = "MODERATE / ACTIVE_OUTBREAK_MONITORING"
            recommendation = "Initiate sector quarantine and scheduled canopy/trunk therapeutic intervention."
        else:
            health_tier = "POOR / SYSTEMIC_INTERVENTION_REQUIRED"
            recommendation = "Severe epidemic pressure observed. Immediate agricultural extension intervention mandated."

        return {
            "meta": {
                "plantation_id": plantation_id,
                "observation_session_id": observation_session_id or f"SESSION_{int(np.random.randint(1000, 9999))}",
                "engine_version": "Phase-5-v1.0.0",
                "operational_confidence_threshold": image_records[0].get("operational_threshold", 0.25) if image_records else 0.25
            },
            "observation_counts": {
                "total_images_submitted": total_obs,
                "successfully_processed_images": num_successful,
                "processing_failures": num_failed,
                "disease_positive_images": num_disease_pos,
                "disease_negative_images": num_disease_neg,
                "low_confidence_candidate_images": num_low_conf
            },
            "observation_rates": {
                "disease_positive_image_percentage": pos_rate,
                "disease_negative_image_percentage": neg_rate,
                "low_confidence_image_percentage": low_conf_rate,
                "rate_definition": "Percentage of successfully decoded observation photos showing confirmed target disease."
            },
            "impact_indicators": {
                "mean_boxes_per_image": mean_boxes,
                "median_boxes_per_image": median_boxes,
                "max_boxes_in_single_image": max_boxes,
                "mean_relative_affected_area_all_images": mean_area_all,
                "mean_relative_affected_area_positive_images": mean_area_pos,
                "max_relative_affected_area": max_area,
                "confidence_distribution": conf_stats
            },
            "disease_distribution": dist_report["class_distributions"],
            "plantation_health_index": {
                "phi_composite_score": phi_composite,
                "phi_incidence_only_score": phi_incidence,
                "health_tier": health_tier,
                "formula_used": "PHI_composite = 100 * (1.0 - (w1 * (N_pos / N_success) + w2 * Mean_Affected_Area_Proxy))",
                "weights": {
                    "w1_incidence": self.w1,
                    "w2_area_proxy": self.w2
                },
                "recommended_action": recommendation
            },
            "methodological_assumptions_and_limitations": [
                "Observation-level rates reflect submitted photo samples and do NOT represent formal epidemiological prevalence across a whole plantation block without random grid sampling.",
                "Relative Affected Area Proxy measures 2D visible bounding box footprint in the photograph and does not replace in-situ plant pathology tissue staging.",
                "Weights w1=0.85 and w2=0.15 are engineered methodological assumptions prioritizing tree infection incidence over photographic frame coverage."
            ]
        }
