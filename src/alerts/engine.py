"""
Phase 7 Automated Agricultural Progression Alert Engine.
Translates current health metrics, historical observation sequences, and
projected severity into actionable monitoring and intervention tiers.
"""

from typing import Dict, Any, List, Optional
from enum import Enum

class AlertLevel(str, Enum):
    NO_ALERT = "NO_ALERT"
    MONITOR = "MONITOR"
    ATTENTION = "ATTENTION"
    URGENT_REVIEW = "URGENT_REVIEW"

class ProgressionTrajectory(str, Enum):
    STABLE = "STABLE"
    SLOW_EXPANSION = "SLOW_EXPANSION"
    RAPID_EXPANSION = "RAPID_EXPANSION"
    INDETERMINATE = "INDETERMINATE"

class ProgressionAlertEngine:
    """
    Evaluates multi-visit observation history and computes alert levels
    strictly as an engineering decision-support tool.
    """

    def __init__(
        self,
        stable_threshold: float = 0.005,
        rapid_threshold: float = 0.030,
        attention_severity: float = 0.05,
        urgent_severity: float = 0.15
    ):
        self.stable_threshold = stable_threshold
        self.rapid_threshold = rapid_threshold
        self.attention_severity = attention_severity
        self.urgent_severity = urgent_severity

    def evaluate_progression(
        self,
        current_severity: float,
        projected_severity: float,
        primary_disease: Optional[str] = None,
        phi_score: float = 100.0
    ) -> Dict[str, Any]:
        """
        Determines progression status and assigns action alert based on deterministic rules.
        """
        curr = min(1.0, max(0.0, float(current_severity)))
        proj = min(1.0, max(0.0, float(projected_severity)))
        delta_proj = proj - curr
        disease = str(primary_disease).lower() if primary_disease else None

        # 1. Trajectory Categorization
        if delta_proj > self.rapid_threshold:
            trajectory = ProgressionTrajectory.RAPID_EXPANSION
        elif delta_proj > self.stable_threshold:
            trajectory = ProgressionTrajectory.SLOW_EXPANSION
        else:
            trajectory = ProgressionTrajectory.STABLE

        # 2. Alert Assignment Logic
        if curr == 0.0 and proj == 0.0:
            alert = AlertLevel.NO_ALERT
            rec = "Palm exhibits no active disease symptoms. Continue routine scheduled field scouting."
        elif disease == "bud rot" and (curr > 0.02 or trajectory != ProgressionTrajectory.STABLE):
            # Bud rot is a lethal meristem disease requiring elevated urgency
            alert = AlertLevel.URGENT_REVIEW
            rec = "Lethal bud rot detected with active progression. Urgent phytosanitary inspection required to preserve palm crown."
        elif curr > self.urgent_severity or proj > self.urgent_severity:
            alert = AlertLevel.URGENT_REVIEW
            rec = "High lesion area proxy detected. Systemic intervention recommended to prevent canopy collapse."
        elif curr > self.attention_severity or trajectory == ProgressionTrajectory.RAPID_EXPANSION:
            alert = AlertLevel.ATTENTION
            rec = "Active lesion expansion detected. Conduct ground agronomic check and consider localized fungicide treatment."
        else:
            alert = AlertLevel.MONITOR
            rec = "Localized minimal lesion area. Maintain prophylactic monitoring and re-inspect palm in next scouting cycle."

        return {
            "current_severity_proxy": round(curr, 6),
            "projected_severity_proxy": round(proj, 6),
            "projected_severity_delta": round(delta_proj, 6),
            "progression_trajectory": trajectory.value,
            "alert_level": alert.value,
            "primary_disease": disease,
            "phi_score": round(phi_score, 2),
            "recommended_action": rec,
            "disclaimer": (
                "Notice: Progression alerts represent heuristic engineering decision-support indicators. "
                "They do NOT constitute certified agronomic prescriptions or plant pathology diagnoses."
            )
        }
