"""
Temporal Feature Engineering and Longitudinal Data Structures.
Implements sequence extraction, specimen-level temporal grouping,
and baseline progression heuristics (LOCF & Linear Slope).
"""

from typing import List, Dict, Any, Optional, Tuple
import math
from datetime import datetime

CLASS_TO_ID = {
    "bud root dropping": 0,
    "bud rot": 1,
    "gray leaf spot": 2,
    "leaf rot": 3,
    "stembleeding": 4
}

class PalmTemporalObservation:
    """Represents a single observation visit of an individual palm."""
    def __init__(
        self,
        observation_id: str,
        tree_id: str,
        timestamp_str: str,
        severity_ratio: float,
        primary_disease: Optional[str] = None,
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.observation_id = str(observation_id)
        self.tree_id = str(tree_id)
        self.timestamp_str = str(timestamp_str)
        self.dt = self._parse_iso(timestamp_str)
        self.severity_ratio = min(1.0, max(0.0, float(severity_ratio)))
        self.primary_disease = str(primary_disease).lower() if primary_disease else None
        self.confidence = min(1.0, max(0.0, float(confidence)))
        self.metadata = metadata or {}

    @staticmethod
    def _parse_iso(ts: str) -> datetime:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")

class TemporalSequenceExtractor:
    """Extracts leakage-free, chronologically sorted feature matrices from palm observation logs."""

    @staticmethod
    def sort_observations(observations: List[PalmTemporalObservation]) -> List[PalmTemporalObservation]:
        """Ensures strictly monotonic chronological ordering."""
        return sorted(observations, key=lambda x: x.dt)

    @staticmethod
    def construct_feature_matrix(observations: List[PalmTemporalObservation]) -> List[List[float]]:
        """
        Builds feature matrix where each row represents:
        [S_k, delta_S_k, growth_rate, delta_days, confidence, d_0, d_1, d_2, d_3, d_4]
        Dimension D = 10.
        """
        sorted_obs = TemporalSequenceExtractor.sort_observations(observations)
        features = []
        prev_sev = None
        prev_dt = None

        for obs in sorted_obs:
            sev = obs.severity_ratio
            if prev_sev is None:
                delta_sev = 0.0
                delta_days = 0.0
                growth_rate = 0.0
            else:
                delta_sev = sev - prev_sev
                delta_days = max(0.0, (obs.dt - prev_dt).total_seconds() / 86400.0)
                growth_rate = delta_sev / max(delta_days, 1.0)

            # Disease one-hot encoding
            d_vec = [0.0] * 5
            if obs.primary_disease and obs.primary_disease in CLASS_TO_ID:
                d_vec[CLASS_TO_ID[obs.primary_disease]] = 1.0

            row = [
                round(sev, 6),
                round(delta_sev, 6),
                round(growth_rate, 6),
                round(delta_days, 2),
                round(obs.confidence, 4),
                *d_vec
            ]
            features.append(row)
            prev_sev = sev
            prev_dt = obs.dt

        return features

class BaselineProgressionForecaster:
    """
    Implements mandatory non-neural baselines:
    1. Last Observation Carried Forward (LOCF)
    2. Empirical Linear Growth Rate Projection
    """

    @staticmethod
    def forecast_locf(history: List[PalmTemporalObservation], horizon_days: int = 7) -> Dict[str, Any]:
        """Forecasts future severity using LOCF persistence."""
        if not history:
            raise ValueError("History cannot be empty for LOCF forecast.")
        sorted_obs = TemporalSequenceExtractor.sort_observations(history)
        current_obs = sorted_obs[-1]
        return {
            "baseline_model": "Last_Observation_Carried_Forward_LOCF",
            "horizon_days": horizon_days,
            "current_severity": current_obs.severity_ratio,
            "predicted_severity": current_obs.severity_ratio,
            "projected_severity_change": 0.0,
            "trajectory_status": "STABLE"
        }

    @staticmethod
    def forecast_linear(history: List[PalmTemporalObservation], horizon_days: int = 7) -> Dict[str, Any]:
        """Forecasts future severity using instantaneous linear velocity projection."""
        if len(history) < 2:
            return BaselineProgressionForecaster.forecast_locf(history, horizon_days)
        
        sorted_obs = TemporalSequenceExtractor.sort_observations(history)
        c_obs = sorted_obs[-1]
        p_obs = sorted_obs[-2]
        
        delta_days = max(0.01, (c_obs.dt - p_obs.dt).total_seconds() / 86400.0)
        daily_velocity = (c_obs.severity_ratio - p_obs.severity_ratio) / delta_days
        
        projected_change = daily_velocity * horizon_days
        predicted_sev = min(1.0, max(0.0, c_obs.severity_ratio + projected_change))

        if projected_change > 0.03:
            traj = "RAPID_EXPANSION"
        elif projected_change > 0.005:
            traj = "SLOW_EXPANSION"
        else:
            traj = "STABLE"

        return {
            "baseline_model": "Empirical_Linear_Slope_Projection",
            "horizon_days": horizon_days,
            "current_severity": round(c_obs.severity_ratio, 6),
            "predicted_severity": round(predicted_sev, 6),
            "projected_severity_change": round(projected_change, 6),
            "trajectory_status": traj
        }
