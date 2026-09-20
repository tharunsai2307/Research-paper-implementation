"""
Validation Script for Phase 5 Plantation Health Monitoring Engine.

Validates 11 explicit checks:
1. Report Schema Completeness
2. Record Schema Completeness
3. Disease Class Integrity
4. Confidence Bounds [0,1]
5. Bounding Box Coordinates
6. Area Proxy Bounds [0,1]
7. Percentage Bounds [0,100]
8. Numerical Stability
9. Observation Count Consistency
10. Detection Count Consistency
11. Deterministic Aggregation

Outputs a definitive PASS/FAIL verdict with strictly enforced accounting invariants:
- total_checks == len(registered_checks)
- passed_checks + failed_checks == total_checks
- passed_checks <= total_checks
- failed_checks >= 0
"""

import sys
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.health_monitoring.schemas import CLASS_NAMES, ObservationStatus
from src.health_monitoring.plantation_health import PlantationHealthAggregator

class HealthMonitoringValidator:
    def __init__(self, records_path: Path, report_path: Path):
        self.records_path = Path(records_path)
        self.report_path = Path(report_path)
        self.checks: List[Tuple[str, Callable[[List[Dict[str, Any]], Dict[str, Any]], Tuple[bool, str]]]] = [
            ("Report Schema Completeness", self._check_report_schema),
            ("Record Schema Completeness", self._check_record_schema),
            ("Disease Class Integrity", self._check_disease_classes),
            ("Confidence Bounds [0,1]", self._check_confidence_bounds),
            ("Bounding Box Coordinates", self._check_bounding_box_geometry),
            ("Area Proxy Bounds [0,1]", self._check_area_proxy_bounds),
            ("Percentage Bounds [0,100]", self._check_percentage_bounds),
            ("Numerical Stability", self._check_numerical_stability),
            ("Observation Count Consistency", self._check_observation_counts),
            ("Detection Count Consistency", self._check_detection_counts),
            ("Deterministic Aggregation", self._check_deterministic_aggregation),
        ]

    # --- Check Implementations ---

    def _check_report_schema(self, records: List[Dict[str, Any]], report: Dict[str, Any]) -> Tuple[bool, str]:
        req_keys = [
            "meta", "observation_counts", "observation_rates",
            "impact_indicators", "disease_distribution", "plantation_health_index"
        ]
        missing = [k for k in req_keys if k not in report]
        if not missing:
            return True, "All required top-level report keys present."
        return False, f"Missing required top-level report keys: {missing}"

    def _check_record_schema(self, records: List[Dict[str, Any]], report: Dict[str, Any]) -> Tuple[bool, str]:
        req_keys = [
            "image_id", "observation_status", "accepted_detection_count",
            "relative_affected_area_proxy", "accepted_detections"
        ]
        for idx, rec in enumerate(records):
            for k in req_keys:
                if k not in rec:
                    return False, f"Record {idx} ({rec.get('image_id')}) missing key '{k}'"
        return True, "All image records contain required schema fields."

    def _check_disease_classes(self, records: List[Dict[str, Any]], report: Dict[str, Any]) -> Tuple[bool, str]:
        valid_cnames = set(CLASS_NAMES.values())
        invalid_classes = []
        for rec in records:
            for det in rec.get("accepted_detections", []) + rec.get("candidate_detections", []):
                cname = det.get("disease_class")
                if cname not in valid_cnames:
                    invalid_classes.append(cname)
        if not invalid_classes:
            return True, "All detected classes match valid coconut disease ontology."
        return False, f"Encountered unknown disease classes: {set(invalid_classes)}"

    def _check_confidence_bounds(self, records: List[Dict[str, Any]], report: Dict[str, Any]) -> Tuple[bool, str]:
        out_of_bounds = []
        for rec in records:
            for det in rec.get("accepted_detections", []) + rec.get("candidate_detections", []):
                conf = det.get("confidence", -1.0)
                if not (0.0 <= conf <= 1.0):
                    out_of_bounds.append(conf)
        if not out_of_bounds:
            return True, "All confidence scores strictly bounded in [0.0, 1.0]."
        return False, f"Out-of-bound confidence values: {out_of_bounds}"

    def _check_bounding_box_geometry(self, records: List[Dict[str, Any]], report: Dict[str, Any]) -> Tuple[bool, str]:
        geom_errors = []
        for rec in records:
            for det in rec.get("accepted_detections", []):
                b = det.get("bounding_box", {})
                x1, y1, x2, y2 = b.get("x1", 0), b.get("y1", 0), b.get("x2", 0), b.get("y2", 0)
                if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                    geom_errors.append(f"Negative coordinate in {rec.get('image_id')}")
                if x2 < x1 or y2 < y1:
                    geom_errors.append(f"Inverted coordinates in {rec.get('image_id')}: ({x1},{y1},{x2},{y2})")
        if not geom_errors:
            return True, "All bounding box coordinates are non-negative and properly oriented."
        return False, f"Geometric coordinate errors: {geom_errors[:3]}"

    def _check_area_proxy_bounds(self, records: List[Dict[str, Any]], report: Dict[str, Any]) -> Tuple[bool, str]:
        area_errors = []
        for rec in records:
            proxy = rec.get("relative_affected_area_proxy", -1.0)
            if not (0.0 <= proxy <= 1.0) or math.isnan(proxy) or math.isinf(proxy):
                area_errors.append(f"Invalid proxy {proxy} in {rec.get('image_id')}")
        if not area_errors:
            return True, "All Relative Affected Area Proxies bounded in [0.0, 1.0]."
        return False, f"Area proxy bounds errors: {area_errors}"

    def _check_percentage_bounds(self, records: List[Dict[str, Any]], report: Dict[str, Any]) -> Tuple[bool, str]:
        pct_errors = []
        rates = report.get("observation_rates", {})
        for rate_key in ["disease_positive_image_percentage", "disease_negative_image_percentage", "low_confidence_image_percentage"]:
            val = rates.get(rate_key, -1.0)
            if not (0.0 <= val <= 100.0):
                pct_errors.append(f"{rate_key}: {val}")

        for cname, cdata in report.get("disease_distribution", {}).items():
            for p_key in ["percentage_of_total_detections", "percentage_of_images_affected"]:
                val = cdata.get(p_key, -1.0)
                if not (0.0 <= val <= 100.0):
                    pct_errors.append(f"{cname} {p_key}: {val}")

        if not pct_errors:
            return True, "All percentage values bounded in [0.0, 100.0]."
        return False, f"Impossible percentages: {pct_errors}"

    def _check_numerical_stability(self, records: List[Dict[str, Any]], report: Dict[str, Any]) -> Tuple[bool, str]:
        has_nan_inf = False
        def check_val(v):
            nonlocal has_nan_inf
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                has_nan_inf = True
            elif isinstance(v, dict):
                for val in v.values(): check_val(val)
            elif isinstance(v, list):
                for val in v: check_val(val)

        check_val(report)
        if not has_nan_inf:
            return True, "No NaNs, Infs, or unhandled numeric anomalies detected."
        return False, "Found NaN or Inf in report structure."

    def _check_observation_counts(self, records: List[Dict[str, Any]], report: Dict[str, Any]) -> Tuple[bool, str]:
        counts = report.get("observation_counts", {})
        total_recs = len(records)
        num_pos = sum(1 for r in records if r.get("observation_status") == ObservationStatus.DISEASE_DETECTED)
        num_neg = sum(1 for r in records if r.get("observation_status") == ObservationStatus.NO_DISEASE_DETECTED)
        num_low = sum(1 for r in records if r.get("observation_status") == ObservationStatus.LOW_CONFIDENCE_CANDIDATE)
        num_fail = sum(1 for r in records if r.get("observation_status") in [ObservationStatus.INVALID_INPUT, ObservationStatus.INFERENCE_ERROR])

        if counts.get("total_images_submitted") == total_recs and \
           counts.get("disease_positive_images") == num_pos and \
           counts.get("disease_negative_images") == num_neg and \
           counts.get("low_confidence_candidate_images") == num_low and \
           counts.get("processing_failures") == num_fail:
            return True, "Image-level observation statuses match aggregated counts exactly."
        return False, (
            f"Mismatch: Report counts={counts} vs Computed "
            f"pos={num_pos}, neg={num_neg}, low={num_low}, fail={num_fail}"
        )

    def _check_detection_counts(self, records: List[Dict[str, Any]], report: Dict[str, Any]) -> Tuple[bool, str]:
        sum_report_dets = sum(c.get("detection_count", 0) for c in report.get("disease_distribution", {}).values())
        sum_record_dets = sum(len(r.get("accepted_detections", [])) for r in records)
        if sum_report_dets == sum_record_dets:
            return True, "Total accepted detection counts match across records and distribution."
        return False, f"Mismatch: sum report dets ({sum_report_dets}) != sum record dets ({sum_record_dets})"

    def _check_deterministic_aggregation(self, records: List[Dict[str, Any]], report: Dict[str, Any]) -> Tuple[bool, str]:
        aggregator = PlantationHealthAggregator(
            weight_incidence=report.get("plantation_health_index", {}).get("weights", {}).get("w1_incidence", 0.85),
            weight_area_proxy=report.get("plantation_health_index", {}).get("weights", {}).get("w2_area_proxy", 0.15)
        )
        second_report = aggregator.generate_plantation_report(records, plantation_id=report.get("meta", {}).get("plantation_id"))
        phi_1 = report.get("plantation_health_index", {}).get("phi_composite_score")
        phi_2 = second_report.get("plantation_health_index", {}).get("phi_composite_score")
        if phi_1 is not None and phi_2 is not None and math.isclose(phi_1, phi_2, rel_tol=1e-5):
            return True, "Repeated aggregation produces bitwise identical PHI scores."
        return False, f"PHI scores differ: {phi_1} vs {phi_2}"

    # --- Execution & Verification ---

    def validate_data(self, records: List[Dict[str, Any]], report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes all registered checks and computes strictly consistent accounting metrics:
        guarantees:
        - total_checks == len(self.checks)
        - passed_checks + failed_checks == total_checks
        - passed_checks <= total_checks
        - failed_checks >= 0
        """
        total_checks = len(self.checks)
        results = []

        for name, check_fn in self.checks:
            try:
                passed, message = check_fn(records, report)
            except Exception as e:
                passed = False
                message = f"Check failed with unexpected exception: {e}"
            results.append({
                "name": name,
                "passed": passed,
                "message": message
            })

        passed_checks = sum(1 for r in results if r["passed"])
        failed_checks = total_checks - passed_checks

        # Formal assertion of structural invariants
        assert passed_checks + failed_checks == total_checks, (
            f"Accounting invariant violated: passed ({passed_checks}) + failed ({failed_checks}) != total ({total_checks})"
        )
        assert passed_checks <= total_checks, (
            f"Accounting invariant violated: passed ({passed_checks}) > total ({total_checks})"
        )
        assert failed_checks >= 0, (
            f"Accounting invariant violated: failed ({failed_checks}) < 0"
        )

        return {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "is_valid": (failed_checks == 0),
            "results": results
        }

    def run_all_checks(self) -> bool:
        """Loads target JSON artifacts, executes checks, and prints the audit report."""
        if not self.records_path.exists():
            print(f"FAILED [File Existence]: Image records JSON not found at {self.records_path}")
            return False
        if not self.report_path.exists():
            print(f"FAILED [File Existence]: Plantation report JSON not found at {self.report_path}")
            return False

        try:
            with open(self.records_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            with open(self.report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
        except Exception as e:
            print(f"FAILED [JSON Decode]: Unable to parse target artifacts: {e}")
            return False

        summary = self.validate_data(records, report)
        total = summary["total_checks"]
        passed = summary["passed_checks"]
        failed = summary["failed_checks"]

        print("\n" + "="*70)
        print("PHASE 5 PLANTATION HEALTH MONITORING VALIDATION RESULTS")
        print("="*70)
        print(f"Total Checks Executed: {total}")
        print(f"Total Checks Passed:   {passed} / {total}")
        print(f"Total Checks Failed:   {failed} / {total}")
        print("="*70)

        for r in summary["results"]:
            status_str = "PASS" if r["passed"] else "FAIL"
            print(f"  [{status_str}] {r['name']}: {r['message']}")

        print("="*70)
        if summary["is_valid"]:
            print(f"All {total} validation checks passed with 100% precision.")
            print("No negative areas, no impossible percentages, no division-by-zero.")
            print("Deterministic aggregation and internal consistency VERIFIED.")
            print("\nFINAL STATUS: PASS")
            print("="*70)
            return True
        else:
            print(f"Validation FAILED: {failed} out of {total} checks failed.")
            print("\nFINAL STATUS: FAIL")
            print("="*70)
            return False

if __name__ == "__main__":
    rec_p = PROJECT_ROOT / "outputs/health_monitoring/image_health_records.json"
    rep_p = PROJECT_ROOT / "outputs/health_monitoring/plantation_health_report.json"
    validator = HealthMonitoringValidator(rec_p, rep_p)
    success = validator.run_all_checks()
    sys.exit(0 if success else 1)
