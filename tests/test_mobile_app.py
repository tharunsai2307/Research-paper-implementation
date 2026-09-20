"""
Phase 6 Mobile & API Integration Test Suite
Evaluates FastAPI backend endpoints, model serving integration,
Phase 5 health monitoring contracts, error handling, and scouting session aggregation.
"""

import os
import unittest
from pathlib import Path
import numpy as np
import cv2
from fastapi.testclient import TestClient

from app.api.main import app

class TestMobileAppAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.val_image_path = Path("data/processed/coconut_detection_clean/val/images/LeafRot007.jpg")
        
        # Synthetic plain neutral image for control testing (zero detections)
        img = np.full((512, 512, 3), 128, dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", img)
        cls.synthetic_image_bytes = encoded.tobytes()

    def test_01_health_endpoint(self):
        """Test GET /api/v1/health returns operational status and model specs."""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "HEALTHY")
        self.assertEqual(data["model_id"], "EXP-002_imgsz512")
        # BUG FIX 2026-09-20: operational threshold corrected from 0.25 to 0.10
        # per Phase 12 threshold sensitivity study (see research/bug_fix_detection_pipeline.md)
        self.assertEqual(data["operational_confidence_threshold"], 0.10)
        self.assertEqual(data["candidate_confidence_threshold"], 0.05)
        self.assertIn("classes", data)
        self.assertEqual(len(data["classes"]), 5)
        # Verify all 5 research classes are present in the response
        classes_values = list(data["classes"].values()) if isinstance(data["classes"], dict) else data["classes"]
        self.assertIn("leaf rot", classes_values)
        self.assertIn("bud rot", classes_values)
        self.assertIn("gray leaf spot", classes_values)
        self.assertIn("stembleeding", classes_values)
        self.assertIn("bud root dropping", classes_values)

    def test_02_web_app_home(self):
        """Test GET / serves the mobile web application HTML."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Coconut Disease Detection & Plantation Health Monitor", response.text)
        self.assertIn("Engineering Notice:", response.text)

    def test_03_analyze_endpoint_real_image(self):
        """Test POST /api/v1/analyze with a real image produces valid Phase 5 schema."""
        if self.val_image_path.exists():
            with open(self.val_image_path, "rb") as f:
                img_bytes = f.read()
            filename = self.val_image_path.name
        else:
            img_bytes = self.synthetic_image_bytes
            filename = "synthetic_palm.jpg"

        files = {"file": (filename, img_bytes, "image/jpeg")}
        response = self.client.post("/api/v1/analyze", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify schema structure produced by Phase 5 engine
        self.assertIn("image_id", data)
        self.assertIn("observation_status", data)
        self.assertIn("accepted_detections", data)
        self.assertIn("candidate_detections", data)
        self.assertIn("relative_affected_area_proxy", data)
        self.assertIn("health_assessment", data)
        self.assertIn("inference_latency_ms", data)
        
        # Latency must be genuinely measured and strictly positive
        self.assertGreater(data["inference_latency_ms"], 0.0)
        
        # Health assessment adheres to Phase 5 structure
        assessment = data["health_assessment"]
        self.assertIn("phi_score", assessment)
        self.assertIn("health_tier", assessment)
        self.assertIn("disclaimer", assessment)
        self.assertGreaterEqual(assessment["phi_score"], 0.0)
        self.assertLessEqual(assessment["phi_score"], 100.0)

    def test_04_analyze_endpoint_synthetic_control(self):
        """Test POST /api/v1/analyze with synthetic plain image yields NO_DISEASE_DETECTED."""
        files = {"file": ("control.jpg", self.synthetic_image_bytes, "image/jpeg")}
        response = self.client.post("/api/v1/analyze", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["observation_status"], "NO_DISEASE_DETECTED")
        self.assertEqual(len(data["accepted_detections"]), 0)
        self.assertEqual(data["health_assessment"]["phi_score"], 100.0)
        self.assertIn("EXCELLENT", data["health_assessment"]["health_tier"])

    def test_05_analyze_invalid_mime_type(self):
        """Test POST /api/v1/analyze rejects non-image MIME types."""
        files = {"file": ("script.sh", b"echo 'hello'", "text/plain")}
        response = self.client.post("/api/v1/analyze", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", response.json()["detail"])

    def test_06_analyze_corrupt_image(self):
        """Test POST /api/v1/analyze rejects un-decodable byte content."""
        files = {"file": ("corrupt.jpg", b"NOT_AN_IMAGE_DATA_BYTES", "image/jpeg")}
        response = self.client.post("/api/v1/analyze", files=files)
        self.assertEqual(response.status_code, 422)
        self.assertIn("OpenCV failed to decode image", response.json()["detail"])

    def test_07_scouting_aggregate_endpoint(self):
        """Test POST /api/v1/aggregate correctly aggregates multiple palm inspections using Phase 5."""
        records = [
            {
                "image_id": "palm_001.jpg",
                "image_path": "palm_001.jpg",
                "plantation_id": "DEMO_COCONUT_BLOCK_VAL_01",
                "timestamp": "2026-09-20T05:00:00Z",
                "gps_location": None,
                "model_id": "EXP-002_imgsz512",
                "model_input_size": 512,
                "image_width": 768,
                "image_height": 1024,
                "observation_status": "DISEASE_DETECTED",
                "accepted_detection_count": 1,
                "candidate_detection_count": 0,
                "accepted_detections": [
                    {
                        "class_id": 3,
                        "disease_class": "leaf rot",
                        "confidence": 0.85,
                        "bounding_box": {"x1": 10.0, "y1": 20.0, "x2": 100.0, "y2": 120.0},
                        "normalized_box": {"xc": 0.071, "yc": 0.068, "w": 0.117, "h": 0.097}
                    }
                ],
                "candidate_detections": [],
                "detected_classes": ["leaf rot"],
                "relative_affected_area_proxy": 0.0114,
                "confidence_summary": {"mean": 0.85, "max": 0.85, "min": 0.85}
            },
            {
                "image_id": "palm_002.jpg",
                "image_path": "palm_002.jpg",
                "plantation_id": "DEMO_COCONUT_BLOCK_VAL_01",
                "timestamp": "2026-09-20T05:00:00Z",
                "gps_location": None,
                "model_id": "EXP-002_imgsz512",
                "model_input_size": 512,
                "image_width": 768,
                "image_height": 1024,
                "observation_status": "NO_DISEASE_DETECTED",
                "accepted_detection_count": 0,
                "candidate_detection_count": 0,
                "accepted_detections": [],
                "candidate_detections": [],
                "detected_classes": [],
                "relative_affected_area_proxy": 0.0,
                "confidence_summary": {"mean": 0.0, "max": 0.0, "min": 0.0}
            }
        ]
        
        payload = {
            "plantation_id": "DEMO_COCONUT_BLOCK_VAL_01",
            "session_id": "TEST_SESSION_01",
            "records": records
        }
        
        response = self.client.post("/api/v1/aggregate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        counts = data["observation_counts"]
        self.assertEqual(counts["total_images_submitted"], 2)
        self.assertEqual(counts["disease_positive_images"], 1)
        self.assertEqual(counts["disease_negative_images"], 1)
        
        rates = data["observation_rates"]
        self.assertAlmostEqual(rates["disease_positive_image_percentage"], 50.0, places=1)
        
        dist = data["disease_distribution"]
        self.assertEqual(dist["leaf rot"]["detection_count"], 1)
        self.assertEqual(dist["leaf rot"]["unique_images_affected"], 1)
        
        phi = data["plantation_health_index"]
        self.assertIn("phi_composite_score", phi)
        self.assertIn("health_tier", phi)
        self.assertGreaterEqual(phi["phi_composite_score"], 0.0)

    def test_08_scouting_aggregate_empty_payload(self):
        """Test POST /api/v1/aggregate rejects empty record lists."""
        payload = {
            "plantation_id": "DEMO_COCONUT_BLOCK_VAL_01",
            "session_id": "TEST_SESSION_02",
            "records": []
        }
        response = self.client.post("/api/v1/aggregate", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("No observation records provided", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()
