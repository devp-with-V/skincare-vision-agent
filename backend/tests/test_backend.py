import pytest
from fastapi.testclient import TestClient
import numpy as np

from app.main import app
from app.core.face_detector import FaceDetector
from app.core.skin_analyzer import SkinAnalyzer

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "skincare-vision-backend"}

def test_skin_analyzer():
    analyzer = SkinAnalyzer()
    
    # Create dummy numpy arrays representing regions
    regions = {
        "forehead": np.zeros((50, 50, 3), dtype=np.uint8),
        "left_cheek": np.zeros((50, 50, 3), dtype=np.uint8),
        "right_cheek": np.zeros((50, 50, 3), dtype=np.uint8),
        "nose": np.zeros((50, 50, 3), dtype=np.uint8),
        "chin": np.zeros((50, 50, 3), dtype=np.uint8)
    }
    
    analysis = analyzer.analyze_regions(regions)
    assert len(analysis) == 5
    assert "forehead" in analysis
    assert "left_cheek" in analysis
    
    overall = analyzer.calculate_overall_severity(analysis)
    assert 0.0 <= overall <= 1.0

def test_face_detector_stub():
    detector = FaceDetector()
    dummy_bytes = b"dummy_image_data"
    
    # Passing invalid bytes should log an error and return False
    face_detected, landmarks, regions = detector.process_frame(dummy_bytes)
    assert face_detected is False
    assert len(landmarks) == 0
    assert len(regions) == 0
