import pytest
from fastapi.testclient import TestClient
import numpy as np

from app.main import app
from app.core.face_detector import FaceDetector
from app.core.skin_analyzer import SkinAnalyzer
from app.core.severity_scorer import calculate_overall_severity

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
    
    overall = calculate_overall_severity(analysis)
    assert 0.0 <= overall <= 1.0

def test_face_detector_stub():
    detector = FaceDetector()
    dummy_bytes = b"dummy_image_data"
    
    # Passing invalid bytes should log an error and return False
    face_detected, landmarks, regions = detector.process_frame(dummy_bytes)
    assert face_detected is False
    assert len(landmarks) == 0
    assert len(regions) == 0

def test_onnx_model_inference_latency():
    import time
    analyzer = SkinAnalyzer()
    
    # Assert model loaded successfully
    assert analyzer.session is not None, "ONNX model session should be initialized successfully"
    
    # Create dummy skin crop image
    dummy_crop = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Measure execution speed
    start_time = time.time()
    detections = analyzer._run_onnx_inference(dummy_crop)
    elapsed = (time.time() - start_time) * 1000  # in ms
    
    print(f"\nYOLOv8 ONNX Inference latency: {elapsed:.2f}ms")
    assert elapsed < 100.0, f"ONNX inference took {elapsed:.2f}ms, which is above the 100ms threshold!"
    assert isinstance(detections, list)

