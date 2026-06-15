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

def test_skin_agent_mock_generator():
    from app.agent.skin_agent import SkinAgent
    agent = SkinAgent()
    
    dummy_regions_data = {
        "forehead": {"severity_score": 0.15, "dominant_concern": "dryness_patch", "detections": []},
        "left_cheek": {"severity_score": 0.35, "dominant_concern": "acne", "detections": []}
    }
    
    res = agent._generate_mock_agent_response(0.25, dummy_regions_data)
    assert res["condition_name"] == "Mild-to-Moderate Acne vulgaris"
    assert res["dermatologist_flag"] is False
    assert len(res["routine"]) > 0
    assert len(res["lifestyle_tips"]) > 0

def test_analyze_route():
    # Call with a dummy base64 payload
    payload = {
        "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "analysis" in data
    assert data["analysis"]["face_detected"] is False  # Dummy image has no face
    assert data["recommendations"] is None


