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
    detections = analyzer._run_onnx_inference(dummy_crop, "left_cheek", 0.5)
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

def test_compile_user_prompt():
    from app.agent.prompts import compile_user_prompt
    regions_data = {
        "forehead": {"severity_score": 0.15, "dominant_concern": "dryness_patch", "detections": [{"class_name": "dryness"}]},
        "left_cheek": {"severity_score": 0.35, "dominant_concern": "acne", "detections": [{"class_name": "acne"}]}
    }
    prompt = compile_user_prompt(0.25, regions_data)
    assert "Overall Skin Severity Rating: 25%" in prompt
    assert "- FOREHEAD:" in prompt
    assert "Severity Score: 15%" in prompt
    assert "Dominant Concern: dryness patch" in prompt
    assert "Localized Blemishes: 1 count" in prompt

@pytest.mark.anyio
async def test_skin_agent_claude_api_parsing():
    from app.agent.skin_agent import SkinAgent
    from unittest.mock import AsyncMock
    import os
    
    # Temporarily set ANTHROPIC_API_KEY to test the api client path
    os.environ["ANTHROPIC_API_KEY"] = "fake_key"
    agent = SkinAgent()
    
    # Mock the client message return value
    mock_message = AsyncMock()
    mock_message.content = [
        AsyncMock(text='{"condition_name": "Test Acne", "condition_desc": "Desc", "overall_summary": "Summary", "dermatologist_flag": false, "dermatologist_reason": null, "disclaimer": "Disclaimer", "routine": [], "lifestyle_tips": []}')
    ]
    agent.client = AsyncMock()
    agent.client.messages.create = AsyncMock(return_value=mock_message)
    
    res = await agent.generate_analysis(0.2, {})
    assert res["condition_name"] == "Test Acne"
    assert res["dermatologist_flag"] is False
    
    # Test wrapping in markdown block
    mock_message.content = [
        AsyncMock(text='```json\n{"condition_name": "Test Markdown", "condition_desc": "Desc", "overall_summary": "Summary", "dermatologist_flag": true, "dermatologist_reason": "Severe", "disclaimer": "Disclaimer", "routine": [], "lifestyle_tips": []}\n```')
    ]
    res = await agent.generate_analysis(0.6, {})
    assert res["condition_name"] == "Test Markdown"
    assert res["dermatologist_flag"] is True
    
    # Cleanup environment
    del os.environ["ANTHROPIC_API_KEY"]

def test_vlm_agent_mock_generator():
    from app.agent.vlm_agent import VLMAgent
    agent = VLMAgent()
    res = agent._generate_mock_vlm_response()
    assert res["condition_name"] == "Mild Acne & Dehydrated Forehead (VLM Cloud Agent)"
    assert res["overall_severity"] == 0.28
    assert "forehead" in res["regions"]
    assert len(res["routine"]) > 0

@pytest.mark.anyio
async def test_vlm_agent_openrouter_parsing():
    from app.agent.vlm_agent import VLMAgent
    from unittest.mock import AsyncMock
    import os
    
    os.environ["OPENROUTER_API_KEY"] = "fake_openrouter_key"
    agent = VLMAgent()
    
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = lambda: {
        "choices": [
            {
                "message": {
                    "content": '{"condition_name": "VLM Test", "overall_severity": 0.4, "regions": {}}'
                }
            }
        ]
    }
    
    import httpx
    # Mock httpx.AsyncClient.post
    original_post = httpx.AsyncClient.post
    httpx.AsyncClient.post = AsyncMock(return_value=mock_response)
    
    try:
        res = await agent.analyze_image("dummy_image")
        assert res["condition_name"] == "VLM Test"
        assert res["overall_severity"] == 0.4
    finally:
        httpx.AsyncClient.post = original_post
        del os.environ["OPENROUTER_API_KEY"]

def test_analyze_vlm_route():
    payload = {
        "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
    }
    response = client.post("/api/analyze-vlm", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "analysis" in data
    assert data["analysis"]["face_detected"] is False  # Dummy has no face
    assert data["recommendations"] is None


def test_websocket_endpoint_decoupled():
    # Send a dummy base64 frame (no face)
    payload = {
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
    }
    with client.websocket_connect("/api/ws") as websocket:
        websocket.send_json(payload)
        
        # We expect immediate landmarks response
        msg = websocket.receive_json()
        assert "type" in msg
        assert msg["type"] == "landmarks"
        assert msg["face_detected"] is False
        assert msg["landmarks"] == []



