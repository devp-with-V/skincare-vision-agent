import base64
import io
import json
import logging
import time
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.models.schemas import ScanRequest, ScanResponse, AnalysisResult, LandmarkPoint
from app.core.face_detector import FaceDetector
from app.core.skin_analyzer import SkinAnalyzer
from app.core.severity_scorer import calculate_overall_severity
from app.agent.skin_agent import SkinAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skincare-vision-backend")

app = FastAPI(
    title="Skin Care Vision Agent API",
    description="Backend service for face mesh detection, skin condition analysis, and LLM advice",
    version="1.0.0"
)

# Enable CORS for Next.js frontend (allow any local network origin)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Engines
face_detector = FaceDetector()
skin_analyzer = SkinAnalyzer()
skin_agent = SkinAgent()

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "skincare-vision-backend"}

def decode_base64_image(base64_str: str) -> bytes:
    """Decode base64 string (optionally containing prefix data:image/jpeg;base64,) to raw bytes."""
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        return base64_str.encode("utf-8")
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid base64 image encoding")

@app.post("/api/scan", response_model=ScanResponse)
async def scan_image(request: ScanRequest):
    logger.info("Received scan request via REST API")
    try:
        # 1. Decode image
        img_bytes_base64 = decode_base64_image(request.image_base64)
        image_bytes = base64.b64decode(img_bytes_base64)

        # 2. Process face detection & crop regions
        face_detected, landmarks, regions = face_detector.process_frame(image_bytes)

        # Convert landmarks to schema format
        landmark_schemas = [
            LandmarkPoint(x=lm["x"], y=lm["y"], z=lm["z"]) for lm in landmarks
        ]

        if not face_detected:
            return ScanResponse(
                analysis=AnalysisResult(
                    face_detected=False,
                    landmarks=[],
                    regions={},
                    overall_severity=0.0
                )
            )

        # 3. Analyze cropped skin regions
        region_analyses = skin_analyzer.analyze_regions(regions)

        # 4. Calculate overall severity
        overall_severity = calculate_overall_severity(region_analyses)

        return ScanResponse(
            analysis=AnalysisResult(
                face_detected=True,
                landmarks=landmark_schemas,
                regions=region_analyses,
                overall_severity=overall_severity
            )
        )

    except Exception as e:
        logger.exception("Error during rest api scan")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/api/analyze", response_model=ScanResponse)
async def analyze_image(request: ScanRequest):
    logger.info("Received analyze request via REST API")
    try:
        # 1. Decode image
        img_bytes_base64 = decode_base64_image(request.image_base64)
        image_bytes = base64.b64decode(img_bytes_base64)

        # 2. Process face detection
        face_detected, landmarks, regions = face_detector.process_frame(image_bytes)
        
        landmark_schemas = [
            LandmarkPoint(x=lm["x"], y=lm["y"], z=lm["z"]) for lm in landmarks
        ]

        if not face_detected:
            return ScanResponse(
                analysis=AnalysisResult(
                    face_detected=False,
                    landmarks=[],
                    regions={},
                    overall_severity=0.0
                ),
                recommendations=None
            )

        # 3. Analyze cropped skin regions
        region_analyses = skin_analyzer.analyze_regions(regions)

        # 4. Calculate overall severity
        overall_severity = calculate_overall_severity(region_analyses)

        # 5. Generate LLM recommendations
        regions_serialized = {
            k: v.model_dump() for k, v in region_analyses.items()
        }
        recommendations = await skin_agent.generate_analysis(overall_severity, regions_serialized)

        return ScanResponse(
            analysis=AnalysisResult(
                face_detected=True,
                landmarks=landmark_schemas,
                regions=region_analyses,
                overall_severity=overall_severity
            ),
            recommendations=recommendations
        )

    except Exception as e:
        logger.exception("Error during rest api analyze")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    ws_write_lock = asyncio.Lock()
    active_analysis_task = None

    async def run_blemish_analysis(regions_to_analyze):
        try:
            loop = asyncio.get_running_loop()
            region_analyses = await loop.run_in_executor(
                None, skin_analyzer.analyze_regions, regions_to_analyze
            )
            overall_severity = calculate_overall_severity(region_analyses)
            regions_serialized = {
                k: v.model_dump() for k, v in region_analyses.items()
            }
            async with ws_write_lock:
                await websocket.send_json({
                    "type": "blemishes",
                    "regions": regions_serialized,
                    "overall_severity": overall_severity
                })
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("Error in background blemish analysis task:")

    try:
        while True:
            # Expect either Text (JSON base64) or Binary message (raw bytes)
            message = await websocket.receive()
            
            image_bytes = None
            if "bytes" in message:
                image_bytes = message["bytes"]
            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    if "image" in data:
                        base64_data = data["image"]
                        img_bytes_base64 = decode_base64_image(base64_data)
                        image_bytes = base64.b64decode(img_bytes_base64)
                except Exception as e:
                    logger.error(f"WebSocket text parse error: {str(e)}")
                    async with ws_write_lock:
                        await websocket.send_json({"error": "Invalid JSON format or base64 image"})
                    continue

            if image_bytes is None:
                async with ws_write_lock:
                    await websocket.send_json({"error": "No image data received"})
                continue

            # Process frame using MediaPipe + CV
            try:
                start_time = time.time()
                face_detected, landmarks, regions = face_detector.process_frame(image_bytes)
                
                # Send face mesh landmarks immediately to guarantee 30+ FPS live tracking
                async with ws_write_lock:
                    await websocket.send_json({
                        "type": "landmarks",
                        "face_detected": face_detected,
                        "landmarks": landmarks if face_detected else []
                    })

                # If face is detected, trigger blemish detection asynchronously in a background task
                if face_detected and regions:
                    if active_analysis_task is None or active_analysis_task.done():
                        active_analysis_task = asyncio.create_task(
                            run_blemish_analysis(regions)
                        )
                    # If active_analysis_task is still running, skip analysis of this frame (backpressure handling)

                elapsed = (time.time() - start_time) * 1000
                if not face_detected:
                    logger.info(f"Frame processed: No face detected. Latency: {elapsed:.1f}ms")
                else:
                    logger.info(f"Frame processed: Face landmarks tracked. Latency: {elapsed:.1f}ms")

            except Exception as e:
                logger.exception("WebSocket frame processing error during streaming:")
                try:
                    async with ws_write_lock:
                        await websocket.send_json({"error": f"Error processing frame: {str(e)}"})
                except Exception:
                    pass

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed by client")
    except Exception as e:
        logger.error(f"WebSocket exception: {str(e)}")
    finally:
        if active_analysis_task and not active_analysis_task.done():
            active_analysis_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass
