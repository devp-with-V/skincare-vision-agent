from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class LandmarkPoint(BaseModel):
    x: float = Field(..., description="Normalized X coordinate (0.0 to 1.0)")
    y: float = Field(..., description="Normalized Y coordinate (0.0 to 1.0)")
    z: float = Field(..., description="Normalized Z coordinate (depth)")

class Detection(BaseModel):
    class_name: str = Field(..., description="Detected skin condition class (e.g., acne, dark_spot)")
    confidence: float = Field(..., description="Detection confidence score (0.0 to 1.0)")
    bbox: List[float] = Field(..., description="Normalized bounding box [x_min, y_min, x_max, y_max]")

class RegionAnalysis(BaseModel):
    region: str = Field(..., description="Name of the face region (e.g., forehead, left_cheek, right_cheek, nose, chin)")
    severity_score: float = Field(..., description="Region severity score (0.0 to 1.0)")
    dominant_concern: Optional[str] = Field(None, description="Primary concern in this region")
    detections: List[Detection] = Field(default_factory=list, description="Detections found in this region")

class AnalysisResult(BaseModel):
    face_detected: bool = Field(..., description="Whether a face was detected in the frame")
    landmarks: List[LandmarkPoint] = Field(default_factory=list, description="468 face mesh landmarks")
    regions: Dict[str, RegionAnalysis] = Field(default_factory=dict, description="Per-region skin analysis details")
    overall_severity: float = Field(0.0, description="Overall face severity score (0.0 to 1.0)")

class ScanRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image frame")
    user_id: Optional[str] = Field(None, description="Optional user ID for session tracking")

class ScanResponse(BaseModel):
    session_id: Optional[str] = Field(None, description="Generated session ID")
    analysis: AnalysisResult
    recommendations: Optional[Dict] = Field(None, description="AI agent recommendation results")
