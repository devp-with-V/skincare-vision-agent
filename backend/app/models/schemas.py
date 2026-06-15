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

class SkincareStep(BaseModel):
    step_number: int = Field(..., description="Chronological step index")
    time_of_day: str = Field(..., description="AM or PM")
    step_name: str = Field(..., description="Action category (e.g. Cleanse, Treat, Moisturize)")
    ingredients: List[str] = Field(default_factory=list, description="Recommended active ingredients")
    instructions: str = Field(..., description="Specific usage directions")

class SkincareRecommendations(BaseModel):
    condition_name: str = Field(..., description="Overall suggested condition category")
    condition_desc: str = Field(..., description="Paragraph description of the condition findings")
    overall_summary: str = Field(..., description="High-level advisor summary")
    dermatologist_flag: bool = Field(..., description="True if severe concerns are found")
    dermatologist_reason: Optional[str] = Field(None, description="Reasoning for dermatologist flag")
    disclaimer: str = Field(..., description="Medical advice waiver disclaimer text")
    routine: List[SkincareStep] = Field(default_factory=list, description="AM and PM steps")
    lifestyle_tips: List[str] = Field(default_factory=list, description="Lifestyle suggestions")

class ScanRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image frame")
    user_id: Optional[str] = Field(None, description="Optional user ID for session tracking")

class ScanResponse(BaseModel):
    session_id: Optional[str] = Field(None, description="Generated session ID")
    analysis: AnalysisResult
    recommendations: Optional[SkincareRecommendations] = Field(None, description="AI agent recommendation results")
