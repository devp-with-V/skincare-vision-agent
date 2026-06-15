from typing import List, Dict
from app.models.schemas import RegionAnalysis, Detection

def calculate_region_severity(detections: List[Detection]) -> float:
    """
    Calculate region severity score based on the number and confidence of detections.
    Returns: float (0.0 to 1.0).
    """
    if not detections:
        return 0.0
        
    count = len(detections)
    
    # Highest confidence detection
    max_conf = max(d.confidence for d in detections)
    
    # Score calculation: base weight per detection + max confidence weight
    # 3 detections of high confidence should yield around 0.6 - 0.7 severity
    raw_score = (count * 0.12) + (max_conf * 0.3)
    
    # Cap between 0.0 and 1.0 and round
    return min(1.0, round(raw_score, 2))

def get_dominant_concern(detections: List[Detection]) -> str | None:
    """Find the most frequent detected blemish class in a region."""
    if not detections:
        return None
        
    counts = {}
    for d in detections:
        counts[d.class_name] = counts.get(d.class_name, 0) + 1
        
    return max(counts, key=counts.get)

def calculate_overall_severity(region_analyses: Dict[str, RegionAnalysis]) -> float:
    """
    Calculate overall face severity score.
    Weighted based on region surface area representation:
    - cheeks: 30% each (60% total)
    - forehead: 25%
    - nose: 10%
    - chin: 5%
    """
    if not region_analyses:
        return 0.0
        
    weights = {
        "forehead": 0.25,
        "left_cheek": 0.30,
        "right_cheek": 0.30,
        "nose": 0.10,
        "chin": 0.05
    }
    
    weighted_sum = 0.0
    weight_total = 0.0
    
    for region_name, analysis in region_analyses.items():
        weight = weights.get(region_name, 0.20)
        weighted_sum += analysis.severity_score * weight
        weight_total += weight
        
    if weight_total == 0.0:
        return 0.0
        
    return round(weighted_sum / weight_total, 2)
