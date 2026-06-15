import numpy as np
from typing import Dict
from app.models.schemas import RegionAnalysis, Detection

class SkinAnalyzer:
    def __init__(self):
        # Placeholder for ML models loaded in Phase 2
        pass

    def analyze_regions(self, regions: Dict[str, np.ndarray]) -> Dict[str, RegionAnalysis]:
        """
        Analyze cropped region images.
        For Phase 1, we return mock/placeholder results for testing integration.
        """
        analysis = {}
        for region_name, region_img in regions.items():
            if region_img is None or region_img.size == 0:
                continue

            # Mock analysis results for demo verification
            # Generate a mild severity score based on the region name just to show differentiation
            if region_name == "forehead":
                severity = 0.15
                concern = "dryness_patch"
                detections = [
                    Detection(
                        class_name="dryness_patch",
                        confidence=0.82,
                        bbox=[0.2, 0.2, 0.4, 0.4] # Mock bounding box inside region
                    )
                ]
            elif "cheek" in region_name:
                severity = 0.35
                concern = "acne"
                detections = [
                    Detection(
                        class_name="acne",
                        confidence=0.88,
                        bbox=[0.3, 0.3, 0.35, 0.35]
                    ),
                    Detection(
                        class_name="acne",
                        confidence=0.75,
                        bbox=[0.5, 0.4, 0.55, 0.45]
                    )
                ]
            elif region_name == "nose":
                severity = 0.42
                concern = "redness"
                detections = [
                    Detection(
                        class_name="redness",
                        confidence=0.91,
                        bbox=[0.4, 0.4, 0.6, 0.6]
                    )
                ]
            else: # chin
                severity = 0.1
                concern = None
                detections = []

            analysis[region_name] = RegionAnalysis(
                region=region_name,
                severity_score=severity,
                dominant_concern=concern,
                detections=detections
            )
        
        return analysis

    def calculate_overall_severity(self, region_analyses: Dict[str, RegionAnalysis]) -> float:
        """
        Calculate overall skin severity score.
        Weighted average of region severity scores.
        """
        if not region_analyses:
            return 0.0
        
        # Simple average for now
        total_score = sum(r.severity_score for r in region_analyses.values())
        return round(total_score / len(region_analyses), 2)
