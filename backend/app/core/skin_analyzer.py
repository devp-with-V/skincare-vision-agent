import cv2
import numpy as np
import os
import logging
from typing import Dict, List
import onnxruntime as ort

from app.models.schemas import RegionAnalysis, Detection
from app.core.preprocessing import preprocess_skin_crop
from app.core.severity_scorer import calculate_region_severity, get_dominant_concern

logger = logging.getLogger("skincare-vision-backend")

class SkinAnalyzer:
    def __init__(self):
        self.model_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "models", "yolov8n.onnx"
        ))
        
        self.session = None
        self.classes = ["acne", "dark_spot", "redness", "dryness_patch"]
        
        # Initialize ONNX runtime session
        self._init_model()

    def _init_model(self):
        """Attempts to load the YOLOv8 ONNX model. Logs a warning on failure."""
        if not os.path.exists(self.model_path):
            logger.warning(
                f"YOLOv8 ONNX weights not found at: {self.model_path}. "
                "Backend will run in mock mode. Run the training script in 'ml/' to generate model weights."
            )
            return

        try:
            # Initialize with CPU provider
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            logger.info("YOLOv8 ONNX model loaded successfully inside FastAPI backend.")
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8 ONNX session: {str(e)}. Falling back to mock mode.")
            self.session = None

    def analyze_regions(self, regions: Dict[str, np.ndarray]) -> Dict[str, RegionAnalysis]:
        """
        Runs white-balance/lighting normalization and YOLOv8 ONNX inference on each skin region.
        Falls back to mock analysis if the model weights are not loaded.
        """
        analysis = {}
        for region_name, region_img in regions.items():
            if region_img is None or region_img.size == 0:
                continue

            # Calculate luminance of the color-corrected region to adjust confidence thresholds adaptively
            lab = cv2.cvtColor(region_img, cv2.COLOR_BGR2LAB)
            mean_luminance = float(np.mean(lab[:, :, 0])) / 255.0

            # 1. Preprocess skin crop (Gray World White Balance + CLAHE)
            preprocessed = preprocess_skin_crop(region_img)

            # 2. Run inference (ONNX or Mock)
            if self.session is not None:
                detections = self._run_onnx_inference(preprocessed, region_name, mean_luminance)
            else:
                detections = self._get_mock_detections(region_name)

            # 3. Calculate metrics
            severity = calculate_region_severity(detections)
            concern = get_dominant_concern(detections)

            analysis[region_name] = RegionAnalysis(
                region=region_name,
                severity_score=severity,
                dominant_concern=concern,
                detections=detections
            )
            
        return analysis

    def _run_onnx_inference(self, img: np.ndarray, region_name: str, mean_luminance: float) -> List[Detection]:
        """Runs the raw preprocessed image through the YOLOv8 ONNX session using adaptive thresholds."""
        try:
            # Resize image to standard YOLOv8 input size (640x640)
            img_resized = cv2.resize(img, (640, 640))
            # Convert BGR (OpenCV) to RGB (YOLOv8)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to float, normalize [0, 1], transpose to BCHW format
            input_data = img_rgb.astype(np.float32) / 255.0
            input_data = input_data.transpose(2, 0, 1) # HWC -> CHW
            input_data = np.expand_dims(input_data, axis=0) # CHW -> BCHW
            input_data = np.ascontiguousarray(input_data)

            # Run ONNX inference
            outputs = self.session.run(None, {self.input_name: input_data})
            output = outputs[0] # Shape: (1, 4 + num_classes, 8400) -> (1, 8, 8400)
            
            # Post-processing: Filter predictions by adaptive confidence thresholds
            predictions = np.squeeze(output).T # Shape: (8400, 8)
            
            # Class-specific baseline confidence thresholds
            baseline_thresholds = {
                "acne": 0.18,
                "dark_spot": 0.22,
                "redness": 0.25,
                "dryness_patch": 0.20
            }

            # Region-specific modifiers
            region_modifiers = {
                "nose": {"redness": 1.25, "dark_spot": 1.1},
                "forehead": {"dryness_patch": 1.1, "redness": 1.1},
                "chin": {"acne": 1.1}
            }

            boxes = []
            confidences = []
            class_ids = []
            
            for pred in predictions:
                scores = pred[4:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                class_name = self.classes[class_id]
                
                # Get baseline threshold
                base_conf = baseline_thresholds.get(class_name, 0.15)
                
                # Apply regional modifier
                region_mod = region_modifiers.get(region_name, {}).get(class_name, 1.0)
                conf_thresh = base_conf * region_mod
                
                # Apply luminance modifier: if image is dark (< 0.4), scale up threshold to reduce shadow false positives
                if mean_luminance < 0.4:
                    shadow_factor = 1.0 + (0.4 - mean_luminance) * 0.8
                    conf_thresh *= shadow_factor
                # If image is very bright (> 0.75), scale down slightly to capture low-contrast washed-out spots
                elif mean_luminance > 0.75:
                    conf_thresh *= 0.9
                    
                conf_thresh = max(0.10, min(0.85, conf_thresh))
                
                if confidence >= conf_thresh:
                    xc, yc, w, h = pred[:4]
                    
                    # Convert center coords to min/max box coords (0.0 to 1.0)
                    x_min = max(0.0, min(1.0, (xc - w / 2) / 640.0))
                    y_min = max(0.0, min(1.0, (yc - h / 2) / 640.0))
                    x_max = max(0.0, min(1.0, (xc + w / 2) / 640.0))
                    y_max = max(0.0, min(1.0, (yc + h / 2) / 640.0))
                    
                    boxes.append([x_min, y_min, x_max, y_max])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))
            
            # Apply NMS
            pixel_boxes = []
            for box in boxes:
                x_min, y_min, x_max, y_max = box
                pixel_boxes.append([
                    int(x_min * 640), int(y_min * 640),
                    int((x_max - x_min) * 640), int((y_max - y_min) * 640)
                ])
                
            # Filter with NMS using IoU threshold of 0.45
            indices = cv2.dnn.NMSBoxes(pixel_boxes, confidences, 0.0, 0.45)
            
            detections_list = []
            if len(indices) > 0:
                for i in indices.flatten():
                    detections_list.append(Detection(
                        class_name=self.classes[class_ids[i]],
                        confidence=round(confidences[i], 2),
                        bbox=boxes[i]
                    ))
                    
            return detections_list

        except Exception as e:
            logger.error(f"Error running ONNX model inference: {str(e)}")
            return []

    def _get_mock_detections(self, region_name: str) -> List[Detection]:
        """Provides mock detections for Phase 1 verification if model weights are missing."""
        if region_name == "forehead":
            return [Detection(class_name="dryness_patch", confidence=0.82, bbox=[0.2, 0.2, 0.4, 0.4])]
        elif "cheek" in region_name:
            return [
                Detection(class_name="acne", confidence=0.88, bbox=[0.3, 0.3, 0.45, 0.45]),
                Detection(class_name="acne", confidence=0.75, bbox=[0.55, 0.55, 0.7, 0.7])
            ]
        elif region_name == "nose":
            return [Detection(class_name="redness", confidence=0.91, bbox=[0.35, 0.35, 0.65, 0.65])]
        return []
