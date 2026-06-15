import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request
import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger("skincare-vision-backend")

class FaceDetector:
    def __init__(self):
        # 1. Path to local model weight
        self.model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        self.model_path = os.path.join(self.model_dir, "face_landmarker.task")
        
        # Ensure model exists or download it automatically
        self._ensure_model_downloaded()

        # 2. Initialize MediaPipe Face Landmarker Tasks API
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Define landmark indices for skin region segmentation
        self.REGION_LANDMARKS = {
            "forehead": [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 297, 332, 284, 251, 389, 356, 454, 323, 361],
            "left_cheek": [111, 116, 117, 118, 101, 50, 187, 205, 207, 206, 203, 98, 36, 142, 228, 229, 230, 231, 232, 233],
            "right_cheek": [340, 345, 346, 347, 330, 280, 411, 425, 427, 426, 423, 327, 266, 371, 448, 449, 450, 451, 452, 453],
            "nose": [168, 6, 197, 195, 5, 4, 122, 196, 3, 51, 45, 275, 274, 351, 419, 420, 294, 327, 98, 197],
            "chin": [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 58, 172, 136, 150, 149, 176, 148, 18, 200, 199]
        }

    def _ensure_model_downloaded(self):
        """Ensure the MediaPipe Task model is downloaded locally; download if not."""
        if not os.path.exists(self.model_path):
            logger.info(f"Model file not found. Creating directories and downloading to: {self.model_path}")
            os.makedirs(self.model_dir, exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            try:
                urllib.request.urlretrieve(url, self.model_path)
                logger.info("MediaPipe Face Landmarker model downloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to download MediaPipe task model: {str(e)}")
                raise RuntimeError(f"Could not download FaceLandmarker model task file: {str(e)}")

    def process_frame(self, image_bytes: bytes) -> Tuple[bool, List[Dict[str, float]], Dict[str, np.ndarray]]:
        """
        Process a single image frame to detect face landmarks and segment skin regions.
        
        Args:
            image_bytes: Raw JPEG/PNG image bytes.
            
        Returns:
            - face_detected: Boolean indicating if a face was detected.
            - landmarks: List of 468 landmark dicts (x, y, z).
            - regions: Dict mapping region name (e.g. 'forehead') to the cropped BGR image array.
        """
        # 1. Decode image bytes to OpenCV format (BGR)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Failed to decode image bytes")
            return False, [], {}

        h, w, _ = img.shape

        # 2. Convert to RGB for MediaPipe and wrap in mp.Image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # 3. Detect Face Landmarks
        results = self.detector.detect(mp_image)

        # 4. Handle case when no face is detected
        if not results.face_landmarks:
            return False, [], {}

        # Get first detected face landmarks
        face_landmarks = results.face_landmarks[0]
        
        # 5. Extract landmarks list (top 468 points)
        landmarks_list = []
        for lm in face_landmarks[:468]:
            landmarks_list.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z
            })

        # 6. Extract skin regions of interest (ROIs)
        regions_crops = {}
        for region_name, indices in self.REGION_LANDMARKS.items():
            try:
                # Get pixel coordinates for the region's landmarks
                pts = []
                for idx in indices:
                    if idx < len(face_landmarks):
                        lm = face_landmarks[idx]
                        px_x = int(lm.x * w)
                        px_y = int(lm.y * h)
                        pts.append([px_x, px_y])
                
                if len(pts) < 3:
                    continue
                
                pts = np.array(pts, dtype=np.int32)
                
                # Create convex hull for a smoother mask boundary
                hull = cv2.convexHull(pts)
                
                # Create mask
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, hull, 255)
                
                # Apply mask to image
                masked_img = cv2.bitwise_and(img, img, mask=mask)
                
                # Crop bounding box of the region
                x, y, crop_w, crop_h = cv2.boundingRect(hull)
                
                # Check for valid bounding box coordinates
                x = max(0, x)
                y = max(0, y)
                crop_w = min(w - x, crop_w)
                crop_h = min(h - y, crop_h)
                
                if crop_w > 0 and crop_h > 0:
                    cropped_region = masked_img[y:y+crop_h, x:x+crop_w]
                    regions_crops[region_name] = cropped_region
                    
            except Exception as e:
                logger.error(f"Error extracting region {region_name}: {str(e)}")
                continue

        return True, landmarks_list, regions_crops
