import base64
import os
import cv2
import numpy as np
import traceback

from app.core.face_detector import FaceDetector
from app.core.skin_analyzer import SkinAnalyzer

def test_simulation():
    print("Starting WebSocket frame simulation test...")
    try:
        # Create a dummy image representing a face
        # We'll draw a mock circle representing a face to see if MediaPipe handles it
        # or load a sample image if one exists.
        # Since we just want to verify the pipeline doesn't raise exceptions:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add basic light skin colored circle
        cv2.circle(img, (320, 240), 150, (180, 200, 230), -1)
        
        # Encode to bytes
        _, img_encoded = cv2.imencode(".jpg", img)
        image_bytes = img_encoded.tobytes()

        print("Initializing FaceDetector...")
        face_detector = FaceDetector()
        print("Initializing SkinAnalyzer...")
        skin_analyzer = SkinAnalyzer()

        print("Processing simulated frame...")
        face_detected, landmarks, regions = face_detector.process_frame(image_bytes)
        print(f"Frame processed. Face detected: {face_detected}")
        print(f"Number of landmarks: {len(landmarks)}")
        print(f"Number of regions: {len(regions)}")
        
        if face_detected:
            print("Analyzing regions...")
            region_analyses = skin_analyzer.analyze_regions(regions)
            print(f"Analysis completed for {len(region_analyses)} regions.")

        print("Simulation completed successfully without exceptions!")

    except Exception as e:
        print("\n!!! EXCEPTION CAUGHT DURING SIMULATION !!!")
        traceback.print_exc()

if __name__ == "__main__":
    test_simulation()
