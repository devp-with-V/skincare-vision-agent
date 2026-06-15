import os
import shutil
import sys
import subprocess

def ensure_dependencies():
    """Ensure ultralytics is installed for training."""
    try:
        import ultralytics
        print("Ultralytics library is already installed.")
    except ImportError:
        print("Ultralytics not found. Installing ultralytics via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        print("Ultralytics installed successfully.")

ensure_dependencies()

from ultralytics import YOLO

# Resolve paths relative to ml folder
ML_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_YAML = os.path.join(ML_DIR, "data", "acne04", "dataset.yaml")
MODELS_DIR = os.path.join(ML_DIR, "models")
BACKEND_MODELS_DIR = os.path.abspath(os.path.join(ML_DIR, "..", "backend", "app", "models"))

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(BACKEND_MODELS_DIR, exist_ok=True)
    
    print("Initializing YOLOv8n model...")
    # Load pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")
    
    print(f"Starting model fine-tuning on dataset config: {DATASET_YAML}")
    # Train model for 3 epochs (quick run to verify pipeline & get weights)
    results = model.train(
        data=DATASET_YAML,
        epochs=3,
        imgsz=640,
        batch=8,
        device="cpu", # Force CPU for local dev compatibility
        workers=0     # Prevent multiprocessing issues on Windows
    )
    
    print("Training finished! Exporting model to ONNX format...")
    # Export model
    onnx_path = model.export(format="onnx", imgsz=640)
    print(f"Model exported to ONNX: {onnx_path}")
    
    # Locate the exported ONNX model
    # Ultralytics exports ONNX in the same directory as the weights/train folder (runs/detect/train/weights/best.onnx)
    runs_dir = os.path.join(os.getcwd(), "runs")
    best_onnx = None
    for root, dirs, files in os.walk(runs_dir):
        for file in files:
            if file == "best.onnx" or (file.endswith(".onnx") and "best" in file):
                best_onnx = os.path.join(root, file)
                break
    
    if not best_onnx:
        # Fallback if runs folder not in current cwd
        for root, dirs, files in os.walk(os.path.dirname(DATASET_YAML)):
            for file in files:
                if file.endswith(".onnx"):
                    best_onnx = os.path.join(root, file)
                    break

    if best_onnx and os.path.exists(best_onnx):
        # Copy to ml/models/yolov8n.onnx
        target_ml_onnx = os.path.join(MODELS_DIR, "yolov8n.onnx")
        shutil.copy(best_onnx, target_ml_onnx)
        print(f"Saved ONNX model to ML models directory: {target_ml_onnx}")
        
        # Copy to backend/app/models/yolov8n.onnx
        target_backend_onnx = os.path.join(BACKEND_MODELS_DIR, "yolov8n.onnx")
        shutil.copy(best_onnx, target_backend_onnx)
        print(f"Deployed ONNX model to FastAPI backend directory: {target_backend_onnx}")
    else:
        print("Error: Could not locate exported ONNX weights.")

if __name__ == "__main__":
    main()
