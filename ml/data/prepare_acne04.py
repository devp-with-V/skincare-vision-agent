import os
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw
from typing import Tuple, List, Dict

# Create standard dataset directory paths
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "acne04"))
IMAGES_TRAIN = os.path.join(DATASET_DIR, "images", "train")
IMAGES_VAL = os.path.join(DATASET_DIR, "images", "val")
LABELS_TRAIN = os.path.join(DATASET_DIR, "labels", "train")
LABELS_VAL = os.path.join(DATASET_DIR, "labels", "val")

# Classes: 0=acne, 1=dark_spot, 2=redness, 3=dryness_patch
CLASSES = ["acne", "dark_spot", "redness", "dryness_patch"]

def create_directories():
    """Create directory structure for YOLOv8 training."""
    for path in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        os.makedirs(path, exist_ok=True)
    print(f"Created YOLO dataset directories in {DATASET_DIR}")

def generate_mock_skin_image(width=640, height=640) -> Tuple[np.ndarray, List[Dict]]:
    """
    Generate a mock skin image simulating face skin with blemishes and labels.
    - Simulates skin tone gradients
    - Adds random blemishes (acne, dark spots, redness, dryness)
    - Returns the image array (BGR) and a list of detections/labels
    """
    # 1. Base skin tone (RGB)
    base_r = random.randint(210, 245)
    base_g = random.randint(165, 200)
    base_b = random.randint(140, 175)
    
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create soft skin tone gradient
    for y in range(height):
        factor = y / height
        r = int(base_r - (factor * 20))
        g = int(base_g - (factor * 15))
        b = int(base_b - (factor * 10))
        img[y, :] = [b, g, r] # BGR format
        
    # Apply soft Gaussian blur to smooth the skin texture
    img = cv2.GaussianBlur(img, (15, 15), 0)
    
    # Add subtle skin texture noise
    noise = np.random.normal(0, 2, (height, width, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    labels = []
    num_blemishes = random.randint(3, 8)
    
    for _ in range(num_blemishes):
        class_idx = random.randint(0, 3)
        b_w = random.randint(15, 45)
        b_h = random.randint(15, 45)
        
        # Ensure blemish is well within the image border
        x_center = random.randint(b_w * 2, width - b_w * 2)
        y_center = random.randint(b_h * 2, height - b_h * 2)
        
        x_min = x_center - b_w // 2
        y_min = y_center - b_h // 2
        x_max = x_center + b_w // 2
        y_max = y_center + b_h // 2

        # Draw the blemish onto the image
        if class_idx == 0:  # Acne (red raised bump, optional white center)
            # Red halo
            cv2.circle(img, (x_center, y_center), b_w // 2, (20, 25, 200), -1)
            # Soften halo edge
            img = cv2.GaussianBlur(img, (3, 3), 0)
            # White/Yellow pus tip (Grade 3/4 acne)
            if random.random() > 0.4:
                cv2.circle(img, (x_center, y_center), b_w // 6, (200, 240, 245), -1)
                
        elif class_idx == 1:  # Dark Spot (brown, flat, irregular)
            # Draw irregular blob
            pts = np.array([
                [x_min + random.randint(0, 5), y_min + random.randint(0, 5)],
                [x_max - random.randint(0, 5), y_min + random.randint(0, 5)],
                [x_max - random.randint(0, 5), y_max - random.randint(0, 5)],
                [x_min + random.randint(0, 5), y_max - random.randint(0, 5)]
            ], dtype=np.int32)
            cv2.fillPoly(img, [pts], (40, 60, 110)) # Dark brown
            img = cv2.GaussianBlur(img, (5, 5), 0)
            
        elif class_idx == 2:  # Redness (large, soft red patches)
            radius = random.randint(35, 75)
            # Overlay soft red circle with weight
            overlay = img.copy()
            cv2.circle(overlay, (x_center, y_center), radius, (50, 60, 210), -1)
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
            
        elif class_idx == 3:  # Dryness Patch (greyish/flaky rough oval)
            radius_x = random.randint(30, 60)
            radius_y = random.randint(15, 35)
            overlay = img.copy()
            cv2.ellipse(overlay, (x_center, y_center), (radius_x, radius_y), random.randint(0, 180), 0, 360, (190, 200, 210), -1)
            cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
            
        # Normalize coordinates for YOLO
        norm_x_center = x_center / width
        norm_y_center = y_center / height
        norm_w = b_w / width
        norm_h = b_h / height
        
        labels.append({
            "class": class_idx,
            "bbox": [norm_x_center, norm_y_center, norm_w, norm_h]
        })
        
    return img, labels

def save_data(img: np.ndarray, labels: list, img_path: str, label_path: str):
    """Save OpenCV BGR image and YOLO format annotations."""
    # Save Image
    cv2.imwrite(img_path, img)
    
    # Save Labels
    with open(label_path, "w") as f:
        for label in labels:
            c = label["class"]
            bx, by, bw, bh = label["bbox"]
            f.write(f"{c} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")

def apply_augmentations(img: np.ndarray, labels: list) -> Tuple[np.ndarray, List[Dict]]:
    """Apply random brightness, contrast and horizontal flip augmentations."""
    aug_img = img.copy()
    aug_labels = []

    # 1. Random Flip (Horizontal)
    if random.random() > 0.5:
        aug_img = cv2.flip(aug_img, 1)
        for label in labels:
            c = label["class"]
            bx, by, bw, bh = label["bbox"]
            # Flip x coordinate: new_x = 1.0 - old_x
            aug_labels.append({
                "class": c,
                "bbox": [1.0 - bx, by, bw, bh]
            })
    else:
        aug_labels = [l.copy() for l in labels]

    # 2. Random Brightness / Contrast
    alpha = random.uniform(0.85, 1.15) # Contrast
    beta = random.randint(-20, 20)      # Brightness
    aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

    return aug_img, aug_labels

def main(num_train=80, num_val=20):
    """Main data generator process."""
    create_directories()
    
    print(f"Generating mock Acne04 training data ({num_train} samples)...")
    for i in range(num_train):
        img, labels = generate_mock_skin_image()
        
        # Save base image
        img_name = f"mock_skin_train_{i}.jpg"
        lbl_name = f"mock_skin_train_{i}.txt"
        save_data(img, labels, os.path.join(IMAGES_TRAIN, img_name), os.path.join(LABELS_TRAIN, lbl_name))
        
        # Generate and save an augmented image
        aug_img, aug_labels = apply_augmentations(img, labels)
        aug_img_name = f"mock_skin_train_{i}_aug.jpg"
        aug_lbl_name = f"mock_skin_train_{i}_aug.txt"
        save_data(aug_img, aug_labels, os.path.join(IMAGES_TRAIN, aug_img_name), os.path.join(LABELS_TRAIN, aug_lbl_name))

    print(f"Generating mock Acne04 validation data ({num_val} samples)...")
    for i in range(num_val):
        img, labels = generate_mock_skin_image()
        img_name = f"mock_skin_val_{i}.jpg"
        lbl_name = f"mock_skin_val_{i}.txt"
        save_data(img, labels, os.path.join(IMAGES_VAL, img_name), os.path.join(LABELS_VAL, lbl_name))

    # Generate dataset config yaml file for YOLOv8
    yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {DATASET_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("\nnames:\n")
        for idx, name in enumerate(CLASSES):
            f.write(f"  {idx}: {name}\n")
            
    print(f"Dataset config written to {yaml_path}")
    print("Dataset preparation complete.")

if __name__ == "__main__":
    main()
