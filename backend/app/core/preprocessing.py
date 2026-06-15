import cv2
import numpy as np

def normalize_lighting(img: np.ndarray) -> np.ndarray:
    """
    Normalizes lighting variance in skin crop using CLAHE on LAB color space.
    This maintains the chromaticity (color values) while equalizing the lightness.
    """
    if img is None or img.size == 0:
        return img
        
    # Convert from BGR to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    limg = cv2.merge((cl, a, b))
    result = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return result

def correct_white_balance(img: np.ndarray) -> np.ndarray:
    """
    Applies the Gray World algorithm for white balance correction.
    Corrects color casts caused by ambient webcam lighting.
    """
    if img is None or img.size == 0:
        return img
        
    # Calculate average values of each channel
    b, g, r = cv2.split(img.astype(np.float32))
    
    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)
    
    # Average of all channels
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    
    if mean_b == 0 or mean_g == 0 or mean_r == 0:
        return img
        
    # Scaling factors
    k_b = mean_gray / mean_b
    k_g = mean_gray / mean_g
    k_r = mean_gray / mean_r
    
    # Apply scales
    b = np.clip(b * k_b, 0, 255)
    g = np.clip(g * k_g, 0, 255)
    r = np.clip(r * k_r, 0, 255)
    
    result = cv2.merge((b, g, r)).astype(np.uint8)
    return result

def preprocess_skin_crop(img: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline for region images prior to model inference."""
    balanced = correct_white_balance(img)
    normalized = normalize_lighting(balanced)
    return normalized
