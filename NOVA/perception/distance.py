import json
import os
from pathlib import Path
from typing import Optional

KNOWN_SIZES = {
    "chair": 0.45,       # seat width in meters
    "table": 0.80,
    "door": 0.90,
    "person": 0.45,      # shoulder width
    "cup": 0.08,
    "bottle": 0.07,
    "laptop": 0.35,
    "tv": 0.80,
    "couch": 1.80,
    "bed": 1.40,
}

def get_focal_length() -> float:
    """
    Loads focal length from camera_calibration.json.
    Returns 800.0 as default if not found.
    """
    config_path = Path(__file__).parent.parent / "config" / "camera_calibration.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
                return float(data.get("focal_length_px", 800.0))
        except Exception as e:
            print(f"[Distance] Failed to load calibration: {e}")
    else:
        print("[Distance] Warning: camera_calibration.json not found. Using default focal length 800.0")
    
    return 800.0

def estimate_distance(bbox_width_px: int, real_width_m: float, focal_length_px: float) -> float:
    """
    Estimates distance to an object using its apparent width in pixels.
    Formula: distance = (real_width_m * focal_length_px) / bbox_width_px
    """
    if bbox_width_px <= 0:
        return 0.0
    return (real_width_m * focal_length_px) / bbox_width_px

def get_distance_for_class(class_name: str, bbox_width_px: int, focal_length_px: float = None) -> Optional[float]:
    """
    Convenience function that looks up the known size for a class name.
    Returns None if class is not in KNOWN_SIZES.
    """
    real_width_m = KNOWN_SIZES.get(class_name.lower())
    if real_width_m is None:
        return None
        
    if focal_length_px is None:
        focal_length_px = get_focal_length()
        
    return estimate_distance(bbox_width_px, real_width_m, focal_length_px)
