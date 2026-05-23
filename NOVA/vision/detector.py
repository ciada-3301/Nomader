"""
NOVA Vision Detector
--------------------
Handles object detection using YOLOv8 (or falls back to mock).
Estimates distance to objects based on bounding box size.
"""

import time
import math
from typing import List, Dict, Any

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class ObjectDetector:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.classes = {}
        
        # Approximate real-world heights of objects in meters (for distance heuristic)
        # Add common objects here
        self.known_heights = {
            "person": 1.7,
            "chair": 1.0,
            "bottle": 0.25,
            "cup": 0.1,
            "laptop": 0.25,
            "tv": 0.6,
            "cell phone": 0.15,
            "cat": 0.3,
            "dog": 0.5
        }
        
        # Camera focal length in pixels (approximate for Pi Camera 2 at 640x480)
        self.focal_length = 500.0 
        
        self._initialize_model()

    def _initialize_model(self):
        if YOLO_AVAILABLE:
            try:
                # Load the model (will download yolov8n.pt if not present)
                self.model = YOLO(f"{self.config.vision.detection_model}.pt")
                self.classes = self.model.names
                print(f"[NOVA Vision] Loaded {self.config.vision.detection_model}")
            except Exception as e:
                print(f"[NOVA Vision] Error loading YOLO: {e}")
                self.model = None
        else:
            print("[NOVA Vision] Ultralytics not installed. Using mock detector.")
            
    def detect(self, frame) -> List[Dict[str, Any]]:
        """
        Run detection on a single frame.
        Returns a list of dicts:
        {
            "class": "person",
            "confidence": 0.95,
            "bbox": [x1, y1, x2, y2],
            "center": (cx, cy),
            "distance_m": 2.5
        }
        """
        if frame is None:
            return []
            
        if self.model is None:
            return self._mock_detect()
            
        # Run inference
        results = self.model(frame, 
                             conf=self.config.vision.detection_confidence,
                             iou=self.config.vision.detection_iou,
                             imgsz=self.config.vision.detection_input_size,
                             verbose=False)
                             
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = self.classes.get(cls_id, "unknown")
                
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                h_px = y2 - y1
                
                # Estimate distance
                dist = self._estimate_distance(class_name, h_px)
                
                detections.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "center": (cx, cy),
                    "distance_m": dist
                })
                
        return detections
        
    def _estimate_distance(self, class_name: str, height_pixels: float) -> float:
        """Estimate distance to object using pinhole camera model."""
        real_h = self.known_heights.get(class_name, 0.5) # default 0.5m
        if height_pixels <= 0:
            return 99.9
            
        # Distance = (Real Height * Focal Length) / Pixel Height
        dist = (real_h * self.focal_length) / height_pixels
        return round(dist, 2)
        
    def _mock_detect(self):
        """Return fake detections for testing without model."""
        return []
