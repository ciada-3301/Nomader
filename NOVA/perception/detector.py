from typing import List, Dict, Optional, Any
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("[Detector] Warning: ultralytics not installed. Detector will not work.")
    YOLO = None

from ..tools.base import Tool, ToolResult

class GroundingDetector:
    """
    Open-vocabulary object detector using YOLOWorld-S.
    """
    def __init__(self, confidence_threshold: float = 0.35):
        self.threshold = confidence_threshold
        self.model = None
        if YOLO:
            print("[Detector] Loading YOLOWorld-S model...")
            # Load the model only once
            self.model = YOLO("yolov8s-worldv2.pt")
            
    def detect(self, frame: np.ndarray, class_names: List[str]) -> List[Dict[str, Any]]:
        """
        Detects objects in the provided frame matching the class names.
        """
        if self.model is None or frame is None:
            return []
            
        # Set classes for open-vocabulary detection
        self.model.set_classes(class_names)
        
        # Run inference
        results = self.model(frame, verbose=False)
        
        if not results:
            return []
            
        result = results[0]
        boxes = result.boxes
        
        detections = []
        h, w = frame.shape[:2]
        
        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            if conf < self.threshold:
                continue
                
            cls_idx = int(boxes.cls[i])
            if cls_idx >= len(class_names):
                continue
                
            class_name = class_names[cls_idx]
            
            # xyxy returns [x1, y1, x2, y2]
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Normalize coordinates to 0-1
            x1n, y1n, x2n, y2n = x1/w, y1/h, x2/w, y2/h
            
            detections.append({
                "class": class_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "bbox_normalized": [x1n, y1n, x2n, y2n]
            })
            
        return detections

# Instantiate singleton for the app if needed, or agent will instantiate
_global_detector: Optional[GroundingDetector] = None

def get_detector() -> GroundingDetector:
    global _global_detector
    if _global_detector is None:
        _global_detector = GroundingDetector()
    return _global_detector

class GroundingDetectorTool(Tool):
    name = "grounding_detector"
    description = "Detect objects in the current camera view using an open-vocabulary model. Provide a list of class names (noun phrases) to find."
    parameters = {
        "type": "object",
        "properties": {
            "class_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of object names to search for (e.g., ['chair', 'red cup', 'door'])"
            }
        },
        "required": ["class_names"]
    }

    async def execute(self, class_names: List[str]) -> ToolResult:
        if not class_names:
            return ToolResult(success=False, message="No class names provided.")
            
        # Grab frame from camera via agent context
        if not self.context.camera:
            return ToolResult(success=False, message="Camera not initialized.")
            
        frame = self.context.camera.get_latest_frame(downscale_to=(416, 234))
        if frame is None:
            return ToolResult(success=False, message="Could not acquire camera frame.")
            
        detector = get_detector()
        detections = detector.detect(frame, class_names)
        
        if not detections:
            return ToolResult(success=False, message="TargetNotFound")
            
        # Return the detections
        info_strs = [f"{d['class']} ({d['confidence']:.2f}) at {d['bbox']}" for d in detections]
        return ToolResult(
            success=True, 
            message=f"Found {len(detections)} targets: " + "; ".join(info_strs),
            data=detections
        )
