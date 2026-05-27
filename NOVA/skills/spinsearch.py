import asyncio
from typing import List, Optional, Dict, Any
import time

from ..tools.base import Tool, ToolResult

SPIN_STEP_DURATION_MS = 400

class SpinSearch(Tool):
    name = "spin_search"
    description = (
        "Rotates the robot in discrete steps of 60 degrees to search for an object using the camera. "
        "Use this when the target is not in the current camera view."
    )
    parameters = {
        "type": "object",
        "properties": {
            "class_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of object names to search for."
            }
        },
        "required": ["class_names"]
    }

    async def execute(self, class_names: List[str]) -> ToolResult:
        if not class_names:
            return ToolResult(success=False, message="No class names provided.")

        if not self.context.robot or not self.context.camera:
            return ToolResult(success=False, message="Hardware not available.")

        # Ensure we have the detector and tracker
        from ..perception.detector import get_detector
        from ..perception.tracker import get_tracker
        from ..perception.distance import get_distance_for_class
        
        detector = get_detector()
        tracker = get_tracker()
        nav = self.context.config.navigation
        speed = nav.turn_speed

        steps = 6
        rotation_duration_s = SPIN_STEP_DURATION_MS / 1000.0
        
        for step in range(steps):
            if not self.context.is_running:
                break
                
            # Stop motors completely
            self.context.driver._send_motors(0, 0)
            
            # Wait for blur to settle
            await asyncio.sleep(0.3)
            
            # Grab sharpest frame
            frame = self.context.camera.get_sharpest_frame(downscale_to=(416, 234))
            
            if frame is not None:
                detections = detector.detect(frame, class_names)
                if detections:
                    # Found target
                    best_det = detections[0]
                    # Estimate distance if possible
                    dist = get_distance_for_class(best_det['class'], best_det['bbox'][2] - best_det['bbox'][0])
                    
                    result_data = {
                        "found": True,
                        "class_name": best_det['class'],
                        "bbox": best_det['bbox'],
                        "confidence": best_det['confidence'],
                        "estimated_distance": dist
                    }
                    
                    # Immediately lock tracker on full res frame
                    full_frame = self.context.camera.get_latest_frame(downscale_to=(640, 360))
                    if full_frame is not None:
                        # Scale bbox to tracker resolution (416x234 -> 640x360)
                        sx, sy = 640/416.0, 360/234.0
                        bx1, by1, bx2, by2 = best_det['bbox']
                        tracker_bbox = [int(bx1*sx), int(by1*sy), int(bx2*sx), int(by2*sy)]
                        tracker.lock(tracker_bbox, full_frame)
                        
                    return ToolResult(
                        success=True, 
                        message=f"Found {best_det['class']} during spin search.", 
                        data=result_data
                    )
            
            # Spin next step
            self.context.driver._send_motors(speed, -speed)
            start = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start < rotation_duration_s:
                if not self.context.is_running:
                    break
                await asyncio.sleep(0.05)
                
            self.context.driver._send_motors(0, 0)
            
        return ToolResult(success=True, message="Spin search completed (360 degrees), target not found.", data={"found": False})
