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

        # Ensure we have the tracker
        from ..perception.tracker import get_tracker
        from ..perception.distance import get_distance_for_class
        
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
            
            # Grab sharpest frame (downscale to tracker resolution to match VLM/Tracker dimensions)
            frame = self.context.camera.get_sharpest_frame(downscale_to=(640, 360))
            
            if frame is not None:
                best_det = None
                for class_name in class_names:
                    # Query cloud VLM for bounding box
                    bbox = self.context.get_vlm_bounding_box(frame, class_name)
                    if bbox is not None:
                        x, y, w, h = bbox
                        best_det = {
                            "class": class_name,
                            "bbox": [x, y, x + w, y + h]
                        }
                        break  # Found one of the targets

                if best_det is not None:
                    # Estimate distance if possible
                    dist = get_distance_for_class(best_det['class'], best_det['bbox'][2] - best_det['bbox'][0])
                    
                    result_data = {
                        "found": True,
                        "class_name": best_det['class'],
                        "bbox": best_det['bbox'],
                        "confidence": 1.0,
                        "estimated_distance": dist
                    }
                    
                    # Immediately lock tracker on a fresh tracker-resolution frame
                    full_frame = self.context.camera.get_latest_frame(downscale_to=(640, 360))
                    if full_frame is not None:
                        # bbox is already in 640x360 space, so no scaling is needed
                        tracker.lock(best_det['bbox'], full_frame, label=best_det['class'])
                        
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
