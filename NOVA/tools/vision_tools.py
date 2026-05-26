import asyncio
import cv2
from .base import Tool, ToolResult

class ScanFor(Tool):
    name = "scan_for"
    description = "Scan the immediate area using the camera to look for a specific object. Use this to quickly look around without driving."
    parameters = {
        "type": "object",
        "properties": {
            "object_name": {"type": "string", "description": "The name of the object to look for (e.g. 'person', 'cup')."}
        },
        "required": ["object_name"]
    }

    async def execute(self, object_name: str) -> ToolResult:
        self.context.send_status(f"Scanning for {object_name}...")

        found = False

        # Simple pan scan
        pan_angles = [90, 60, 30, 60, 90, 120, 150, 120, 90]

        for angle in pan_angles:
            if not self.context.is_running:
                break

            if self.context.robot:
                self.context.robot.gimbal.pan(angle)
                self.context.robot.gimbal.tilt(90)
            await asyncio.sleep(0.5)

            frame = self.context.camera.get_frame()
            if frame is not None:
                bbox = self.context.get_vlm_bounding_box(frame, object_name)
                if bbox is not None:
                    found = True
                    pose = self.context.odometry.get_pose()
                    self.context.memory.log_sighting(
                        object_name, pose[0], pose[1],
                        f"Seen at pan angle {angle}deg, bbox {bbox}."
                    )
                    # Broadcast bounding box to webapp overlay
                    if self.context.send_ws_callback:
                        x, y, w, h = bbox
                        fh, fw = frame.shape[:2]
                        self.context.send_ws_callback("nova_detection", {
                            "label": object_name,
                            "x": x / fw, "y": y / fh,
                            "w": w / fw, "h": h / fh,
                        })
                    break

        if self.context.robot:
            self.context.robot.gimbal.reset()

        if found:
            return ToolResult(True, f"Found {object_name}.")
        else:
            return ToolResult(False, f"Could not find any {object_name} in the area.")