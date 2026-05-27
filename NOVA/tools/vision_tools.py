import asyncio
import cv2
from .base import Tool, ToolResult

class ScanFor(Tool):
    name = "scan_for"
    description = (
        "Scan the immediate area using the camera gimbal to look for a specific object. "
        "If the object is found, the tracker is AUTOMATICALLY locked onto it — "
        "you can call approach_tracked_target directly after this tool succeeds."
    )
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
        found_bbox_xyxy = None  # will hold [x1, y1, x2, y2] at frame resolution

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
                    x, y, w, h = bbox
                    # Convert (x, y, w, h) to [x1, y1, x2, y2]
                    found_bbox_xyxy = [x, y, x + w, y + h]

                    pose = self.context.odometry.get_pose()
                    self.context.memory.log_sighting(
                        object_name, pose[0], pose[1],
                        f"Seen at pan angle {angle}deg, bbox {bbox}."
                    )
                    # Broadcast bounding box to webapp overlay
                    if self.context.send_ws_callback:
                        fh, fw = frame.shape[:2]
                        self.context.send_ws_callback("nova_detection", {
                            "label": object_name,
                            "x": x / fw, "y": y / fh,
                            "w": w / fw, "h": h / fh,
                        })
                    break

        if self.context.robot:
            self.context.robot.gimbal.reset()

        if found and found_bbox_xyxy is not None:
            # ── Auto-lock tracker ─────────────────────────────────────────
            from ..perception.tracker import get_tracker
            tracker = get_tracker()

            # Grab a fresh frame at tracker resolution
            lock_frame = self.context.camera.get_latest_frame(downscale_to=(640, 360))
            if lock_frame is not None:
                # Scale bbox from original frame resolution to 640x360
                frame = self.context.camera.get_frame()
                if frame is not None:
                    fh, fw = frame.shape[:2]
                else:
                    fh, fw = 480, 640  # fallback

                sx, sy = 640.0 / fw, 360.0 / fh
                x1, y1, x2, y2 = found_bbox_xyxy
                tracker_bbox = [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]

                tracker.lock(tracker_bbox, lock_frame, label=object_name)
                print(f"[ScanFor] Auto-locked tracker on '{object_name}' at {tracker_bbox}")
                return ToolResult(True, f"Found {object_name}. Tracker LOCKED automatically.")
            else:
                return ToolResult(True, f"Found {object_name} but could not acquire lock frame.")
        else:
            return ToolResult(False, f"Could not find any {object_name} in the area.")