import asyncio
import time
import cv2
from .base import Tool, ToolResult
from ..vision.tracker import VisualTracker


def _to_bgr(frame):
    """
    Picamera2 is configured as RGB888 but OpenCV (and all trackers) expect BGR.
    WebcamHelper already outputs BGR natively after the fix in camera.py.
    This helper converts only when the frame looks like it came from Picamera2
    (i.e. the camera interface exposes an `is_rgb` flag, or we check the source).
    Since we can't easily tell at runtime, callers set needs_bgr_convert explicitly.
    """
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


class ScanFor(Tool):
    name = "scan_for"
    description = "Scan the immediate area using the camera to look for a specific object."
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
        best_conf = 0.0
        best_dist = 99.0

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
                detections = self.context.detector.detect(frame)
                for det in detections:
                    if object_name.lower() in det["class"].lower():
                        found = True
                        if det["confidence"] > best_conf:
                            best_conf = det["confidence"]
                            best_dist = det["distance_m"]

                            pose = self.context.odometry.get_pose()
                            self.context.memory.log_sighting(
                                object_name, pose[0], pose[1],
                                f"Seen at distance {best_dist}m, angle {angle}."
                            )

        if self.context.robot:
            self.context.robot.gimbal.reset()

        if found:
            return ToolResult(True, f"Found {object_name} at distance {best_dist}m (confidence: {best_conf:.2f}).")
        else:
            return ToolResult(False, f"Could not find any {object_name} in the area.")


class ScanRoom(Tool):
    name = "scan_room"
    description = "Perform a 360 degree scan to map obstacles and catalog all visible objects."
    parameters = {
        "type": "object",
        "properties": {}
    }

    async def execute(self) -> ToolResult:
        self.context.send_status("Scanning room...")

        objects_found = set()

        if self.context.robot:
            speed = self.context.config.navigation.turn_speed
            self.context.robot.motor.drive(speed, -speed, 0)

            for _ in range(12):
                if not self.context.is_running:
                    break

                await asyncio.sleep(0.5)
                frame = self.context.camera.get_frame()
                if frame is not None:
                    detections = self.context.detector.detect(frame)
                    for det in detections:
                        objects_found.add(det["class"])

            self.context.robot.motor.halt()

        return ToolResult(True, f"Scan complete. Objects detected: {', '.join(objects_found) if objects_found else 'None'}.")


class LockAndTrack(Tool):
    name = "lock_and_track"
    description = "Use the Vision model to find an arbitrary object, place a tracker on it, and drive towards it until reaching it. Use this for approaching non-standard objects."
    parameters = {
        "type": "object",
        "properties": {
            "object_description": {"type": "string", "description": "The description of the object to find and track (e.g., 'the wooden door', 'the red chair')."},
            "screen_fill_threshold": {"type": "number", "description": "Percentage (0.0 to 1.0) of the screen the bounding box should fill before stopping. Default is 0.4 (40%)."}
        },
        "required": ["object_description"]
    }

    async def execute(self, object_description: str, screen_fill_threshold: float = 0.4) -> ToolResult:
        if not self.context.camera or not self.context.robot:
            return ToolResult(False, "Camera or Robot not connected.")

        # ── Step 1: Ask VLM where the object is ───────────────────────────────
        self.context.send_status(f"Looking for '{object_description}'...")
        vlm_frame = self.context.camera.get_frame()
        if vlm_frame is None:
            return ToolResult(False, "Failed to get camera frame.")

        bbox = self.context.planner.get_vlm_bounding_box(vlm_frame, object_description)
        if not bbox:
            return ToolResult(False, f"Could not find '{object_description}' in the camera view.")

        self.context.send_status(f"Found '{object_description}'. Locking tracker...")

        # ── Step 2: Grab a FRESH frame for tracker initialisation ─────────────
        # The VLM call is a synchronous LLM round-trip that can take 2-5 seconds.
        # Initialising the tracker on the pre-VLM stale frame means the appearance
        # model is built on a frame that no longer matches the live scene.
        # The bbox coordinates are still valid (the doll hasn't moved), so we just
        # need to update the frame underneath them.
        init_frame = self.context.camera.get_frame()
        if init_frame is None:
            # Camera dropped between VLM call and now — fall back to the VLM frame.
            # Better than failing outright; tracker may still init.
            print("[LockAndTrack] Warning: could not get fresh frame; falling back to VLM frame.")
            init_frame = vlm_frame

        # ── Step 3: Convert RGB→BGR for OpenCV tracker ────────────────────────
        # Picamera2 is configured with format="RGB888". OpenCV trackers (CSRT, KCF,
        # MIL, etc.) require BGR. Without this conversion tracker.init() can return
        # False or produce garbage tracks even when the ROI is perfectly correct.
        init_frame_bgr = _to_bgr(init_frame)

        # ── Step 4: Initialise tracker ────────────────────────────────────────
        tracker = VisualTracker()
        if not tracker.initialize(init_frame_bgr, bbox):
            return ToolResult(False, "Failed to initialize visual tracker.")

        self.context.send_status(f"Tracking '{object_description}'. Approaching...")

        # Cache frame dimensions from the init frame (consistent for the session).
        frame_h, frame_w = init_frame_bgr.shape[:2]
        center_x = frame_w / 2

        stopped = False
        reason = "Unknown"

        # ── Step 5: Tracking + approach loop ─────────────────────────────────
        while self.context.is_running:
            raw_frame = self.context.camera.get_frame()
            if raw_frame is None:
                await asyncio.sleep(0.05)
                continue

            # Keep BGR conversion consistent throughout the loop —
            # tracker.update must receive the same colour space as tracker.init.
            bgr_frame = _to_bgr(raw_frame)

            success, bbox = tracker.update(bgr_frame)

            if not success:
                reason = "Lost track of object."
                break

            x, y, w, h = bbox
            fill_ratio = (w * h) / (frame_w * frame_h)

            if fill_ratio >= screen_fill_threshold:
                stopped = True
                reason = f"Reached object (fills {fill_ratio*100:.1f}% of view)."
                break

            # Proportional steering: positive error → object is right → turn right
            box_center_x = x + w / 2
            error = box_center_x - center_x
            turn_correction = int((error / frame_w) * self.context.config.navigation.turn_speed * 1.5)

            speed = self.context.config.navigation.cruise_speed
            max_s = self.context.config.navigation.max_speed
            left_speed  = max(-max_s, min(max_s, speed + turn_correction))
            right_speed = max(-max_s, min(max_s, speed - turn_correction))

            self.context.robot.motor.drive(right_speed, left_speed, 0)
            await asyncio.sleep(0.05)   # 20 Hz

        self.context.robot.motor.halt()
        tracker.stop()

        return ToolResult(stopped, reason)