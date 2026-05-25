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

        bbox = self.context.get_vlm_bounding_box(vlm_frame, object_description)
        if not bbox:
            return ToolResult(False, f"Could not find '{object_description}' in the camera view.")

        # ── Guard 1: object must be sufficiently centred ─────────────────────
        # If the bbox centre is outside the middle 50% of the frame width, the
        # object is at the edge and partially out of view. Tracking a sliver is
        # unreliable and the stop condition can fire immediately on a false positive.
        # Return a clear failure so Nova can spin to centre the object first.
        _fh, _fw = vlm_frame.shape[:2]
        _bx, _by, _bw, _bh = bbox
        _box_cx = _bx + _bw / 2
        _left_limit  = _fw * 0.25
        _right_limit = _fw * 0.75
        if _box_cx < _left_limit or _box_cx > _right_limit:
            return ToolResult(
                False,
                f"Object is at frame edge (center_x={_box_cx:.0f}, frame_w={_fw}). "
                f"Use spin_search to center it before approaching."
            )

        # ── Guard 2: bbox must be large enough to track reliably ──────────────
        # A bbox narrower than 8% of frame width gives too few feature points
        # for LK optical flow. Reject it so Nova doesn't attempt a doomed track.
        MIN_BBOX_WIDTH_RATIO  = 0.08
        MIN_BBOX_HEIGHT_RATIO = 0.08
        if _bw < _fw * MIN_BBOX_WIDTH_RATIO or _bh < _fh * MIN_BBOX_HEIGHT_RATIO:
            return ToolResult(
                False,
                f"Object bbox too small to track reliably (w={_bw}px, h={_bh}px). "
                f"Move closer or use scan_for instead."
            )

        self.context.send_status(f"Found '{object_description}'. Locking tracker...")
        # Broadcast bounding box to webapp overlay
        if self.context.send_ws_callback:
            self.context.send_ws_callback("nova_detection", {
                "label": object_description,
                "x": _bx / _fw, "y": _by / _fh,
                "w": _bw / _fw, "h": _bh / _fh,
            })

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

        # Run at 60% of cruise speed while tracking.
        # Full cruise speed causes motion blur that kills LK optical flow.
        TRACKING_SPEED = int(self.context.config.navigation.cruise_speed * 0.6)

        # Deadzone: ignore errors smaller than 5% of frame width.
        # Prevents constant jittery corrections when the object is roughly centred.
        DEADZONE_PX = frame_w * 0.05

        stopped = False
        reason  = "Unknown"

        # Re-lock threshold: if the tracker retains fewer than this fraction of
        # its original points, trigger a VLM re-lock before full loss occurs.
        RELOCK_THRESHOLD = 0.20
        initial_points   = tracker.point_count()   # snapshot at init

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

            # ── VLM re-lock when tracker is degrading ─────────────────────────
            # If tracked points have dropped below RELOCK_THRESHOLD (20%) of the
            # original count, re-query the VLM now — while the rover is still close
            # enough to see the object — rather than waiting for full loss.
            if success and initial_points > 0:
                surviving = tracker.point_count()
                if surviving / initial_points < RELOCK_THRESHOLD:
                    print(f"[LockAndTrack] Tracker degrading ({surviving}/{initial_points} pts) — re-locking via VLM...")
                    self.context.robot.motor.halt()
                    relock_frame = self.context.camera.get_frame()
                    if relock_frame is not None:
                        new_bbox = self.context.get_vlm_bounding_box(relock_frame, object_description)
                        if new_bbox:
                            tracker.stop()
                            tracker = VisualTracker()
                            relock_bgr = _to_bgr(relock_frame)
                            if tracker.initialize(relock_bgr, new_bbox):
                                bbox = new_bbox
                                initial_points = tracker.point_count()
                                bgr_frame = relock_bgr
                                print(f"[LockAndTrack] Re-lock successful with {initial_points} points.")
                                if self.context.send_ws_callback:
                                    _bx, _by, _bw, _bh = new_bbox
                                    self.context.send_ws_callback("nova_detection", {
                                        "label": object_description,
                                        "x": _bx / frame_w, "y": _by / frame_h,
                                        "w": _bw / frame_w, "h": _bh / frame_h,
                                    })
                            else:
                                reason = "Re-lock failed — tracker could not reinitialize."
                                break
                        else:
                            reason = "Re-lock failed — VLM could not find object for re-lock."
                            break

            if not success:
                reason = "Lost track of object."
                break

            x, y, w, h = bbox

            # ── Stop condition ────────────────────────────────────────────────
            # Two conditions must BOTH be true to declare arrival:
            #   1. bbox bottom >= 75% of frame height  (close enough)
            #   2. bbox centre is within the middle 50% of frame width (centred)
            #
            # Requiring both prevents false positives when a partially-visible
            # edge object satisfies the bottom threshold while barely in frame.
            reached = False

            bbox_bottom  = y + h
            box_center_x = x + w / 2
            frame_bottom = frame_h * 0.75
            centre_left  = frame_w * 0.25
            centre_right = frame_w * 0.75

            is_close    = bbox_bottom  >= frame_bottom
            is_centred  = centre_left  <= box_center_x <= centre_right

            if is_close and is_centred:
                reached = True
                reason  = (
                    f"Reached object — bbox bottom at {bbox_bottom}px "
                    f"(threshold {frame_bottom:.0f}px), centred at x={box_center_x:.0f}."
                )
            elif is_close and not is_centred:
                # Close but not centred — keep steering, don't stop yet
                print(f"[LockAndTrack] Close but not centred (cx={box_center_x:.0f}) — continuing to steer.")

            if not reached:
                try:
                    dist_cm = self.context.obstacle_detector.last_distance_cm
                    stop_cm = self.context.config.navigation.obstacle_stop_cm * 2
                    if 0 < dist_cm < stop_cm:
                        reached = True
                        reason  = f"Reached object — ultrasonic distance {dist_cm:.0f}cm."
                except AttributeError:
                    pass

            if reached:
                self.context.robot.motor.halt()
                stopped = True
                break

            # Broadcast live tracker bbox to webapp overlay every frame
            if self.context.send_ws_callback:
                self.context.send_ws_callback("nova_detection", {
                    "label": object_description,
                    "x": x / frame_w, "y": y / frame_h,
                    "w": w / frame_w, "h": h / frame_h,
                })

            fill_ratio = (w * h) / (frame_w * frame_h)

            # Proportional steering.
            # Positive error → object is right of centre → increase left speed to turn right.
            # box_center_x already computed above in stop condition block.
            error = box_center_x - center_x

            # Apply deadzone — zero correction when error is small
            if abs(error) < DEADZONE_PX:
                turn_correction = 0
            else:
                # Scale so full frame-width offset → full turn_speed
                turn_correction = int(
                    (error / center_x) * self.context.config.navigation.turn_speed
                )

            max_s       = self.context.config.navigation.max_speed
            left_speed  = max(-max_s, min(max_s, TRACKING_SPEED + turn_correction))
            right_speed = max(-max_s, min(max_s, TRACKING_SPEED - turn_correction))

            print(
                f"[LockAndTrack] box_cx={box_center_x:.0f} err={error:.0f} "
                f"corr={turn_correction} L={left_speed} R={right_speed} "
                f"fill={fill_ratio:.2f} bbox_bottom={y+h}"
            )

            self.context.robot.motor.drive(left_speed, right_speed, 0)
            await asyncio.sleep(0.05)   # 20 Hz

        self.context.robot.motor.halt()
        tracker.stop()

        return ToolResult(stopped, reason)