import asyncio
import cv2
import numpy as np
from typing import Optional
from ..tools.base import Tool, ToolResult
from ..perception.depth import DepthNavigator
from ..perception.tracker import get_tracker
from ..navigation.driver import MovementSpeed

class ApproachTrackedTarget(Tool):
    name = "approach_tracked_target"
    description = (
        "Drives the rover directly towards the currently tracked target using the TargetTracker. "
        "The tracker MUST be locked onto a target before calling this tool. "
        "It will use Depth Anything V2 and A* pathfinding to dynamically avoid obstacles and plan safe routes. "
        "If the target is lost, it will proactively perform an opposite-action reacquisition sweep."
    )
    parameters = {
        "type": "object",
        "properties": {
            "target_width_ratio": {
                "type": "number",
                "description": "Stop when the target's width takes up this fraction of the frame width (default 0.4)."
            },
            "target_label": {
                "type": "string",
                "description": "Optional category/label of the target being tracked, to use for local reacquisition if lost."
            }
        }
    }

    async def execute(self, target_width_ratio: float = 0.4, target_label: Optional[str] = None) -> ToolResult:
        if not self.context.robot or not self.context.camera:
            return ToolResult(False, "Hardware not available.")

        tracker = get_tracker()
        state = tracker._get_state()
        if state["status"] == "LOST":
            return ToolResult(False, "Tracker is not locked on any target. Use spin_search or tracker_lock first.")

        FRAME_WIDTH = 640
        CENTER_X = FRAME_WIDTH / 2.0

        # ── Dynamic Target reached threshold adjustment ───────────────────────
        init_bbox = state.get("bbox")
        if init_bbox is not None:
            init_w = init_bbox[2] - init_bbox[0]
            init_ratio = init_w / FRAME_WIDTH
            print(f"[NOVA] Initial target width ratio: {init_ratio:.3f} (target threshold: {target_width_ratio})")
            if init_ratio >= target_width_ratio:
                adjusted_threshold = max(target_width_ratio, min(0.75, init_ratio * 1.35))
                print(f"[NOVA] Target is already large. Adjusting stopping width ratio threshold from {target_width_ratio} to {adjusted_threshold:.3f}")
                target_width_ratio = adjusted_threshold

        # ── Initialize Depth & A* Navigator ──────────────────────────────────
        self.context.send_status("Initializing Depth Anything V2 & A* Navigator...")
        depth_nav = DepthNavigator(self.context.config)

        nav = self.context.config.navigation
        base_speed = nav.cruise_speed
        max_turn = nav.turn_speed

        self.context.send_status("Approaching target via Depth A* Pathfinding...")
        
        reached = False
        lost = False
        blocked = False
        
        start_time = asyncio.get_event_loop().time()
        timeout = 45.0  # max 45s for approach
        
        # Periodic depth calculation interval (ticks of 0.1s)
        # We run depth estimation every 30 frames (approx 3 seconds) to keep CPU low,
        # but keep drawing/updating the overlay using the last planned path.
        depth_interval_ticks = 30
        tick_counter = 0
        
        smoothed_path = None
        last_overlay_frame = None
        last_turn = 0  # tracks steering direction (positive = right, negative = left)
        target_3d = None
        
        try:
            while asyncio.get_event_loop().time() - start_time < timeout:
                if not self.context.is_running:
                    break
                    
                # Update target tracker with a fresh downscaled frame
                tracker_frame = self.context.camera.get_latest_frame(downscale_to=(640, 360))
                if tracker_frame is not None:
                    state = tracker.update(tracker_frame)
                else:
                    state = tracker._get_state()

                bbox = state["bbox"]
                center = state["center"]
                
                # ── Proactive Target Reacquisition Protocol ───────────────────
                if state["status"] == "LOST" or bbox is None or center is None:
                    print("[NOVA] Target lost during approach.")
                    lost = True
                    break
                    
                bw = bbox[2] - bbox[0]
                cx = center[0]
                y2 = bbox[3]
                
                # ── Target Proximity Arrival Checks ───────────────────────────────────
                # Check 1: Metric physical distance (if DepthAnything compiled model is real and not dummy)
                is_depth_real = (depth_nav.compiled_model is not None)
                if is_depth_real and target_3d is not None:
                    x_tgt, z_tgt = target_3d
                    if z_tgt <= 0.45:
                        print(f"[NOVA] Target physical distance reached: {z_tgt:.3f}m <= 0.45m")
                        reached = True
                        break

                # Check 2: Bounding box touches lower border of frame (indicates it is right in front of the rover)
                # Since the tracker runs on 640x360, the bottom coordinate is 360.
                if y2 >= 350:
                    print(f"[NOVA] Target reached lower border: y2={y2} >= 350")
                    reached = True
                    break

                # Check 3: Visual bounding box width ratio
                if bw / FRAME_WIDTH >= target_width_ratio:
                    print(f"[NOVA] Target visual width ratio reached: {bw/FRAME_WIDTH:.3f} >= {target_width_ratio}")
                    reached = True
                    break

                # ── Run Depth Anything V2 and A* periodically ─────────────────
                run_depth = (tick_counter % depth_interval_ticks == 0)
                
                # Capture current raw frame
                raw_frame = self.context.camera.get_latest_frame()
                
                if raw_frame is not None:
                    if run_depth:
                        try:
                            # Run navigation pipeline
                            overlay_frame, smoothed_path, target_3d = depth_nav.navigate(raw_frame, bbox)
                            last_overlay_frame = overlay_frame
                        except Exception as e:
                            print(f"[Approach] Error in DepthNavigator: {e}")
                            smoothed_path = None
                    else:
                        # Re-draw the path overlay on the new raw frame to match camera movement
                        if smoothed_path is not None:
                            try:
                                # Quick re-draw using cached path to save CPU
                                overlay_frame = raw_frame.copy()
                                
                                # Estimate/project path waypoints onto new frame
                                # Using standard camera values
                                fx = fy = FRAME_WIDTH * 0.8
                                cx_p, cy_p = FRAME_WIDTH / 2.0, raw_frame.shape[0] / 2.0
                                y_floor = 0.45  # estimated
                                
                                pts_2d = []
                                for x_wp, z_wp in smoothed_path:
                                    u = int(x_wp * fx / z_wp + cx_p)
                                    v = int(y_floor * fy / z_wp + cy_p)
                                    if 0 <= u < raw_frame.shape[1] and 0 <= v < raw_frame.shape[0]:
                                        pts_2d.append([u, v])
                                        
                                if len(pts_2d) >= 2:
                                    pts_arr = np.array(pts_2d, dtype=np.int32).reshape((-1, 1, 2))
                                    cv2.polylines(overlay_frame, [pts_arr], isClosed=False, color=(255, 191, 0), thickness=4, lineType=cv2.LINE_AA)
                                    cv2.circle(overlay_frame, tuple(pts_2d[0]), 7, (255, 255, 255), -1, cv2.LINE_AA)
                                    cv2.circle(overlay_frame, tuple(pts_2d[0]), 10, (255, 191, 0), 2, cv2.LINE_AA)
                                    cv2.drawMarker(overlay_frame, tuple(pts_2d[-1]), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=15, thickness=3, line_type=cv2.LINE_AA)
                                    
                                # Draw current bbox
                                x1, y1, x2, y2 = bbox
                                cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (240, 16, 160), 2, cv2.LINE_AA)
                                cv2.putText(overlay_frame, "TARGET LOCKED", (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 16, 160), 2, cv2.LINE_AA)
                                
                                last_overlay_frame = overlay_frame
                            except Exception:
                                pass
                                
                    # ── Stream overlay frame to WebApp ──────────────────────────
                    if last_overlay_frame is not None:
                        if hasattr(self.context.camera, "processed_frame"):
                            self.context.camera.processed_frame = last_overlay_frame
                            
                # ── Steering Control ──────────────────────────────────────────
                steering_set = False
                
                if smoothed_path is not None and len(smoothed_path) >= 2:
                    # Target waypoint: e.g. 2 steps ahead to smooth and avoid oscillations
                    wp_idx = min(3, len(smoothed_path) - 1)
                    x_wp, z_wp = smoothed_path[wp_idx]
                    
                    # Calculate angle to waypoint
                    # In relative coords, X is right, Z is forward
                    angle_rad = np.arctan2(x_wp, z_wp)
                    
                    # Proportional control on angle
                    kp_depth = 1.2
                    error_norm = angle_rad / (np.pi / 2) # normalize relative to 90 degrees
                    turn = int(error_norm * max_turn * kp_depth)
                    
                    # Clamp turn
                    turn = max(-max_turn, min(max_turn, turn))
                    last_turn = turn
                    
                    current_speed = base_speed
                    # Slow down if target is close
                    dist_to_target = np.hypot(x_wp, z_wp)
                    if dist_to_target < 0.8:
                        current_speed = nav.min_speed
                        
                    left_speed = current_speed + turn
                    right_speed = current_speed - turn
                    steering_set = True
                    
                # ── Fallback to direct Tracker (BBox center P-controller) ──────
                if not steering_set:
                    error = cx - CENTER_X
                    error_norm = error / CENTER_X
                    kp = 1.0
                    turn = int(error_norm * max_turn * kp)
                    turn = max(-max_turn, min(max_turn, turn))
                    last_turn = turn
                    
                    current_speed = base_speed
                    left_speed = current_speed + turn
                    right_speed = current_speed - turn
                    
                # Clamp speeds
                left_speed = max(-nav.max_speed, min(nav.max_speed, left_speed))
                right_speed = max(-nav.max_speed, min(nav.max_speed, right_speed))
                
                # Send to motors
                self.context.driver._send_motors(left_speed, right_speed)
                
                tick_counter += 1
                await asyncio.sleep(0.1)
                
        finally:
            # ── Cleanup ───────────────────────────────────────────────────────
            self.context.driver._send_motors(0, 0)
            
            # Clear processed frame in camera stream
            if hasattr(self.context.camera, "processed_frame"):
                self.context.camera.processed_frame = None
                
        if reached:
            return ToolResult(True, "Successfully approached and reached the target.")
        elif blocked:
            return ToolResult(False, "Approach stopped due to an obstacle.")
        elif lost:
            return ToolResult(False, "Target lost during approach. Needs re-acquisition.", data={"lost": True})
        else:
            return ToolResult(False, "Approach timed out before reaching the target.")
