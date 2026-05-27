import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any

from ..events import global_event_bus, RobotEvent
from ..tools.base import Tool, ToolResult

class TargetTracker:
    def __init__(self):
        self.status = "LOST"
        self.bbox: Optional[List[int]] = None
        self.center: Optional[Tuple[int, int]] = None
        self.confidence = 0.0
        self.label: Optional[str] = None
        
        self.nano_tracker = None
        self.lk_points = None
        self.lk_initial_count = 0
        self.prev_gray = None
        
        # LK parameters
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)

    def lock(self, bbox: List[int], frame: np.ndarray, label: Optional[str] = None) -> None:
        """Initializes both LK and NanoTrack on the given bbox."""
        self.label = label
        if frame is None:
            return
            
        # bbox format: [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        if w <= 0 or h <= 0:
            print("[Tracker] Invalid bbox for lock")
            return
            
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize NanoTrack (requires opencv-contrib-python)
        try:
            self.nano_tracker = cv2.TrackerNano_create()
            self.nano_tracker.init(frame, (x1, y1, w, h))
        except Exception as e:
            print(f"[Tracker] Error initializing NanoTrack: {e}")
            self.nano_tracker = None
            
        # Initialize LK feature points inside the bbox
        mask = np.zeros_like(self.prev_gray)
        mask[y1:y2, x1:x2] = 255
        
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, **self.feature_params)
        
        if p0 is not None:
            self.lk_points = p0
            self.lk_initial_count = len(p0)
        else:
            self.lk_points = None
            self.lk_initial_count = 0
            
        self.bbox = bbox
        self.center = (x1 + w//2, y1 + h//2)
        self.status = "TRACKING"
        self.confidence = 1.0
        print(f"[Tracker] Locked on target at {self.bbox}")

    def update(self, frame: np.ndarray) -> Dict[str, Any]:
        """Updates tracker state with the new frame."""
        if self.status == "LOST" or frame is None:
            return self._get_state()
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        use_nano = False
        
        # 1. Try LK Tracking first
        if self.lk_points is not None and self.lk_initial_count > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.lk_points, None, **self.lk_params)
            
            if p1 is not None:
                good_new = p1[st == 1]
                
                # Confidence is the ratio of surviving points
                self.confidence = len(good_new) / self.lk_initial_count
                
                if self.confidence >= 0.4 and len(good_new) > 0:
                    # Update bbox based on point motion (simple bounding box around points)
                    x, y, w, h = cv2.boundingRect(np.int32(good_new))
                    self.bbox = [x, y, x + w, y + h]
                    self.center = (x + w//2, y + h//2)
                    self.status = "TRACKING"
                    self.lk_points = good_new.reshape(-1, 1, 2)
                    self.prev_gray = gray
                else:
                    use_nano = True
            else:
                use_nano = True
        else:
            use_nano = True
            
        # 2. Fallback to NanoTrack
        if use_nano and self.nano_tracker is not None:
            success, bbox_nano = self.nano_tracker.update(frame)
            if success:
                # NanoTrack returns (x, y, w, h)
                nx, ny, nw, nh = [int(v) for v in bbox_nano]
                self.bbox = [nx, ny, nx + nw, ny + nh]
                self.center = (nx + nw//2, ny + nh//2)
                self.status = "DEGRADED"
                # We don't have a direct confidence score from NanoTrack standard API easily available,
                # but we will assume it's valid if success is true. If we want an explicit score,
                # we'd need to extract it (not always available in simple python bindings).
                # We'll set a synthetic confidence for DEGRADED state.
                self.confidence = 0.5
                self.prev_gray = gray
                
                # Re-initialize LK from NanoTrack bbox to try and recover LK tracking
                mask = np.zeros_like(gray)
                mask[max(0, ny):ny+nh, max(0, nx):nx+nw] = 255
                p0 = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
                if p0 is not None:
                    self.lk_points = p0
                    self.lk_initial_count = len(p0)
            else:
                self._mark_lost()
        elif use_nano:
            self._mark_lost()
            
        return self._get_state()

    def _mark_lost(self):
        self.status = "LOST"
        self.bbox = None
        self.center = None
        self.confidence = 0.0
        global_event_bus.emit(RobotEvent.TARGET_LOST, {"reason": "tracking_failed"})

    def release(self) -> None:
        """Clears tracker state."""
        self.status = "LOST"
        self.bbox = None
        self.center = None
        self.confidence = 0.0
        self.label = None
        self.nano_tracker = None
        self.lk_points = None
        self.prev_gray = None

    def _get_state(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "bbox": self.bbox,
            "center": self.center,
            "confidence": self.confidence,
            "label": self.label
        }

# Singleton instance
_global_tracker: Optional[TargetTracker] = None

def get_tracker() -> TargetTracker:
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TargetTracker()
    return _global_tracker

# ── LangGraph Tools ──────────────────────────────────────────────────────────

class TrackerLockTool(Tool):
    name = "tracker_lock"
    description = "Locks the target tracker onto a specific bounding box [x1, y1, x2, y2]."
    parameters = {
        "type": "object",
        "properties": {
            "bbox": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Bounding box coordinates [x1, y1, x2, y2]"
            }
        },
        "required": ["bbox"]
    }

    async def execute(self, bbox: List[int]) -> ToolResult:
        # Check current movement speed
        if hasattr(self.context, 'driver') and hasattr(self.context.driver, 'get_current_speed'):
            from ..navigation.driver import MovementSpeed
            current_speed = self.context.driver.get_current_speed()
            if current_speed not in (MovementSpeed.STOPPED, MovementSpeed.SLOW_CRAWL):
                return ToolResult(success=False, message="Cannot lock tracker while moving fast. Switch to SLOW_CRAWL or STOPPED.")
                
        if not self.context.camera:
            return ToolResult(success=False, message="Camera not available")
            
        frame = self.context.camera.get_latest_frame(downscale_to=(640, 360))
        if frame is None:
            return ToolResult(success=False, message="Could not acquire frame")
            
        tracker = get_tracker()
        tracker.lock(bbox, frame)
        return ToolResult(success=True, message=f"Tracker locked on {bbox}")

class TrackerStatusTool(Tool):
    name = "tracker_status"
    description = "Returns the current state of the target tracker."
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        tracker = get_tracker()
        state = tracker._get_state()
        return ToolResult(success=True, message=f"Tracker status: {state['status']}", data=state)

class TrackerReleaseTool(Tool):
    name = "tracker_release"
    description = "Releases the current tracker lock."
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        tracker = get_tracker()
        tracker.release()
        return ToolResult(success=True, message="Tracker released.")
