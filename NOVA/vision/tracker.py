"""
NOVA Visual Tracker  (V2 ported from V1)
------------------------------------------
Lucas-Kanade sparse optical flow tracker.

Replaces OpenCV contrib CSRT/KCF which:
 Requires opencv-contrib (often absent on ARM Pi builds)
 Fails silently when the contrib module is stale/mismatched
 Produces garbage tracks under motion blur from the moving rover

How it works
------------
Init:
  - Crop the ROI from the init frame
  - Run cv2.goodFeaturesToTrack inside the ROI to find strong corners
  - Store those points + the current bbox

Update:
  - Run cv2.calcOpticalFlowPyrLK to find where each point moved
  - Keep only points the algorithm is confident about
  - Take the MEDIAN dx, dy across surviving points
    (median ignores outliers from background bleed / partial occlusion)
  - Shift the bbox by (median_dx, median_dy)
  - Periodically re-seed points from the updated ROI so the cloud never dies

Loss detection:
  - If surviving points drop below MIN_POINT_RATIO of the original seed count,
    declare tracking lost
"""

import cv2
import numpy as np


# -- Tuning constants -----------------------------------------------------------

MAX_CORNERS     = 200   # max feature points to seed from the ROI
QUALITY_LEVEL   = 0.01  # minimum corner quality (lower ? more points)
MIN_DISTANCE_PX = 5     # minimum pixel gap between corners

LK_PARAMS = dict(
    winSize  = (21, 21),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

MIN_POINT_RATIO = 0.20  # lose track if < 20 % of seed points survive
RESEED_INTERVAL = 10    # re-detect features every N frames
MIN_SEED_POINTS = 6     # abort init if fewer than this many corners found


# -- Helpers --------------------------------------------------------------------

def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _seed_points(gray: np.ndarray, bbox) -> np.ndarray | None:
    """
    Detect strong corners inside *bbox* on the given grayscale frame.
    Returns an (N,1,2) float32 array in full-frame coordinates,
    or None if too few points were found.
    """
    x, y, w, h = (int(v) for v in bbox)
    roi_gray    = gray[y : y + h, x : x + w]

    if roi_gray.size == 0:
        return None

    pts = cv2.goodFeaturesToTrack(
        roi_gray,
        maxCorners   = MAX_CORNERS,
        qualityLevel = QUALITY_LEVEL,
        minDistance  = MIN_DISTANCE_PX,
    )

    if pts is None or len(pts) < MIN_SEED_POINTS:
        return None

    # Shift from ROI-local full-frame coordinates
    pts[:, 0, 0] += x
    pts[:, 0, 1] += y
    return pts.astype(np.float32)


# -- Tracker class --------------------------------------------------------------

class VisualTracker:
    """
    Lucas-Kanade optical flow tracker.

    Drop-in replacement for the old OpenCV contrib CSRT tracker.
    Same public API:
        tracker = VisualTracker()
        ok = tracker.initialize(bgr_frame, (x, y, w, h))
        ok, bbox = tracker.update(bgr_frame)
        tracker.stop()
    """

    def __init__(self):
        self.is_tracking  = False
        self._bbox        = None   # (x, y, w, h) — plain Python ints
        self._pts         = None   # (N,1,2) float32 current point cloud
        self._n_seed      = 0      # number of points at last seed
        self._prev_gray   = None   # grayscale of previous frame
        self._frame_count = 0      # frames processed since last reseed

    # -- Public API -------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox) -> bool:
        """
        Seed the tracker.

        Args:
            frame: BGR uint8 frame — must be C-contiguous.
            bbox:  (x, y, w, h) in pixels.

        Returns:
            True on success, False if the frame/bbox is unusable.
        """
        self.is_tracking = False

        # -- Validate & normalise frame ----------------------------------------
        if frame is None:
            print("[NOVA Tracker] Init failed: frame is None.")
            return False

        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        if frame.dtype != np.uint8:
            print(f"[NOVA Tracker] Converting dtype {frame.dtype} ? uint8.")
            frame = frame.astype(np.uint8)

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # -- Validate bbox -----------------------------------------------------
        x, y, w, h = (int(v) for v in bbox)
        fh, fw     = frame.shape[:2]

        if w <= 0 or h <= 0:
            print(f"[NOVA Tracker] Init failed: non-positive bbox ({w}×{h}).")
            return False

        if x < 0 or y < 0 or (x + w) > fw or (y + h) > fh:
            print(f"[NOVA Tracker] Init failed: bbox ({x},{y},{w},{h}) outside {fw}×{fh}.")
            return False

        # -- Seed feature points -----------------------------------------------
        gray = _to_gray(frame)
        pts  = _seed_points(gray, (x, y, w, h))

        if pts is None:
            print(
                f"[NOVA Tracker] Init failed: not enough feature points in ROI "
                f"({x},{y},{w},{h}). The region may be textureless."
            )
            return False

        self._bbox        = (x, y, w, h)
        self._pts         = pts
        self._n_seed      = len(pts)
        self._prev_gray   = gray
        self._frame_count = 0
        self.is_tracking  = True

        print(
            f"[NOVA Tracker] Initialized with {self._n_seed} points "
            f"in bbox ({x},{y},{w},{h})."
        )
        return True

    def update(self, frame: np.ndarray):
        """
        Advance the tracker by one frame.

        Returns:
            (True,  (x, y, w, h))  — tracking healthy
            (False, None)           — tracking lost
        """
        if not self.is_tracking or self._pts is None:
            return False, None

        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        gray = _to_gray(frame)

        # -- Sparse optical flow -----------------------------------------------
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._pts, None, **LK_PARAMS
        )

        if new_pts is None or status is None:
            print("[NOVA Tracker] Optical flow returned no points — lost.")
            self.is_tracking = False
            return False, None

        good_new = new_pts[status.ravel() == 1]
        good_old = self._pts [status.ravel() == 1]

        survival_ratio = len(good_new) / self._n_seed if self._n_seed > 0 else 0.0

        if len(good_new) < MIN_SEED_POINTS or survival_ratio < MIN_POINT_RATIO:
            print(
                f"[NOVA Tracker] Lost — {len(good_new)}/{self._n_seed} points "
                f"survived ({survival_ratio * 100:.0f}%)."
            )
            self.is_tracking = False
            return False, None

        # -- Median shift ? new bbox -------------------------------------------
        delta      = good_new - good_old
        median_dx  = float(np.median(delta[:, 0, 0]))
        median_dy  = float(np.median(delta[:, 0, 1]))

        x, y, w, h = self._bbox
        x = int(round(x + median_dx))
        y = int(round(y + median_dy))

        # Clamp to frame bounds
        fh, fw = gray.shape[:2]
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = max(1, min(w, fw - x))
        h = max(1, min(h, fh - y))

        self._bbox       = (x, y, w, h)
        self._pts        = good_new
        self._prev_gray  = gray
        self._frame_count += 1

        # -- Periodic reseed ---------------------------------------------------
        if self._frame_count % RESEED_INTERVAL == 0:
            fresh = _seed_points(gray, self._bbox)
            if fresh is not None:
                self._pts    = fresh
                self._n_seed = len(fresh)

        return True, (x, y, w, h)

    def point_count(self) -> int:
        """Return the number of currently tracked feature points."""
        if self._pts is None:
            return 0
        return len(self._pts)

    def stop(self):
        self.is_tracking = False
        self._pts        = None
        self._prev_gray  = None
        self._bbox       = None
