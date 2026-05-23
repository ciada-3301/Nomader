"""
NOVA Visual Tracker
-------------------
Wrapper for high-speed OpenCV tracking (CSRT/KCF).

Changes from v1:
  - Diagnostics: prints frame shape, dtype, contiguity, bbox types on init failure
  - Handles OpenCV 4.5+ legacy API migration (TrackerCSRT moved to cv2.legacy)
  - Ensures frame is C-contiguous and 3-channel before passing to tracker
  - Falls back through CSRT -> KCF -> legacy variants automatically
"""

import cv2
import numpy as np


def _make_tracker():
    """
    Create the best available CSRT (or KCF) tracker on this OpenCV build.

    OpenCV 4.5+ moved the tracking API to cv2.legacy. Both the old and new
    locations are tried so the code works across all Pi OpenCV builds without
    requiring a specific version.

    Returns the tracker object, or raises RuntimeError if none is found.
    """
    candidates = [
        ("CSRT (main)",    lambda: cv2.TrackerCSRT_create()),
        ("CSRT (legacy)",  lambda: cv2.legacy.TrackerCSRT_create()),
        ("KCF (main)",     lambda: cv2.TrackerKCF_create()),
        ("KCF (legacy)",   lambda: cv2.legacy.TrackerKCF_create()),
    ]

    for name, factory in candidates:
        try:
            tracker = factory()
            print(f"[NOVA Tracker] Using {name}.")
            return tracker
        except (AttributeError, cv2.error):
            continue

    raise RuntimeError(
        "No OpenCV tracker available. "
        "Install opencv-contrib-python: pip install opencv-contrib-python"
    )


class VisualTracker:
    def __init__(self):
        self.tracker = None
        self.is_tracking = False

    def initialize(self, frame, bbox):
        """
        Initialize the tracker with a frame and bounding box.
        bbox format: (x, y, w, h) — all plain Python ints.
        """
        # ── Pre-flight diagnostics ─────────────────────────────────────────────
        if frame is None:
            print("[NOVA Tracker] Init failed: frame is None.")
            return False

        # Ensure C-contiguous memory layout — some transformed frames are not,
        # and OpenCV's C++ binding silently returns False on non-contiguous input.
        if not frame.flags["C_CONTIGUOUS"]:
            print("[NOVA Tracker] Frame is not C-contiguous — copying.")
            frame = np.ascontiguousarray(frame)

        # Tracker requires exactly 3 channels (BGR).
        if frame.ndim != 3 or frame.shape[2] != 3:
            print(f"[NOVA Tracker] Init failed: expected 3-channel frame, got shape {frame.shape}.")
            return False

        # Ensure 8-bit unsigned — some capture pipelines give uint16 or float.
        if frame.dtype != np.uint8:
            print(f"[NOVA Tracker] Frame dtype is {frame.dtype}, converting to uint8.")
            frame = frame.astype(np.uint8)

        # Ensure bbox values are plain Python ints (not numpy.int64).
        x, y, w, h = (int(v) for v in bbox)
        fh, fw = frame.shape[:2]

        if w <= 0 or h <= 0:
            print(f"[NOVA Tracker] Init failed: non-positive bbox dimensions ({w}×{h}).")
            return False

        if x < 0 or y < 0 or (x + w) > fw or (y + h) > fh:
            print(
                f"[NOVA Tracker] Init failed: bbox ({x},{y},{w},{h}) "
                f"out of frame bounds ({fw}×{fh})."
            )
            return False

        print(
            f"[NOVA Tracker] Initialising — "
            f"frame: {fw}×{fh} {frame.dtype} {'C' if frame.flags['C_CONTIGUOUS'] else 'non-C'}-contiguous, "
            f"bbox: ({x},{y},{w},{h}), "
            f"bbox types: {[type(v).__name__ for v in (x, y, w, h)]}"
        )

        # ── Create and initialise tracker ──────────────────────────────────────
        try:
            self.tracker = _make_tracker()
            success = self.tracker.init(frame, (x, y, w, h))
            self.is_tracking = success

            if not success:
                print(
                    "[NOVA Tracker] tracker.init() returned False despite valid inputs. "
                    "This usually means the OpenCV tracking module is partially broken "
                    "on this build. Try: pip install --upgrade opencv-contrib-python"
                )
            return success

        except RuntimeError as e:
            print(f"[NOVA Tracker] {e}")
            self.is_tracking = False
            return False
        except Exception as e:
            print(f"[NOVA Tracker] Unexpected init error: {e}")
            self.is_tracking = False
            return False

    def update(self, frame):
        """
        Update tracker with new frame.
        Returns (success, bbox) where bbox is (x, y, w, h) or None.
        """
        if not self.is_tracking or self.tracker is None:
            return False, None

        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        success, bbox = self.tracker.update(frame)
        if not success:
            self.is_tracking = False
            return False, None

        return True, tuple(map(int, bbox))

    def stop(self):
        self.is_tracking = False
        self.tracker = None