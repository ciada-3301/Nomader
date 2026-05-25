"""
NOVA Visual Obstacle Detector
------------------------------
Detects obstacles using only the camera — no ultrasonic sensor.

Inspired by classic desk-robot edge detection architecture:
  • Analyse the floor plane (bottom portion of frame) for edge density.
  • High edge density in the floor zone = obstacle blocking path.
  • Lateral position of the obstacle cluster tells us which way to steer.
  • "Proximity score" (0–1) encodes how urgent the threat is.

This is intentionally lightweight — designed for Raspberry Pi real-time use.
Heavier YOLO detection runs separately in the ObjectDetector.
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple


class VisualObstacleDetector:
    """
    Analyses incoming camera frames to detect forward obstacles.

    All state is frame-based — no ultrasonic, no rangefinder.
    Callers must call update(frame) at their control rate.
    """

    def __init__(self, config):
        self.config = config
        self.last_score = 0.0          # 0=clear, 1=definitely blocked
        self.last_lateral = 0.0        # -1=left, 0=centre, +1=right
        self.last_frame_time = 0.0
        self._prev_gray: Optional[np.ndarray] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, frame: Optional[np.ndarray]) -> Tuple[float, float]:
        """
        Analyse one frame and return (obstacle_score, lateral_bias).

        obstacle_score : 0.0 (clear) → 1.0 (fully blocked)
        lateral_bias   : -1.0 (obstacle on left) → +1.0 (obstacle on right)
                          0.0 means centred / no obstacle

        Cheap enough to run at 10 Hz on a Pi 4.
        """
        if frame is None:
            return self.last_score, self.last_lateral

        h, w = frame.shape[:2]

        # ── Floor zone: bottom N% of frame ────────────────────────────────────
        floor_top = int(h * (1.0 - self.config.vision.floor_fraction))
        floor_roi  = frame[floor_top:, :]

        # ── Canny edge density in floor zone ──────────────────────────────────
        gray = cv2.cvtColor(floor_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        total_px   = edges.size
        edge_px    = int(np.sum(edges > 0))
        edge_density = edge_px / total_px if total_px > 0 else 0.0

        threshold = self.config.vision.edge_density_threshold
        min_w_frac = self.config.vision.obstacle_width_min_frac

        if edge_density < threshold:
            # Floor is clear
            score   = 0.0
            lateral = 0.0
        else:
            # Normalise score: how far above threshold are we?
            score = min(1.0, (edge_density - threshold) / (threshold * 4))

            # Find lateral centre of mass of edge pixels
            # This tells us which side the obstacle is on
            col_sums = np.sum(edges, axis=0).astype(float)
            total_e  = col_sums.sum()
            if total_e > 0:
                col_indices = np.arange(len(col_sums))
                centroid_x  = float(np.sum(col_indices * col_sums) / total_e)
                # Normalise to -1..+1 (left..right)
                lateral = (centroid_x / w - 0.5) * 2.0
            else:
                lateral = 0.0

            # Extra sanity: if the edge band is very narrow horizontally
            # it might be a floor texture, not an obstacle. Check width.
            nonzero_cols = np.where(col_sums > col_sums.max() * 0.1)[0]
            if len(nonzero_cols) > 0:
                span_frac = (nonzero_cols[-1] - nonzero_cols[0]) / w
                if span_frac < min_w_frac:
                    score *= 0.4   # down-weight narrow vertical features

        self.last_score   = score
        self.last_lateral = lateral
        self.last_frame_time = time.time()
        return score, lateral

    def is_path_clear(self) -> bool:
        stop = self.config.navigation.obstacle_stop_score
        return self.last_score < stop

    def should_slow(self) -> bool:
        slow = self.config.navigation.obstacle_slow_score
        return self.last_score >= slow

    def get_proximity_score(self) -> float:
        return self.last_score

    def get_lateral_bias(self) -> float:
        """Positive = obstacle on right, negative = obstacle on left."""
        return self.last_lateral

    # ── Stale check ────────────────────────────────────────────────────────────
    def is_stale(self, max_age_s: float = 0.5) -> bool:
        """True if no frame has been processed recently."""
        return (time.time() - self.last_frame_time) > max_age_s
