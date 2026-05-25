"""
NOVA Odometry
-------------
Dead-reckoning position estimate from motor PWM commands.
No encoders — PWM-to-velocity calibration is critical.
"""

import time
import math


class Odometry:
    def __init__(self, config):
        self.config = config
        self.x_cm       = 0.0
        self.y_cm       = 0.0
        self.heading_rad = 0.0
        self._last_t     = time.time()
        # Calibrate: cm/s at full PWM (255). Tune on actual hardware.
        self.max_cm_per_sec = 40.0

    def reset(self, x=0.0, y=0.0, heading=0.0):
        self.x_cm        = x
        self.y_cm        = y
        self.heading_rad = heading
        self._last_t     = time.time()

    def update(self, left_pwm: int, right_pwm: int):
        """
        Differential drive kinematics from PWM values.
        Note: left_pwm / right_pwm here are LOGICAL (pre-swap) values.
        """
        now = time.time()
        dt  = now - self._last_t
        self._last_t = now
        if dt > 1.0:
            return  # ignore long pauses

        v_l = (left_pwm  / 255.0) * self.max_cm_per_sec
        v_r = (right_pwm / 255.0) * self.max_cm_per_sec

        v = (v_r + v_l) / 2.0
        w = (v_r - v_l) / self.config.hardware.wheel_base_cm

        if abs(w) < 0.01:
            self.x_cm += v * math.cos(self.heading_rad) * dt
            self.y_cm += v * math.sin(self.heading_rad) * dt
        else:
            r = v / w
            icc_x = self.x_cm - r * math.sin(self.heading_rad)
            icc_y = self.y_cm + r * math.cos(self.heading_rad)
            dtheta = w * dt
            self.x_cm = (
                math.cos(dtheta) * (self.x_cm - icc_x)
                - math.sin(dtheta) * (self.y_cm - icc_y)
                + icc_x
            )
            self.y_cm = (
                math.sin(dtheta) * (self.x_cm - icc_x)
                + math.cos(dtheta) * (self.y_cm - icc_y)
                + icc_y
            )
            self.heading_rad += dtheta

        self.heading_rad = math.atan2(
            math.sin(self.heading_rad), math.cos(self.heading_rad)
        )

    def get_pose(self):
        return self.x_cm, self.y_cm, self.heading_rad
