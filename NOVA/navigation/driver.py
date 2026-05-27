"""
NOVA Navigation Driver
----------------------
Executes waypoint paths with reactive obstacle avoidance.

This driver abstracts motor commands with _send_motors(logical_left, logical_right)
which handles per-motor trim correction internally.
"""

import time
import math
from typing import List, Tuple, Optional
from enum import Enum

class MovementSpeed(Enum):
    STOPPED = 0
    SLOW_CRAWL = 1   # ~20% motor power — use during approach and perception
    NORMAL = 2       # ~60% motor power — use for transit
    FAST = 3         # ~100% motor power — use only for open-space transit


class NavigationDriver:
    def __init__(self, config, robot, odometry, obstacle_detector):
        self.config   = config
        self.robot    = robot
        self.odom     = odometry
        self.obstacle = obstacle_detector  # VisualObstacleDetector

        self.waypoints:       List[Tuple[float, float]] = []
        self.current_wp_idx:  int   = 0
        self.is_navigating:   bool  = False
        self.last_progress_t: float = 0.0
        self.last_dist_goal:  float = 9999.0
        self.current_speed:   MovementSpeed = MovementSpeed.STOPPED

    def get_current_speed(self) -> MovementSpeed:
        return self.current_speed


    # ── Internal motor send ────────────────────────────────────────────────────

    def _send_motors(self, logical_left: int, logical_right: int):
        """
        Send motor commands, applying:
          1. Per-motor trim correction
        """
        if not self.robot:
            return
        nav  = self.config.navigation
        left  = int(logical_left  * nav.left_motor_trim)
        right = int(logical_right * nav.right_motor_trim)
        
        max_s = max(abs(logical_left), abs(logical_right))
        if max_s == 0:
            self.current_speed = MovementSpeed.STOPPED
        elif max_s <= nav.min_speed:
            self.current_speed = MovementSpeed.SLOW_CRAWL
        elif max_s <= nav.cruise_speed + 30:
            self.current_speed = MovementSpeed.NORMAL
        else:
            self.current_speed = MovementSpeed.FAST

        self.robot.motor.drive(left, right, 0)

    # ── Path control ───────────────────────────────────────────────────────────

    def set_path(self, waypoints: List[Tuple[float, float]]):
        self.waypoints       = waypoints
        self.current_wp_idx  = 0
        self.is_navigating   = True
        self.last_progress_t = time.time()
        self.last_dist_goal  = 9999.0

    def stop(self):
        self.is_navigating = False
        if self.robot:
            self.robot.motor.halt()

    # ── Main control step (call at config.navigation.control_hz) ─────────────

    def update(self, camera_frame=None) -> str:
        """
        One control loop tick. Returns status string:
        'running' | 'reached' | 'blocked' | 'stuck'

        camera_frame: optional — passed to obstacle detector for visual avoidance.
        """
        if not self.is_navigating or not self.waypoints:
            return 'idle'

        if self.current_wp_idx >= len(self.waypoints):
            self.stop()
            return 'reached'

        target_x, target_y = self.waypoints[self.current_wp_idx]
        curr_x, curr_y, curr_heading = self.odom.get_pose()

        dist_to_wp = math.hypot(target_x - curr_x, target_y - curr_y)

        # Advance waypoint
        if dist_to_wp < self.config.navigation.waypoint_reach_cm:
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.waypoints):
                self.stop()
                return 'reached'
            target_x, target_y = self.waypoints[self.current_wp_idx]
            dist_to_wp = math.hypot(target_x - curr_x, target_y - curr_y)

        # Stuck detection
        if dist_to_wp < self.last_dist_goal - 5.0:
            self.last_dist_goal  = dist_to_wp
            self.last_progress_t = time.time()
        elif time.time() - self.last_progress_t > self.config.navigation.stuck_timeout_s:
            self.stop()
            return 'stuck'

        # ── Visual obstacle check ──────────────────────────────────────────────
        obs_score  = 0.0
        lateral    = 0.0
        if camera_frame is not None:
            obs_score, lateral = self.obstacle.update(camera_frame)
        else:
            obs_score = self.obstacle.get_proximity_score()
            lateral   = self.obstacle.get_lateral_bias()

        stop_score = self.config.navigation.obstacle_stop_score
        slow_score = self.config.navigation.obstacle_slow_score

        if obs_score >= stop_score:
            self.stop()
            return 'blocked'

        # ── Heading control ────────────────────────────────────────────────────
        angle_to_target = math.atan2(target_y - curr_y, target_x - curr_x)
        angle_diff = math.atan2(
            math.sin(angle_to_target - curr_heading),
            math.cos(angle_to_target - curr_heading),
        )

        nav   = self.config.navigation
        speed = nav.cruise_speed
        if obs_score >= slow_score:
            # Slow down near obstacles; also skew away from the obstacle
            speed = nav.min_speed
            # Steer away: if obstacle is on left, add left motor boost; vice versa
            angle_diff -= lateral * 0.4   # reactive dodge

        # P-controller
        turn = int(angle_diff * 100)
        turn = max(-nav.turn_speed, min(nav.turn_speed, turn))

        left_speed  = max(-nav.max_speed, min(nav.max_speed, speed - turn))
        right_speed = max(-nav.max_speed, min(nav.max_speed, speed + turn))

        self._send_motors(left_speed, right_speed)
        return 'running'
