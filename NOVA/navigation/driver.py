"""
NOVA Navigation Driver
----------------------
Executes a path by following waypoints. Includes reactive obstacle avoidance.
"""

import time
import math
from typing import List, Tuple

class NavigationDriver:
    def __init__(self, config, robot, odometry, obstacle_detector):
        self.config = config
        self.robot = robot
        self.odom = odometry
        self.obstacle_detector = obstacle_detector
        
        self.waypoints = []
        self.current_wp_idx = 0
        self.is_navigating = False
        
        self.last_progress_time = 0
        self.last_dist_to_goal = 9999.0
        
    def set_path(self, waypoints: List[Tuple[float, float]]):
        self.waypoints = waypoints
        self.current_wp_idx = 0
        self.is_navigating = True
        self.last_progress_time = time.time()
        self.last_dist_to_goal = 9999.0
        
    def stop(self):
        self.is_navigating = False
        if self.robot:
            self.robot.motor.halt()
            
    def update(self) -> str:
        """
        Control loop step. Call frequently (e.g. 10Hz).
        Returns status: 'running', 'reached', 'blocked', 'stuck'
        """
        if not self.is_navigating or not self.waypoints:
            return 'idle'
            
        if self.current_wp_idx >= len(self.waypoints):
            self.stop()
            return 'reached'
            
        target_x, target_y = self.waypoints[self.current_wp_idx]
        curr_x, curr_y, curr_heading = self.odom.get_pose()
        
        # Check distance to waypoint
        dist_to_wp = math.sqrt((target_x - curr_x)**2 + (target_y - curr_y)**2)
        
        if dist_to_wp < self.config.navigation.waypoint_reach_cm:
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.waypoints):
                self.stop()
                return 'reached'
            target_x, target_y = self.waypoints[self.current_wp_idx]
            dist_to_wp = math.sqrt((target_x - curr_x)**2 + (target_y - curr_y)**2)
            
        # Stuck detection
        if dist_to_wp < self.last_dist_to_goal - 5.0:
            self.last_dist_to_goal = dist_to_wp
            self.last_progress_time = time.time()
        elif time.time() - self.last_progress_time > self.config.navigation.stuck_timeout_s:
            self.stop()
            return 'stuck'
            
        # Obstacle avoidance check
        forward_dist = self.obstacle_detector.get_forward_distance_cm()
        if forward_dist < self.config.navigation.obstacle_stop_cm:
            self.stop()
            return 'blocked'
            
        # Pure pursuit angle calculation
        angle_to_target = math.atan2(target_y - curr_y, target_x - curr_x)
        angle_diff = angle_to_target - curr_heading
        
        # Normalize angle difference
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
        
        # Convert to motor commands
        speed = self.config.navigation.cruise_speed
        
        if forward_dist < self.config.navigation.obstacle_slow_cm:
            # Slow down if obstacle is getting close
            speed = self.config.navigation.min_speed
            
        # Simple proportional controller for steering
        turn = int(angle_diff * 100)
        
        # Cap turn
        turn = max(-self.config.navigation.turn_speed, min(self.config.navigation.turn_speed, turn))
        
        left_speed = speed - turn
        right_speed = speed + turn
        
        # Cap speeds
        max_s = self.config.navigation.max_speed
        min_s = -max_s
        left_speed = max(min_s, min(max_s, left_speed))
        right_speed = max(min_s, min(max_s, right_speed))
        
        # Send to robot
        if self.robot:
            self.robot.motor.drive(right_speed, left_speed, 0) # Use the existing drive method or custom
            
        return 'running'
