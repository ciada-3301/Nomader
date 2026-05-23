"""
NOVA Odometry
-------------
Estimates position and heading using dead reckoning from motor commands.
"""

import time
import math

class Odometry:
    def __init__(self, config):
        self.config = config
        
        self.x_cm = 0.0
        self.y_cm = 0.0
        self.heading_rad = 0.0 # 0 = facing positive X
        
        self.last_update_time = time.time()
        
        # Max speed calibration (approximate cm/sec at max PWM)
        # Tuning this is required for accurate dead reckoning!
        self.max_cm_per_sec = 40.0 
        
    def reset(self, x=0.0, y=0.0, heading=0.0):
        self.x_cm = x
        self.y_cm = y
        self.heading_rad = heading
        self.last_update_time = time.time()
        
    def update(self, left_pwm: int, right_pwm: int):
        """
        Update position based on differential drive kinematics.
        left_pwm, right_pwm: -255 to 255
        """
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now
        
        if dt > 1.0: # Ignore long pauses
            return
            
        # Convert PWM to approximate cm/sec
        v_left = (left_pwm / 255.0) * self.max_cm_per_sec
        v_right = (right_pwm / 255.0) * self.max_cm_per_sec
        
        # Differential drive kinematics
        # linear velocity = (v_r + v_l) / 2
        # angular velocity = (v_r - v_l) / wheelbase
        
        v = (v_right + v_left) / 2.0
        w = (v_right - v_left) / self.config.hardware.wheel_base_cm
        
        if abs(w) < 0.01:
            # Moving straight
            self.x_cm += v * math.cos(self.heading_rad) * dt
            self.y_cm += v * math.sin(self.heading_rad) * dt
        else:
            # Moving in an arc
            r = v / w
            
            # ICC (Instantaneous Center of Curvature)
            icc_x = self.x_cm - r * math.sin(self.heading_rad)
            icc_y = self.y_cm + r * math.cos(self.heading_rad)
            
            dtheta = w * dt
            
            self.x_cm = math.cos(dtheta) * (self.x_cm - icc_x) - math.sin(dtheta) * (self.y_cm - icc_y) + icc_x
            self.y_cm = math.sin(dtheta) * (self.x_cm - icc_x) + math.cos(dtheta) * (self.y_cm - icc_y) + icc_y
            self.heading_rad += dtheta
            
        # Normalize heading to -pi to pi
        self.heading_rad = math.atan2(math.sin(self.heading_rad), math.cos(self.heading_rad))
        
    def get_pose(self):
        return self.x_cm, self.y_cm, self.heading_rad
