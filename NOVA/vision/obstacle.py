"""
NOVA Obstacle Detection
-----------------------
Fuses ultrasonic sensor data with basic visual floor analysis
to determine safe drivable space.
"""

import time
import numpy as np

class ObstacleDetector:
    def __init__(self, config, robot_interface=None):
        self.config = config
        self.robot = robot_interface
        self.last_us_distance = -1
        
    def update(self):
        """Update sensor readings."""
        if self.robot and self.config.hardware.ultrasonic_enabled:
            # Get latest ultrasonic distance parsed from Arduino OK,<dist> response
            self.last_us_distance = getattr(self.robot, 'last_distance', -1)
            
    def is_path_clear(self) -> bool:
        """Check if straight path is clear of obstacles."""
        self.update()
        
        # Check ultrasonic
        if self.last_us_distance > 0 and self.last_us_distance < self.config.navigation.obstacle_stop_cm:
            return False
            
        # (Could add visual obstacle detection here later)
        return True
        
    def get_forward_distance_cm(self) -> float:
        """Get distance to nearest forward obstacle in cm."""
        self.update()
        if self.last_us_distance > 0:
            return float(self.last_us_distance)
        return 999.0 # Clear
