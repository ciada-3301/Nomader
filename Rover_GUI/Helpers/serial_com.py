# Communicates with the Arduino over USB port
"""
Controls:
    - Left Motor: 0 to 255
    - Right Motor: 0 to 255
    - Gimbal Pitch: 0 to 180 degrees
    - Gimbal Yaw: 0 to 180 degrees
    - Light: 0 to 255 (brightness)
"""





class motor():
    def forward(self,speed_threshold):
        return None
    
    def backward(self, speed):
        return None
    
    def left(self,speed):
        return None
    
    def right(self, speed):
        return None
    
    def halt(self):
        return None
    
    def glide_left(self):
        return None
    
    def glide_right(self):
        return None
        