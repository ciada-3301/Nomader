# Communicates with the Arduino over USB port
"""
Controls:
    - Left Motor: -255 to 255
    - Right Motor: -255 to 255
    - Gimbal Pitch: 0 to 180 degrees
    - Gimbal Yaw: 0 to 180 degrees+
    - Light: 0 to 255 (brightness)
"""

import serial
import time
import sys

def map_value(y, y_min, y_max, x_min, x_max):
    return x_min + (y - y_min) * (x_max - x_min) / (y_max - y_min)

class Robot:
    def __init__(self, port='/dev/ttyUSB0', baud_rate=115200, timeout=1):
        
        try:
            self.serial = serial.Serial(port, baud_rate, timeout=timeout)
            time.sleep(2)  # Wait for Arduino to reset after serial connection
            print(f"Connected to Arduino on {port}")
            
            # Read initial message from Arduino
            if self.serial.in_waiting:
                response = self.serial.readline().decode('utf-8').strip()
                print(f"Arduino: {response}")
                
        except serial.SerialException as e:
            print(f"Error: Could not open serial port {port}")
            print(f"Details: {e}")
            sys.exit(1)
        self.motor = self.Motor(self)
        self.left_speed = 0
        self.right_speed = 0
        self.pitch = 90
        self.yaw = 90
        self.light = 0
    
    def send_command(self, left_speed, right_speed, pitch, yaw, light):
        
        # Constrain values
        left_speed = max(-255, min(255, int(left_speed)))
        right_speed = max(-255, min(255, int(right_speed)))
        pitch = max(0, min(180, int(pitch)))
        yaw = max(0, min(180, int(yaw)))
        light = max(0, min(255, int(light)))
        
        # Format command
        command = f"{left_speed},{right_speed},{pitch},{yaw},{light}\n"
        
        # Send command
        self.serial.write(command.encode('utf-8'))
        
        # Wait for response
        time.sleep(0.01)  # Small delay for Arduino to process
        
        if self.serial.in_waiting:
            response = self.serial.readline().decode('utf-8').strip()
            return response
        return None
    
    def stop_all(self):
        """Emergency stop - set all motors to 0 and center servos & turn off light"""
        return self.send_command(0, 0, 90, 90, 0)
    
    def update(self):
        return self.send_command(
            self.left_speed,
            self.right_speed,
            self.pitch,
            self.yaw,
            self.light
        )
    
    def close(self):
        """Close serial connection"""
        if self.serial.is_open:
            self.stop_all()
            self.serial.close()
            print("Connection closed")
    
    class Motor:
        def __init__(self, robot):
            self.robot = robot
                    
        def forward(self,speed):
            self.robot.left_speed = speed
            self.robot.right_speed = speed
            self.robot.update()
        
        def backward(self, speed):
            self.robot.left_speed = -speed
            self.robot.right_speed = -speed
            self.robot.update()
        
        def right(self, speed):
            self.robot.left_speed = speed
            self.robot.right_speed = -speed
            self.robot.update()
        
        def left(self, speed):
            self.robot.left_speed = -speed
            self.robot.right_speed = speed
            self.robot.update()

        def arc_left(self, speed, bias):
            base_speed = speed if bias >= 0 else -speed
            left_motor_speed = map_value(bias, -1, 1, -speed, speed)
            self.robot.left_speed = left_motor_speed
            self.robot.right_speed = base_speed
            self.robot.update()

        def arc_right(self, speed, bias):
            right_motor_speed = map_value(bias, -1, 1, -speed, speed)
            self.robot.left_speed = base_speed
            self.robot.right_speed = right_motor_speed
            self.robot.update()

        def drive(self, right, left, turn_bias):
            self.robot.left_speed = left
            self.robot.right_speed = right
            self.robot.update()




