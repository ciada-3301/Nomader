#!/usr/bin/env python3
"""
Raspberry Pi Motor Controller
Sends control commands to Arduino Mega via serial USB connection

Usage:
    python motor_control.py

Controls:
    - Left Motor: -255 to 255
    - Right Motor: -255 to 255
    - Gimbal Pitch: 0 to 180 degrees
    - Gimbal Yaw: 0 to 180 degrees
    - Light: 0 to 255 (brightness)
"""

import serial
import time
import sys

class MotorController:
    def __init__(self, port='/dev/ttyUSB0', baud_rate=115200, timeout=1):
        """
        Initialize serial connection to Arduino
        
        Args:
            port: Serial port (usually /dev/ttyACM0 or /dev/ttyUSB0)
            baud_rate: Communication speed (must match Arduino)
            timeout: Read timeout in seconds
        """
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
            print("\nTry these ports:")
            print("  /dev/ttyACM0")
            print("  /dev/ttyACM1")
            print("  /dev/ttyUSB0")
            sys.exit(1)
    
    def send_command(self, left_speed, right_speed, pitch, yaw, light):
        """
        Send control command to Arduino
        
        Args:
            left_speed: Left motor speed (-255 to 255)
            right_speed: Right motor speed (-255 to 255)
            pitch: Gimbal pitch angle (0 to 180)
            yaw: Gimbal yaw angle (0 to 180)
            light: Light brightness (0 to 255)
        
        Returns:
            Response from Arduino or None if no response
        """
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
        """Emergency stop - set all motors to 0 and center servos"""
        return self.send_command(0, 0, 90, 90, 0)
    
    def close(self):
        """Close serial connection"""
        if self.serial.is_open:
            self.stop_all()
            self.serial.close()
            print("Connection closed")


def demo_mode(controller):
    """Run a simple demo sequence"""
    print("\n=== Running Demo Mode ===")
    print("Press Ctrl+C to stop\n")
    
    try:
        # Test sequence
        sequences = [
            (100, 100, 90, 90, 128, "Moving forward, light half brightness"),
            (0, 0, 90, 90, 255, "Stopped, light full brightness"),
            (-100, -100, 90, 90, 0, "Moving backward, light off"),
            (100, -100, 90, 90, 128, "Turning right"),
            (-100, 100, 90, 90, 128, "Turning left"),
            (0, 0, 45, 90, 0, "Gimbal pitch down"),
            (0, 0, 135, 90, 0, "Gimbal pitch up"),
            (0, 0, 90, 45, 0, "Gimbal yaw left"),
            (0, 0, 90, 135, 0, "Gimbal yaw right"),
            (0, 0, 90, 90, 0, "Return to neutral"),
        ]
        
        for left, right, pitch, yaw, light, description in sequences:
            print(f"{description}...")
            response = controller.send_command(left, right, pitch, yaw, light)
            if response:
                print(f"  Arduino: {response}")
            time.sleep(2)
        
        print("\nDemo complete!")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted")
    finally:
        controller.stop_all()


def interactive_mode(controller):
    """Interactive control mode"""
    print("\n=== Interactive Mode ===")
    print("Enter commands in format: left,right,pitch,yaw,light")
    print("Example: 100,100,90,90,128")
    print("Type 'stop' to emergency stop")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("Command: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'stop':
                response = controller.stop_all()
                print(f"Emergency stop! {response}")
                continue
            
            # Parse input
            values = [int(x.strip()) for x in user_input.split(',')]
            
            if len(values) != 5:
                print("Error: Need 5 values (left,right,pitch,yaw,light)")
                continue
            
            response = controller.send_command(*values)
            if response:
                print(f"Arduino: {response}")
            
        except ValueError:
            print("Error: Invalid input. Use numbers separated by commas")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    print("Raspberry Pi Motor Controller")
    print("=" * 50)
    
    # Initialize controller
    # Change port if needed (check with: ls /dev/ttyACM* or ls /dev/ttyUSB*)
    controller = MotorController(port='/dev/ttyACM0', baud_rate=115200)
    
    try:
        # Choose mode
        print("\nSelect mode:")
        print("1. Demo Mode (automated sequence)")
        print("2. Interactive Mode (manual control)")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == '1':
            demo_mode(controller)
        else:
            interactive_mode(controller)
            
    except KeyboardInterrupt:
        print("\nProgram interrupted")
    finally:
        controller.close()