# Nomader

Nomader is a comprehensive autonomous robotics and rover project. It combines low-level hardware control, computer vision, web-based remote control, and advanced AI agents for autonomous operation and task execution.

## Project Architecture

The system is built on a multi-tier architecture, utilizing an Arduino Mega for real-time motor and sensor control, a Raspberry Pi 4 for high-level processing and video streaming, and advanced machine learning models for perception and autonomy.

### Key Components

1. **Hardware Control & Interface (`system_interface.ino`, `Serial_comm_test.py`)**
   - **Arduino Mega**: Runs `system_interface.ino` to control DC motors (left/right drive), servos (pitch/yaw for a camera gimbal), a light module, and an ultrasonic distance sensor.
   - **Raspberry Pi**: Acts as the main brain. It communicates with the Arduino over a USB serial connection, sending control commands (`left,right,pitch,yaw,light`) and receiving ultrasonic sensor data.

2. **Live Video Feed (`camera_server.py`)**
   - A Flask-based web server running on the Raspberry Pi that captures video using `Picamera2` and streams a live MJPEG feed to connected clients.

3. **Rover Web GUI (`Rover_GUI/`)**
   - A Flask web application providing a user-friendly remote control interface for driving the rover, viewing the live camera feed, and monitoring sensor telemetry.

4. **NOVA Autonomous Agent (`NOVA/`)**
   - A sophisticated AI agent framework designed for the rover. It includes modules for:
     - **Perception & Vision**: Processing sensory input and camera feeds.
     - **Navigation**: Path planning and movement.
     - **Memory**: Storing context and environment maps.
     - **Skills & Tools**: Executing complex multi-step tasks.

5. **Reinforcement Learning & AI (`RL-VLA-JEPA/`)**
   - Implements a Vision-Language-Action (VLA) architecture using Joint Embedding Predictive Architecture (JEPA).
   - Used for training local models (`train_local.py`) to enable autonomous behavior.
   - Leverages YOLOv8 (`yolov8s-worldv2.pt`) for real-time object detection and world understanding.

## Repository Structure

```
Nomader/
├── NOVA/                   # Autonomous AI agent framework
├── RL-VLA-JEPA/            # Reinforcement learning and VLA models
├── Rover_GUI/              # Flask-based web interface for remote control
├── my_local_model/         # Saved local model weights
├── system_interface.ino    # Arduino Mega motor & sensor firmware
├── Serial_comm_test.py     # Python script for testing Pi-to-Arduino serial comms
├── camera_server.py        # Flask app for streaming Picamera2 video feed
├── yolov8s-worldv2.pt      # YOLOv8 weights for object detection
└── .env                    # Environment variables configuration
```

## Getting Started

### 1. Hardware Setup
- Connect the Arduino Mega to the Raspberry Pi 4 via USB.
- Connect the motor shield, DC motors, servos (Pins 9 & 10), and Ultrasonic Sensor (Trig: 22, Echo: 23) to the Arduino.
- Connect the Pi Camera module to the Raspberry Pi.

### 2. Arduino Firmware
- Open `system_interface.ino` in the Arduino IDE.
- Install the `Adafruit Motor Shield V1` (`AFMotor.h`) library.
- Compile and upload the code to the Arduino Mega.

### 3. Raspberry Pi Environment
- Clone this repository on the Raspberry Pi.
- Install Python dependencies:
  ```bash
  pip install pyserial flask opencv-python picamera2
  ```
- Make sure you have the correct permissions to access the serial port (e.g., `sudo usermod -a -G dialout $USER`).

### 4. Running the System
- **Test Serial Communication**: 
  Run `python Serial_comm_test.py` to test interactive or demo modes for driving the rover and moving the gimbal.
- **Start the Camera Server**:
  Run `python camera_server.py` and navigate to `http://<RASPBERRY_PI_IP>:5000` to view the live feed.
- **Start the Web GUI**:
  Navigate to `Rover_GUI/` and run `python app.py`. Access the control panel via your web browser.

## AI & Autonomy (Advanced)
For advanced autonomous features, explore the `NOVA/` and `RL-VLA-JEPA/` directories. These modules require additional setup, including PyTorch and related ML libraries, to train and run the Vision-Language-Action models. See `NOVA/README.md` for more details on the agent configuration.
