"""
The Nomader - Mars Rover Web Control Interface
Flask application with real-time WebSocket communication
"""

from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import threading
import time
import sys
import os

# Import rover control modules
from helpers.serialcom import Robot
from helpers.camera import CameraHelper
from helpers.stats import system

app = Flask(__name__)
app.config['SECRET_KEY'] = 'nomader-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global rover instance
rover = None
rover_lock = threading.Lock()
camera_server = CameraHelper()

# Control state
control_state = {
    'command': 'halt',
    'speed': 0,
    'bias': 0,
    'pitch': 90,
    'yaw': 90,
    'light': 0,
    'connected': False
}

global_bias = 0.7
global_speed = 254

def initialize_rover():
    """Initialize rover connection"""
    global rover
    try:
        rover = Robot(port='/dev/ttyUSB0', baud_rate=115200)
        control_state['connected'] = True
        print("✓ Rover connected successfully")
        return True
    except Exception as e:
        print(f"✗ Rover connection failed: {e}")
        control_state['connected'] = False
        return False

def execute_drive_command():
    """Execute the current drive command using Robot methods"""
    global rover
    if not rover or not control_state['connected']:
        return
    
    with rover_lock:
        try:
            command = control_state['command']
            speed = global_speed
            bias = global_bias
            
            # Execute command using Robot's Motor class methods
            if command == 'forward':
                rover.motor.forward(speed)
            elif command == 'backward':
                rover.motor.backward(speed)
            elif command == 'right':
                rover.motor.left(speed-100)
            elif command == 'left':
                rover.motor.right(speed-100)

            elif command == 'arc_right_forward':
                rover.motor.arc_left(speed, bias)
            elif command == 'arc_left_forward':
                rover.motor.arc_right(speed, bias)

            elif command == 'arc_right_backward':
                rover.motor.arc_left(speed, -bias)
            elif command == 'arc_left_backward':
                rover.motor.arc_right(speed, -bias)

            elif command == 'halt':
                rover.motor.halt()
            else:
                # Fallback to halt if unknown command
                rover.motor.halt()
                
        except Exception as e:
            print(f"Error executing drive command: {e}")
            control_state['connected'] = False

def update_gimbal():
    """Update gimbal position"""
    global rover
    if rover and control_state['connected']:
        with rover_lock:
            try:
                rover.gimbal.pan(control_state['yaw'])
                rover.gimbal.tilt(control_state['pitch'])
            except Exception as e:
                print(f"Error updating gimbal: {e}")

def update_light():
    """Update light brightness"""
    global rover
    if rover and control_state['connected']:
        with rover_lock:
            try:
                rover.light = control_state['light']
                rover.update()
            except Exception as e:
                print(f"Error updating light: {e}")

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    try:
        return Response(camera_server.generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Camera error: {e}")
        return "Camera unavailable", 503

@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    try:
        ram_percent, ram_used, ram_total = system.get_ram_usage()
        stats = {
            'cpu_temp': system.get_cpu_temp(),
            'cpu_volts': system.get_cpu_volts(),
            'cpu_usage': system.get_cpu_usage(),
            'ram_percent': ram_percent,
            'ram_used': ram_used,
            'ram_total': ram_total,
            'throttled': system.get_throttled_status()
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('status', {
        'connected': control_state['connected'],
        'state': control_state
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection - emergency stop"""
    print('Client disconnected - emergency stop')
    control_state['command'] = 'halt'
    control_state['speed'] = 0
    control_state['bias'] = 0
    execute_drive_command()

@socketio.on('drive')
def handle_drive(data):
    """
    Handle drive commands from client
    Expected data format:
    {
        'command': 'forward' | 'backward' | 'left' | 'right' | 'arc_left' | 'arc_right' | 'halt',
        'speed': 0-255,
        'bias': 0.0-1.0 (for arc commands)
    }
    """
    control_state['command'] = data.get('command', 'halt')
    control_state['speed'] = int(data.get('speed', 0))
    control_state['bias'] = float(data.get('bias', 0))
    
    execute_drive_command()
    emit('state_update', control_state, broadcast=True)

@socketio.on('gimbal')
def handle_gimbal(data):
    """Handle gimbal commands"""
    if 'pitch' in data:
        control_state['pitch'] = int(data['pitch'])
    if 'yaw' in data:
        control_state['yaw'] = int(data['yaw'])
    
    update_gimbal()
    emit('state_update', control_state, broadcast=True)

@socketio.on('light')
def handle_light(data):
    """Handle light control"""
    control_state['light'] = int(data.get('brightness', 0))
    update_light()
    emit('state_update', control_state, broadcast=True)

@socketio.on('emergency_stop')
def handle_emergency_stop():
    """Emergency stop - kill all motors"""
    print("⚠️ EMERGENCY STOP ACTIVATED")
    control_state['command'] = 'halt'
    control_state['speed'] = 0
    control_state['bias'] = 0
    control_state['light'] = 0
    
    if rover:
        with rover_lock:
            rover.stop_all()
    
    emit('state_update', control_state, broadcast=True)

@socketio.on('reset_gimbal')
def handle_reset_gimbal():
    """Reset gimbal to center position"""
    control_state['pitch'] = 90
    control_state['yaw'] = 90
    
    if rover:
        with rover_lock:
            rover.gimbal.reset()
    
    emit('state_update', control_state, broadcast=True)

def stats_broadcast_thread():
    """Background thread to broadcast system stats"""
    while True:
        try:
            ram_percent, ram_used, ram_total = system.get_ram_usage()
            stats = {
                'cpu_temp': system.get_cpu_temp(),
                'cpu_usage': system.get_cpu_usage(),
                'ram_percent': ram_percent,
                'connected': control_state['connected']
            }
            socketio.emit('stats_update', stats)
        except Exception as e:
            print(f"Stats broadcast error: {e}")
        time.sleep(2)  # Update every 2 seconds

if __name__ == '__main__':
    # Initialize rover
    print("🚀 Initializing The Nomader...")
    initialize_rover()
    
    # Start stats broadcast thread
    stats_thread = threading.Thread(target=stats_broadcast_thread, daemon=True)
    stats_thread.start()
    
    # Run Flask app with SocketIO
    print("🌐 Starting web server on http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)