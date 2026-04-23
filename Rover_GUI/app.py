from flask import Flask, render_template, Response, jsonify, request
import cv2
import io
from datetime import datetime
import os

app = Flask(__name__)

# Create directories for saved media
os.makedirs('static/captures', exist_ok=True)
os.makedirs('static/recordings', exist_ok=True)

# Global variables
camera = None
recording = False
video_writer = None

def get_camera():
    """Initialize camera (you'll connect your actual camera here)"""
    global camera
    if camera is None:
        # Replace with your actual camera initialization
        camera = cv2.VideoCapture(0)
    return camera

def generate_frames():
    """Generate camera frames for streaming"""
    camera = get_camera()
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Serve the main control interface"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Movement Controls
@app.route('/move/forward', methods=['POST'])
def move_forward():
    """Move rover forward"""
    # Add your motor control code here
    print("Moving forward")
    return jsonify({'status': 'success', 'action': 'forward'})

@app.route('/move/backward', methods=['POST'])
def move_backward():
    """Move rover backward"""
    # Add your motor control code here
    print("Moving backward")
    return jsonify({'status': 'success', 'action': 'backward'})

@app.route('/move/left', methods=['POST'])
def move_left():
    """Turn rover left (pivot)"""
    # Add your motor control code here
    print("Turning left")
    return jsonify({'status': 'success', 'action': 'left'})

@app.route('/move/right', methods=['POST'])
def move_right():
    """Turn rover right (pivot)"""
    # Add your motor control code here
    print("Turning right")
    return jsonify({'status': 'success', 'action': 'right'})

@app.route('/move/smooth_left', methods=['POST'])
def smooth_left():
    """Smooth turn left (curved path)"""
    # Add your motor control code here
    print("Smooth turn left")
    return jsonify({'status': 'success', 'action': 'smooth_left'})

@app.route('/move/smooth_right', methods=['POST'])
def smooth_right():
    """Smooth turn right (curved path)"""
    # Add your motor control code here
    print("Smooth turn right")
    return jsonify({'status': 'success', 'action': 'smooth_right'})

@app.route('/move/stop', methods=['POST'])
def stop():
    """Stop all movement"""
    # Add your motor control code here
    print("Stopping")
    return jsonify({'status': 'success', 'action': 'stop'})

# Camera Gimbal Controls
@app.route('/gimbal/up', methods=['POST'])
def gimbal_up():
    """Tilt camera up"""
    # Add your servo control code here
    print("Camera tilting up")
    return jsonify({'status': 'success', 'action': 'gimbal_up'})

@app.route('/gimbal/down', methods=['POST'])
def gimbal_down():
    """Tilt camera down"""
    # Add your servo control code here
    print("Camera tilting down")
    return jsonify({'status': 'success', 'action': 'gimbal_down'})

@app.route('/gimbal/left', methods=['POST'])
def gimbal_left():
    """Pan camera left"""
    # Add your servo control code here
    print("Camera panning left")
    return jsonify({'status': 'success', 'action': 'gimbal_left'})

@app.route('/gimbal/right', methods=['POST'])
def gimbal_right():
    """Pan camera right"""
    # Add your servo control code here
    print("Camera panning right")
    return jsonify({'status': 'success', 'action': 'gimbal_right'})

@app.route('/gimbal/center', methods=['POST'])
def gimbal_center():
    """Center camera position"""
    # Add your servo control code here
    print("Centering camera")
    return jsonify({'status': 'success', 'action': 'gimbal_center'})

# Media Controls
@app.route('/capture_image', methods=['POST'])
def capture_image():
    """Capture and save current frame"""
    camera = get_camera()
    success, frame = camera.read()
    if success:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join('static/captures', filename)
        cv2.imwrite(filepath, frame)
        return jsonify({'status': 'success', 'filename': filename, 'path': filepath})
    return jsonify({'status': 'error', 'message': 'Failed to capture image'})

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """Start video recording"""
    global recording, video_writer
    if not recording:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.avi"
        filepath = os.path.join('static/recordings', filename)
        
        camera = get_camera()
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # type: ignore 
        video_writer = cv2.VideoWriter(filepath, fourcc, 20.0, (640, 480))
        recording = True
        return jsonify({'status': 'success', 'message': 'Recording started', 'filename': filename})
    return jsonify({'status': 'error', 'message': 'Already recording'})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """Stop video recording"""
    global recording, video_writer
    if recording:
        recording = False
        if video_writer:
            video_writer.release()
            video_writer = None
        return jsonify({'status': 'success', 'message': 'Recording stopped'})
    return jsonify({'status': 'error', 'message': 'Not recording'})

@app.route('/execute_command', methods=['POST'])
def execute_command():
    """Execute shell command from the web interface"""
    data = request.get_json()
    command = data.get('command', '')
    
    # SECURITY WARNING: This is unsafe for production!
    # Add proper command validation and sanitization
    # For development on local network only
    
    try:
        import subprocess
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
        output = result.stdout + result.stderr
        return jsonify({'status': 'success', 'output': output})
    except Exception as e:
        return jsonify({'status': 'error', 'output': str(e)})

@app.route('/get_status', methods=['GET'])
def get_status():
    """Get rover status information"""
    # Add your sensor readings here
    status = {
        'battery': 85,  # Placeholder
        'temperature': 28,  # Placeholder
        'recording': recording,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return jsonify(status)

if __name__ == '__main__':
    # Run on all interfaces so you can access from other devices
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)