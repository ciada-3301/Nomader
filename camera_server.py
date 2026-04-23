from flask import Flask, Response
from picamera2 import Picamera2  # camera interface module for the raspberry pi
import cv2

app = Flask(__name__)

# Initialize camera
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

def generate_frames():
    while True:
        frame = picam2.capture_array()

        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
        <head><title>Rover Cam</title></head>
        <body>
            <h1>Live Feed</h1>
            <img src="/video" width="640" height="480">
        </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)