import cv2
from picamera2 import Picamera2

class CameraHelper:
    def __init__(self, resolution=(640, 480)):
        self.picam2 = Picamera2()

        # Configure the camera to output BGR888 directly.
        # This matches OpenCV's default color space, saving CPU time.
        config = self.picam2.create_video_configuration(
            main={"size": resolution, "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()

    def generate_frames(self):
        """Generator that yields JPEG-encoded frames for web streaming."""
        try:
            while True:
                # Capture the array directly from the main stream
                frame = self.picam2.capture_array()

                # Encode to JPEG for the web response
                success, buffer = cv2.imencode('.jpg', frame)
                if not success:
                    continue

                frame_bytes = buffer.tobytes()

                # Standard MJPEG stream boundary
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Streaming error: {e}")
        finally:
            self.stop()

    def stop(self):
        """Safely release the camera resources."""
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
            print("Camera resources released.")


import cv2
import numpy as np
import time

class MockCamera:
    """Generates a placeholder feed when camera is disconnected"""
    
    def __init__(self):
        self.width = 640
        self.height = 480
        print("⚠️  Mock camera initialized (physical camera unavailable)")
    
    def generate_placeholder_frame(self):
        """Generate a static placeholder image"""
        # Create black background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add some color gradient
        for y in range(self.height):
            intensity = int((y / self.height) * 50)
            frame[y, :] = [intensity, intensity // 2, intensity // 4]
        
        # Add text
        text1 = "CAMERA OFFLINE"
        text2 = "Maintenance Mode"
        text3 = time.strftime("%H:%M:%S")
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Main text
        cv2.putText(frame, text1, (150, 200), font, 1.2, (0, 217, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, text2, (180, 250), font, 0.8, (139, 149, 176), 2, cv2.LINE_AA)
        cv2.putText(frame, text3, (250, 300), font, 0.7, (0, 153, 255), 1, cv2.LINE_AA)
        
        # Add border and corners
        cv2.rectangle(frame, (10, 10), (self.width-10, self.height-10), (0, 217, 255), 2)
        
        return frame

    def generate_frames(self):
        """Mock frame generator - encapsulated within the class"""
        while True:
            # Call the internal frame generator
            frame = self.generate_placeholder_frame()
            
            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n' 
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Limit frame rate to save CPU (~10 fps)
            time.sleep(0.1)

# Example Usage:
# camera = MockCamera()
# return Response(camera.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
