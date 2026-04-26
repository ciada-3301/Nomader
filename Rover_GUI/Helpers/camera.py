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