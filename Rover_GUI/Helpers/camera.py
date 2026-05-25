import cv2
import numpy as np
import time
from picamera2 import Picamera2


class CameraHelper:
    def __init__(self, resolution=(640, 480)):
        self.picam2 = Picamera2()
        # BGR888 format outputs in OpenCV-native channel order.
        config = self.picam2.create_video_configuration(
            main={"size": resolution, "format": "BGR888"}
        )
        self.picam2.configure(config)
        self.picam2.start()

    def get_frame(self):
        """Return the latest BGR frame array."""
        try:
            return self.picam2.capture_array()
        except Exception:
            return None

    def generate_frames(self):
        """Generator that yields JPEG-encoded frames for web streaming."""
        try:
            while True:
                frame = self.picam2.capture_array()
                success, buffer = cv2.imencode('.jpg', frame)
                if not success:
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Streaming error: {e}")
        finally:
            self.stop()

    def stop(self):
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
            print("Camera resources released.")


class MockCamera:
    """Generates a placeholder feed when camera is disconnected."""

    def __init__(self):
        self.width = 640
        self.height = 480
        print("⚠️  Mock camera initialized (physical camera unavailable)")

    def get_frame(self):
        return self.generate_placeholder_frame()

    def generate_placeholder_frame(self):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for y in range(self.height):
            intensity = int((y / self.height) * 50)
            frame[y, :] = [intensity, intensity // 2, intensity // 4]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "CAMERA OFFLINE",  (150, 200), font, 1.2, (0, 217, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Maintenance Mode", (180, 250), font, 0.8, (139, 149, 176), 2, cv2.LINE_AA)
        cv2.putText(frame, time.strftime("%H:%M:%S"), (250, 300), font, 0.7, (0, 153, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (10, 10), (self.width - 10, self.height - 10), (0, 217, 255), 2)
        return frame

    def generate_frames(self):
        while True:
            frame = self.generate_placeholder_frame()
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)


class WebcamHelper:
    def __init__(self, device_index=0, resolution=(640, 480)):
        """Initialize the USB webcam using OpenCV."""
        self.device_index = device_index
        self.resolution = resolution
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {device_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    def get_frame(self):
        """Return the latest BGR frame array."""
        if not self.cap or not self.cap.isOpened():
            return None
        try:
            success, frame = self.cap.read()
            # cap.read() already returns BGR — no conversion needed.
            return frame if success else None
        except Exception:
            return None

    def generate_frames(self):
        """Generator that yields JPEG-encoded frames for web streaming."""
        # NOTE: no finally/self.stop() here — the camera is a shared resource.
        # Releasing it on stream drop (e.g. browser refresh) would make it
        # unavailable to the NOVA agent and to the next page load.
        # Call stop() explicitly on application shutdown only.
        try:
            while self.cap and self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    time.sleep(0.01)
                    continue
                success_encode, buffer = cv2.imencode('.jpg', frame)
                if not success_encode:
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Webcam streaming error: {e}")

    def stop(self):
        if self.cap:
            self.cap.release()
            print("Webcam resources released.")