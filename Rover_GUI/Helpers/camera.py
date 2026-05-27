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
        self.processed_frame = None

    def get_frame(self):
        """Return the latest BGR frame array."""
        try:
            return self.picam2.capture_array()
        except Exception:
            return None

    def get_latest_frame(self, downscale_to=None):
        frame = self.get_frame()
        if frame is None:
            return None
        if downscale_to is not None:
            return cv2.resize(frame, downscale_to, interpolation=cv2.INTER_AREA)
        return frame

    def generate_frames(self):
        """Generator that yields JPEG-encoded frames for web streaming."""
        try:
            while True:
                if self.processed_frame is not None:
                    frame = self.processed_frame
                else:
                    frame = self.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                success, buffer = cv2.imencode('.jpg', frame)
                if not success:
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.03)
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
        self.processed_frame = None
        print("⚠️  Mock camera initialized (physical camera unavailable)")

    def get_frame(self):
        return self.generate_placeholder_frame()

    def get_latest_frame(self, downscale_to=None):
        frame = self.get_frame()
        if frame is None:
            return None
        if downscale_to is not None:
            return cv2.resize(frame, downscale_to, interpolation=cv2.INTER_AREA)
        return frame

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
            if self.processed_frame is not None:
                frame = self.processed_frame
            else:
                frame = self.generate_placeholder_frame()
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)


import threading

class WebcamHelper:
    def __init__(self, device_index=0, resolution=(640, 480)):
        """Initialize the USB webcam using OpenCV with a background reading thread."""
        self.device_index = device_index
        self.resolution = resolution
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {device_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        self.latest_frame = None
        self.processed_frame = None
        self.lock = threading.Lock()
        self.running = True
        
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        """Background thread that constantly reads from the camera."""
        while self.running and self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                with self.lock:
                    self.latest_frame = frame.copy()
            else:
                time.sleep(0.01)

    def get_frame(self):
        """Return the latest BGR frame array."""
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None

    def get_latest_frame(self, downscale_to=None):
        with self.lock:
            if self.latest_frame is None:
                return None
            frame = self.latest_frame.copy()
        if downscale_to is not None:
            return cv2.resize(frame, downscale_to, interpolation=cv2.INTER_AREA)
        return frame

    def generate_frames(self):
        """Generator that yields JPEG-encoded frames for web streaming."""
        try:
            while self.running:
                with self.lock:
                    if self.processed_frame is not None:
                        frame = self.processed_frame.copy()
                    else:
                        frame = self.latest_frame.copy() if self.latest_frame is not None else None
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                success_encode, buffer = cv2.imencode('.jpg', frame)
                if not success_encode:
                    time.sleep(0.01)
                    continue
                    
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.03)  # limit stream to ~30fps
        except Exception as e:
            print(f"Webcam streaming error: {e}")

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            print("Webcam resources released.")