import threading
import time
import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple, List

class CameraManager:
    """
    Manages webcam capture in a background thread with a rolling buffer.
    Provides methods to retrieve the sharpest or latest frame.
    """
    def __init__(self, camera_index: int = 0, target_fps: int = 30):
        self.camera_index = camera_index
        self.target_fps = target_fps
        # 0.5 seconds buffer
        self.buffer_size = max(1, int(target_fps * 0.5))
        self.frame_buffer = deque(maxlen=self.buffer_size)
        
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> bool:
        if self._running:
            return True
            
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"[CameraManager] Error: Could not open camera {self.camera_index}")
            return False

        # Request 1080p
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        print(f"[CameraManager] Started capture on camera {self.camera_index} at 1080p")
        return True

    def stop(self) -> None:
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        print("[CameraManager] Stopped capture")

    def _capture_loop(self) -> None:
        while self._running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self.frame_buffer.append(frame)
            else:
                time.sleep(0.01)

    def _resize_frame(self, frame: np.ndarray, downscale_to: Optional[Tuple[int, int]]) -> np.ndarray:
        if downscale_to is not None:
            return cv2.resize(frame, downscale_to, interpolation=cv2.INTER_AREA)
        return frame

    def get_latest_frame(self, downscale_to: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """Returns the most recent frame in the buffer."""
        with self._lock:
            if not self.frame_buffer:
                return None
            frame = self.frame_buffer[-1].copy()
        
        return self._resize_frame(frame, downscale_to)

    def get_sharpest_frame(self, downscale_to: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """
        Scores all buffered frames using Laplacian variance and returns the sharpest one.
        """
        with self._lock:
            if not self.frame_buffer:
                return None
            # Shallow copy the list of frames to avoid holding lock during computation
            frames: List[np.ndarray] = list(self.frame_buffer)

        best_frame = None
        best_score = -1.0

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if score > best_score:
                best_score = score
                best_frame = frame

        if best_frame is None:
            return None

        return self._resize_frame(best_frame.copy(), downscale_to)

    def get_frame(self, downscale_to: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """Helper for compatibility with older code expecting get_frame()"""
        return self.get_latest_frame(downscale_to=downscale_to)
