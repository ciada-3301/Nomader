import cv2
import numpy as np
import argparse
import json
import time
from pathlib import Path

def calibrate_camera(cols: int, rows: int, camera_index: int = 0):
    # termination criteria for subpixel corner detection
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    print(f"Starting calibration. Need 20 good frames of a {cols}x{rows} checkerboard.")
    print("Press 'c' to capture a frame when the checkerboard is detected.")
    print("Press 'q' to quit early.")
    
    good_frames = 0
    required_frames = 20
    last_capture_time = 0
    image_size = None
    
    while good_frames < required_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]
            
        # Find the chess board corners
        ret_corners, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        
        display_frame = frame.copy()
        
        if ret_corners:
            # Draw and display the corners
            cv2.drawChessboardCorners(display_frame, (cols, rows), corners, ret_corners)
            cv2.putText(display_frame, "Checkerboard Detected! Press 'c' to capture.", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Searching for checkerboard...", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
        cv2.putText(display_frame, f"Good Frames: {good_frames}/{required_frames}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
        cv2.imshow('Calibration', display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c') and ret_corners and (time.time() - last_capture_time > 1.0):
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            good_frames += 1
            last_capture_time = time.time()
            print(f"Captured frame {good_frames}/{required_frames}")
            
    cap.release()
    cv2.destroyAllWindows()
    
    if good_frames < required_frames:
        print("Calibration aborted.")
        return
        
    print("Calculating calibration parameters... this may take a moment.")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    
    if ret:
        print("Calibration successful.")
        fx = mtx[0][0]
        
        config_data = {
            "camera_matrix": mtx.tolist(),
            "dist_coeffs": dist.tolist(),
            "focal_length_px": float(fx),
            "image_size": list(image_size)
        }
        
        config_dir = Path(__file__).parent.parent / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "camera_calibration.json"
        
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=4)
            
        print(f"Calibration saved to {config_path}")
        print(f"Focal Length (fx): {fx}")
    else:
        print("Calibration failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate camera using a checkerboard pattern.")
    parser.add_argument("--cols", type=int, default=9, help="Number of inner corners per a chessboard row")
    parser.add_argument("--rows", type=int, default=6, help="Number of inner corners per a chessboard column")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use")
    
    args = parser.parse_args()
    calibrate_camera(args.cols, args.rows, args.camera)
