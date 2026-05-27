import os
import time
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from scipy.interpolate import splprep, splev

try:
    import openvino as ov
except ImportError:
    ov = None
    print("[DepthNavigator] Warning: openvino not installed. Depth estimation will not be available.")

class DepthNavigator:
    """
    Integrates Depth Anything V2 with OpenVINO on CPU/AUTO.
    Projects depth maps into 3D, performs floor slab slicing,
    builds an occupancy grid, plans obstacle-free paths using A*,
    and overlays path visual traces on the WebApp feed.
    """
    def __init__(self, config: Any):
        self.config = config
        self.compiled_model = None
        self.input_layer = None
        self.output_layer = None
        
        # Grid parameters
        self.grid_res = 0.05  # 5 cm per cell
        self.x_min, self.x_max = -1.5, 1.5   # 3.0 meters wide
        self.z_min, self.z_max = 0.3, 3.5    # 3.2 meters forward
        
        self.width_cells = int((self.x_max - self.x_min) / self.grid_res)
        self.height_cells = int((self.z_max - self.z_min) / self.grid_res)
        
        # Intrinsic parameters (assuming 640x480 resolution)
        self.w, self.h = 640, 480
        self.fx = self.fy = self.w * 0.8
        self.cx, self.cy = self.w / 2.0, self.h / 2.0
        
        # Load OpenVINO model
        self._load_model()
        
    def _load_model(self):
        if ov is None:
            return
            
        model_path = self.config.vision.depth_model_path
        device = self.config.vision.depth_device
        
        if not os.path.exists(model_path):
            print(f"[DepthNavigator] Warning: Model path {model_path} not found.")
            return
            
        try:
            print(f"[DepthNavigator] Loading model {model_path} on {device}...")
            core = ov.Core()
            model = core.read_model(model_path)
            self.compiled_model = core.compile_model(model, device)
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)
            print("[DepthNavigator] Compiled model successfully.")
        except Exception as e:
            print(f"[DepthNavigator] Error loading depth model: {e}")

    def run_inference(self, frame: np.ndarray) -> np.ndarray:
        """
        Runs Depth Anything V2 inference on the frame and returns the raw depth map.
        """
        if self.compiled_model is None or frame is None:
            # Return dummy depth map
            return np.ones((self.h, self.w), dtype=np.float32) * 2.0
            
        # Get expected input dimensions
        input_shape = self.input_layer.shape
        # Typically [1, 3, H, W]
        net_h, net_w = input_shape[2], input_shape[3]
        
        # Preprocess
        resized = cv2.resize(frame, (net_w, net_h), interpolation=cv2.INTER_AREA)
        # Normalize
        img = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # HWC to CHW
        img = img.transpose((2, 0, 1))
        # Add batch dimension [1, 3, H, W]
        input_data = np.expand_dims(img, axis=0)
        
        # Inference
        results = self.compiled_model([input_data])
        raw_depth = results[self.output_layer][0] # shape should be [H, W] or similar
        
        # If shape is [1, H, W] or HWC, squeeze or resize
        if len(raw_depth.shape) == 3:
            raw_depth = np.squeeze(raw_depth, axis=0)
            
        # Resize depth map back to original frame dimensions
        depth_map = cv2.resize(raw_depth, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return depth_map

    def get_point_cloud(self, depth_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects a depth map into a 3D point cloud.
        Returns:
            points: (N, 3) float array of (X, Y, Z) coordinates.
            valid_mask: Boolean mask of valid depth pixels.
        """
        # Map depth to physical range
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max > d_min:
            depth_norm = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_map)
            
        # Depth Anything is disparity-like (larger is closer) -> invert
        # Scale to [0.3m, 3.5m]
        depth_metric = 0.3 + (1.0 - depth_norm) * 3.2
        
        # Create pixel coordinate grid
        u, v = np.meshgrid(np.arange(self.w), np.arange(self.h))
        
        # Project using camera intrinsics
        z = depth_metric
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        return points, depth_metric

    def detect_floor_and_obstacles(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Identifies floor pixels and obstacle pixels using Y-axis (height) binning.
        Returns:
            floor_mask: boolean mask for points that belong to the floor
            obstacle_mask: boolean mask for points that belong to obstacles
            y_floor: calculated floor height level
        """
        y_coords = points[:, 1]
        
        # Slicing: create a histogram of heights between -0.8m (height above cam) and 1.2m (below cam)
        valid_indices = np.where((y_coords > -0.8) & (y_coords < 1.2))[0]
        if len(valid_indices) > 0:
            hist, bin_edges = np.histogram(y_coords[valid_indices], bins=40)
            max_bin = np.argmax(hist)
            y_floor = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2.0
        else:
            y_floor = 0.45  # fallback
            
        # Define floor slab
        floor_thickness = 0.08  # 8 cm thickness
        floor_mask = np.abs(y_coords - y_floor) < floor_thickness
        
        # Obstacles are anything above the floor level (remembering smaller Y is higher in Y-down!)
        obstacle_mask = (y_coords < y_floor - floor_thickness) & (y_coords > -0.8)
        
        return floor_mask, obstacle_mask, y_floor

    def build_occupancy_grid(self, points: np.ndarray, obstacle_mask: np.ndarray) -> np.ndarray:
        """
        Constructs a 2D occupancy grid from obstacle points.
        """
        grid = np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)
        
        # Filter obstacle points
        obs_points = points[obstacle_mask]
        
        # Map 3D points to grid cells
        # x_min -> x_max maps to 0 -> width_cells
        # z_min -> z_max maps to 0 -> height_cells
        for pt in obs_points:
            x, _, z = pt
            if self.x_min <= x < self.x_max and self.z_min <= z < self.z_max:
                col = int((x - self.x_min) / self.grid_res)
                row = int((z - self.z_min) / self.grid_res)
                if 0 <= col < self.width_cells and 0 <= row < self.height_cells:
                    grid[row, col] = 1
                    
        # Dilate grid to account for rover footprint (approx 22cm width -> radius ~15cm -> ~3 cells)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_grid = cv2.dilate(grid, kernel)
        
        return dilated_grid

    def target_to_grid(self, target_bbox: List[int], points: np.ndarray) -> Tuple[int, int]:
        """
        Finds the 3D position of the target box and converts to grid cell (row, col).
        """
        x1, y1, x2, y2 = target_bbox
        
        # Reshape point cloud back to 2D grid of points (H, W, 3) to index with bbox
        pts_2d = points.reshape(self.h, self.w, 3)
        
        # Get target points
        target_pts = pts_2d[y1:y2, x1:x2].reshape(-1, 3)
        
        if len(target_pts) > 0:
            # Use median to avoid outliers
            x_target = np.median(target_pts[:, 0])
            z_target = np.median(target_pts[:, 2])
        else:
            # Fallback to direct projection of center pixel with 1.5 meters depth
            cx_pixel = (x1 + x2) / 2.0
            z_target = 1.5
            x_target = (cx_pixel - self.cx) * z_target / self.fx
            
        col = int((x_target - self.x_min) / self.grid_res)
        row = int((z_target - self.z_min) / self.grid_res)
        
        col = max(0, min(self.width_cells - 1, col))
        row = max(0, min(self.height_cells - 1, row))
        
        return row, col, x_target, z_target

    def run_astar(self, grid: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        A* pathfinding on the 2D grid.
        """
        # Node class
        class Node:
            def __init__(self, position: Tuple[int, int], parent=None):
                self.position = position
                self.parent = parent
                self.g = 0
                self.h = 0
                self.f = 0
                
            def __eq__(self, other):
                return self.position == other.position
                
            def __hash__(self):
                return hash(self.position)

        start_node = Node(start)
        end_node = Node(end)
        
        open_set = {start_node}
        closed_set = set()
        
        # 8-connected movements
        movements = [
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        
        start_time = time.time()
        
        while open_set:
            # Timeout A* if taking too long to protect loop
            if time.time() - start_time > 0.05:
                break
                
            # Find node with lowest F score
            current_node = min(open_set, key=lambda n: n.f)
            
            # Remove current from open, add to closed
            open_set.remove(current_node)
            closed_set.add(current_node)
            
            # Check if reached destination
            # If we are within 2 cells of end, call it a success
            dist_to_end = np.hypot(current_node.position[0] - end_node.position[0],
                                   current_node.position[1] - end_node.position[1])
            if dist_to_end <= 2.0 or current_node == end_node:
                path = []
                curr = current_node
                while curr is not None:
                    path.append(curr.position)
                    curr = curr.parent
                return path[::-1] # return reversed path
                
            # Generate children
            for move in movements:
                new_pos = (current_node.position[0] + move[0], current_node.position[1] + move[1])
                
                # Check boundaries
                if not (0 <= new_pos[0] < self.height_cells and 0 <= new_pos[1] < self.width_cells):
                    continue
                    
                # Check obstacle
                if grid[new_pos[0], new_pos[1]] == 1:
                    continue
                    
                child = Node(new_pos, current_node)
                
                if child in closed_set:
                    continue
                    
                # Calculate costs
                step_cost = 1.414 if (move[0] != 0 and move[1] != 0) else 1.0
                child.g = current_node.g + step_cost
                child.h = np.hypot(child.position[0] - end_node.position[0],
                                   child.position[1] - end_node.position[1])
                child.f = child.g + child.h
                
                # Check if child is in open set and has a worse cost
                existing_open = [n for n in open_set if n == child]
                if existing_open:
                    if child.g >= existing_open[0].g:
                        continue
                    open_set.remove(existing_open[0])
                    
                open_set.add(child)
                
        return None

    def smooth_path(self, grid_path: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """
        Smoothes the path using spline interpolation.
        Returns path in 3D relative coords (X, Z).
        """
        # Convert back to (X, Z) coords
        pts_xz = []
        for row, col in grid_path:
            x = col * self.grid_res + self.x_min + self.grid_res / 2.0
            z = row * self.grid_res + self.z_min + self.grid_res / 2.0
            pts_xz.append((x, z))
            
        if len(pts_xz) < 3:
            return pts_xz
            
        try:
            # Downsample to avoid spline fitting overhead
            step = max(1, len(pts_xz) // 8)
            downsampled = pts_xz[::step]
            if downsampled[-1] != pts_xz[-1]:
                downsampled.append(pts_xz[-1])
                
            if len(downsampled) < 3:
                return pts_xz
                
            x_pts = [p[0] for p in downsampled]
            z_pts = [p[1] for p in downsampled]
            
            # splprep expects list of arrays
            tck, u = splprep([x_pts, z_pts], s=0.01, k=min(3, len(downsampled)-1))
            u_new = np.linspace(0, 1.0, 20)
            x_new, z_new = splev(u_new, tck)
            
            smoothed = list(zip(x_new, z_new))
            return smoothed
        except Exception:
            return pts_xz

    def navigate(self, frame: np.ndarray, target_bbox: Optional[List[int]] = None) -> Tuple[np.ndarray, Optional[List[Tuple[float, float]]], Optional[Tuple[float, float]]]:
        """
        High-level navigation pipeline. Takes camera frame and target box.
        Draws overlays and plans A* path.
        Returns:
            overlay_frame: camera frame with green floor & path overlay.
            smoothed_path: list of (X, Z) waypoints or None.
            target_3d: (X, Z) coords of target or None.
        """
        # 1. Inference
        depth_map = self.run_inference(frame)
        
        # 2. Point cloud projection
        points, depth_metric = self.get_point_cloud(depth_map)
        
        # 3. Slicing floor / obstacles
        floor_mask, obstacle_mask, y_floor = self.detect_floor_and_obstacles(points)
        
        # 4. Occupancy Grid
        grid = self.build_occupancy_grid(points, obstacle_mask)
        
        # 5. Determine start and target
        # Start is at x=0.0, z=0.3 (centered, just in front of camera)
        start_col = int((0.0 - self.x_min) / self.grid_res)
        start_row = int((self.z_min - self.z_min) / self.grid_res)
        
        start_col = max(0, min(self.width_cells - 1, start_col))
        start_row = max(0, min(self.height_cells - 1, start_row))
        
        smoothed_path = None
        target_3d = None
        
        if target_bbox is not None:
            # Get target grid pos and 3D pos
            tgt_row, tgt_col, x_tgt, z_tgt = self.target_to_grid(target_bbox, points)
            target_3d = (x_tgt, z_tgt)
            
            # Run A*
            grid_path = self.run_astar(grid, (start_row, start_col), (tgt_row, tgt_col))
            if grid_path is not None:
                smoothed_path = self.smooth_path(grid_path)
                
        # 6. Render premium overlay on camera frame
        overlay = frame.copy()
        
        # Highlighting floor pixels green (translucent)
        floor_idx = floor_mask.reshape(self.h, self.w)
        green_mask = np.zeros_like(overlay)
        # Harmonious digital green color (BGR: 0, 220, 0)
        green_mask[floor_idx] = [0, 220, 0]
        
        # Blending green floor
        cv2.addWeighted(overlay, 0.7, green_mask, 0.3, 0, dst=overlay)
        
        # Trace path (smoothed_path is a list of (X, Z) coordinates)
        if smoothed_path is not None and len(smoothed_path) >= 2:
            pts_2d = []
            for x, z in smoothed_path:
                # Project back to 2D
                u = int(x * self.fx / z + self.cx)
                v = int(y_floor * self.fy / z + self.cy)
                if 0 <= u < self.w and 0 <= v < self.h:
                    pts_2d.append([u, v])
                    
            if len(pts_2d) >= 2:
                pts_arr = np.array(pts_2d, dtype=np.int32).reshape((-1, 1, 2))
                # Premium Cyan/Blue path (BGR: 255, 191, 0)
                cv2.polylines(overlay, [pts_arr], isClosed=False, color=(255, 191, 0), thickness=4, lineType=cv2.LINE_AA)
                
                # Draw starting node with micro-animation halo
                cv2.circle(overlay, tuple(pts_2d[0]), 7, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(overlay, tuple(pts_2d[0]), 10, (255, 191, 0), 2, cv2.LINE_AA)
                
                # Draw end waypoint
                cv2.drawMarker(overlay, tuple(pts_2d[-1]), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=15, thickness=3, line_type=cv2.LINE_AA)
                
        # Draw target bbox
        if target_bbox is not None:
            x1, y1, x2, y2 = target_bbox
            # Neon purple / magenta bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (240, 16, 160), 2, cv2.LINE_AA)
            cv2.putText(overlay, "TARGET LOCKED", (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 16, 160), 2, cv2.LINE_AA)
            
        return overlay, smoothed_path, target_3d
