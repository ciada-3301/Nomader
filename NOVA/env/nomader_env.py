"""
nomader_env.py
--------------
Simulated 2D top-down environment for Nomader 6WD rover.
Mimics what the real robot will face:
  - A birds-eye arena with obstacles
  - A "camera frame" (rendered top-down view, cropped to robot FOV)
  - Sensor readings (IMU, encoders, ultrasonic proximity)
  - Target object represented as a colored blob (SAM2 mask proxy)

Observation returned per step:
  obs = {
      "target_centroid" : (2,)   normalised (cx, cy) in [0,1]
      "target_bbox"     : (4,)   normalised (x,y,w,h)
      "depth_estimate"  : (1,)   heuristic: bbox-area proxy
      "mask_encoding"   : (32,)  flat encoding from MaskEncoder CNN
      "imu"             : (6,)   [ax,ay,az,gx,gy,gz]  (simulated noise)
      "wheel_encoders"  : (6,)   per-wheel speed (6WD)
      "prev_action"     : (2,)   [throttle, steering]
  }
"""

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces


# ── constants ────────────────────────────────────────────────────────────────
ARENA_W, ARENA_H = 600, 600          # pixels = cm in sim
ROBOT_RADIUS     = 18                # px
CAM_FOV_W        = 160               # camera frame width (px) fed to CNN
CAM_FOV_H        = 120               # camera frame height
MAX_STEPS        = 400
TARGET_RADIUS    = 20
OBSTACLE_COUNT   = 8
COLLISION_DIST   = ROBOT_RADIUS + 5


class NomaderEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, render_mode=None, seed=None):
        super().__init__()
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)

        # ── action space: [throttle, steering] both in [-1, 1] ──
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # ── observation space ────────────────────────────────────
        #  We expose a flat dict; the network will unpack it.
        self.observation_space = spaces.Dict({
            "target_centroid": spaces.Box(0.0, 1.0,  shape=(2,),  dtype=np.float32),
            "target_bbox":     spaces.Box(0.0, 1.0,  shape=(4,),  dtype=np.float32),
            "depth_estimate":  spaces.Box(0.0, 1.0,  shape=(1,),  dtype=np.float32),
            "cam_frame":       spaces.Box(0,   255,   shape=(CAM_FOV_H, CAM_FOV_W, 3),
                                         dtype=np.uint8),
            "imu":             spaces.Box(-10.0, 10.0, shape=(6,), dtype=np.float32),
            "wheel_encoders":  spaces.Box(-1.0,  1.0,  shape=(6,), dtype=np.float32),
            "prev_action":     spaces.Box(-1.0,  1.0,  shape=(2,), dtype=np.float32),
        })

        # internal state
        self._robot_pos   = np.zeros(2, dtype=np.float32)
        self._robot_angle = 0.0          # radians
        self._target_pos  = np.zeros(2, dtype=np.float32)
        self._obstacles   = []           # list of (x,y,r)
        self._prev_action = np.zeros(2,  dtype=np.float32)
        self._step_count  = 0
        self._prev_dist   = 0.0

        self._arena = None               # cached background render

    # ── reset ────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count  = 0
        self._prev_action = np.zeros(2, dtype=np.float32)

        # place robot near centre with random angle
        self._robot_pos   = np.array([ARENA_W/2, ARENA_H/2], dtype=np.float32)
        self._robot_pos  += self._rng.uniform(-60, 60, 2).astype(np.float32)
        self._robot_angle = self._rng.uniform(0, 2*np.pi)

        # place target away from robot
        while True:
            tx = self._rng.uniform(TARGET_RADIUS+10, ARENA_W-TARGET_RADIUS-10)
            ty = self._rng.uniform(TARGET_RADIUS+10, ARENA_H-TARGET_RADIUS-10)
            if np.linalg.norm([tx-self._robot_pos[0], ty-self._robot_pos[1]]) > 150:
                self._target_pos = np.array([tx, ty], dtype=np.float32)
                break

        # random obstacles (none overlapping robot or target)
        self._obstacles = []
        attempts = 0
        while len(self._obstacles) < OBSTACLE_COUNT and attempts < 500:
            ox = self._rng.uniform(30, ARENA_W-30)
            oy = self._rng.uniform(30, ARENA_H-30)
            r  = self._rng.uniform(15, 35)
            if (np.linalg.norm([ox-self._robot_pos[0], oy-self._robot_pos[1]]) > r+60 and
                np.linalg.norm([ox-self._target_pos[0], oy-self._target_pos[1]]) > r+40):
                self._obstacles.append((ox, oy, r))
            attempts += 1

        self._prev_dist = np.linalg.norm(self._target_pos - self._robot_pos)
        self._arena = None               # invalidate cache

        obs  = self._get_obs()
        info = {}
        return obs, info

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        throttle, steering = float(action[0]), float(action[1])

        # ── kinematics (differential drive proxy for 6WD) ──
        speed     = throttle * 4.0          # px per step
        turn_rate = steering * 0.08         # rad per step

        self._robot_angle += turn_rate
        dx = np.cos(self._robot_angle) * speed
        dy = np.sin(self._robot_angle) * speed
        new_pos = self._robot_pos + np.array([dx, dy], dtype=np.float32)

        # wall clamp
        new_pos[0] = np.clip(new_pos[0], ROBOT_RADIUS, ARENA_W-ROBOT_RADIUS)
        new_pos[1] = np.clip(new_pos[1], ROBOT_RADIUS, ARENA_H-ROBOT_RADIUS)

        # obstacle collision check
        collided = False
        for ox, oy, r in self._obstacles:
            if np.linalg.norm(new_pos - np.array([ox, oy])) < r + ROBOT_RADIUS:
                collided = True
                break

        if not collided:
            self._robot_pos = new_pos

        self._prev_action = action
        self._step_count += 1

        # ── reward ───────────────────────────────────────────────
        dist      = np.linalg.norm(self._target_pos - self._robot_pos)
        delta     = self._prev_dist - dist          # positive = getting closer
        self._prev_dist = dist

        # centroid alignment bonus (how centred is the target in camera)
        cx_norm, cy_norm = self._target_centroid_in_cam()
        align = 1.0 - 2*abs(cx_norm - 0.5)         # 1 when dead centre, 0 at edge

        reward  =  delta * 0.4                      # approach reward
        reward +=  align * 0.1                      # keep target centred
        reward += -0.01                             # time penalty
        reward += -1.5 if collided else 0.0         # collision

        terminated = False
        if dist < TARGET_RADIUS + ROBOT_RADIUS:
            reward    += 20.0                       # arrival bonus
            terminated = True

        truncated = self._step_count >= MAX_STEPS

        obs  = self._get_obs()
        info = {"distance": dist, "collided": collided}
        return obs, reward, terminated, truncated, info

    # ── render ───────────────────────────────────────────────────────────────
    def render(self):
        frame = self._render_arena()
        if self.render_mode == "human":
            cv2.imshow("Nomader Sim", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        return frame

    # ── internal helpers ─────────────────────────────────────────────────────
    def _render_arena(self):
        img = np.ones((ARENA_H, ARENA_W, 3), dtype=np.uint8) * 40  # dark bg

        # grid
        for i in range(0, ARENA_W, 50):
            cv2.line(img, (i, 0), (i, ARENA_H), (55, 55, 55), 1)
            cv2.line(img, (0, i), (ARENA_W, i), (55, 55, 55), 1)

        # obstacles
        for ox, oy, r in self._obstacles:
            cv2.circle(img, (int(ox), int(oy)), int(r), (100, 80, 60), -1)
            cv2.circle(img, (int(ox), int(oy)), int(r), (150, 120, 90),  1)

        # target (green blob — SAM2 mask proxy)
        tx, ty = int(self._target_pos[0]), int(self._target_pos[1])
        cv2.circle(img, (tx, ty), TARGET_RADIUS, (30, 200, 80), -1)
        cv2.circle(img, (tx, ty), TARGET_RADIUS, (80, 255, 120), 2)

        # robot body
        rx, ry = int(self._robot_pos[0]), int(self._robot_pos[1])
        cv2.circle(img, (rx, ry), ROBOT_RADIUS, (60, 120, 220), -1)
        # heading arrow
        ex = int(rx + np.cos(self._robot_angle) * ROBOT_RADIUS)
        ey = int(ry + np.sin(self._robot_angle) * ROBOT_RADIUS)
        cv2.arrowedLine(img, (rx, ry), (ex, ey), (200, 230, 255), 2)

        # distance line
        cv2.line(img, (rx, ry), (tx, ty), (80, 80, 80), 1)

        # HUD
        dist = np.linalg.norm(self._target_pos - self._robot_pos)
        cv2.putText(img, f"dist:{dist:.1f}  step:{self._step_count}",
                    (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        return img

    def _get_cam_frame(self):
        """
        Simulated camera: crop a forward-facing rectangle from the arena
        centred on the robot, oriented along heading.
        """
        arena = self._render_arena()

        # build rotation matrix around robot centre
        cx, cy = self._robot_pos
        M = cv2.getRotationMatrix2D((cx, cy),
                                    -np.degrees(self._robot_angle), 1.0)
        rotated = cv2.warpAffine(arena, M, (ARENA_W, ARENA_H),
                                 flags=cv2.INTER_LINEAR,
                                 borderValue=(20, 20, 20))

        # crop FOV rectangle in front of robot
        fov_w, fov_h = 160, 120
        x1 = int(cx) - fov_w // 2
        y1 = int(cy) - fov_h         # in front of robot
        y2 = int(cy)
        x2 = x1 + fov_w
        x1c, x2c = max(x1,0), min(x2, ARENA_W)
        y1c, y2c = max(y1,0), min(y2, ARENA_H)
        crop = rotated[y1c:y2c, x1c:x2c]

        if crop.shape[0] == 0 or crop.shape[1] == 0:
            crop = np.zeros((fov_h, fov_w, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (CAM_FOV_W, CAM_FOV_H))
        return crop

    def _target_centroid_in_cam(self):
        """Return normalised (cx, cy) of target in the camera frame [0,1]."""
        # transform target into robot-local space
        dx = self._target_pos[0] - self._robot_pos[0]
        dy = self._target_pos[1] - self._robot_pos[1]
        cos_a = np.cos(-self._robot_angle)
        sin_a = np.sin(-self._robot_angle)
        lx =  cos_a * dx - sin_a * dy   # right in robot frame
        ly =  sin_a * dx + cos_a * dy   # forward in robot frame (negative = front)

        # map into cam frame normalised coords
        cx = (lx + CAM_FOV_W/2) / CAM_FOV_W
        cy = (-ly) / CAM_FOV_H          # forward maps to top of frame
        cx = float(np.clip(cx, 0.0, 1.0))
        cy = float(np.clip(cy, 0.0, 1.0))
        return cx, cy

    def _get_target_bbox(self):
        """Bounding box of target blob in camera frame, normalised."""
        cx, cy = self._target_centroid_in_cam()
        dist = np.linalg.norm(self._target_pos - self._robot_pos)
        # apparent size decreases with distance
        app_r = max(0.01, TARGET_RADIUS / max(dist, 1.0) * 40.0)
        w = np.clip(app_r / CAM_FOV_W, 0.01, 1.0)
        h = np.clip(app_r / CAM_FOV_H, 0.01, 1.0)
        x = np.clip(cx - w/2, 0.0, 1.0)
        y = np.clip(cy - h/2, 0.0, 1.0)
        return np.array([x, y, w, h], dtype=np.float32)

    def _get_obs(self):
        cx, cy      = self._target_centroid_in_cam()
        centroid    = np.array([cx, cy], dtype=np.float32)
        bbox        = self._get_target_bbox()
        depth_est   = np.array([bbox[2] * bbox[3]], dtype=np.float32)  # area proxy

        # simulated IMU (add Gaussian noise)
        imu = self._rng.normal(0, 0.05, 6).astype(np.float32)
        imu[1] += 9.8   # gravity on Y (simulated)

        # wheel encoders: speed proportional to last throttle, small noise per wheel
        enc = np.full(6, self._prev_action[0], dtype=np.float32)
        enc += self._rng.normal(0, 0.02, 6).astype(np.float32)

        cam_frame = self._get_cam_frame()

        return {
            "target_centroid": centroid,
            "target_bbox":     bbox,
            "depth_estimate":  depth_est,
            "cam_frame":       cam_frame,
            "imu":             imu,
            "wheel_encoders":  enc,
            "prev_action":     self._prev_action.copy(),
        }

    def close(self):
        cv2.destroyAllWindows()