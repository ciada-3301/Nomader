"""
NOVA Configuration
------------------
Central config for all NOVA subsystems. Secrets loaded from .env.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class VisionConfig:
    camera_resolution: tuple = (640, 480)
    camera_fps: int = 15
    # Visual obstacle detection (floor-plane analysis)
    floor_fraction: float = 0.55       # bottom % of frame = floor zone
    edge_density_threshold: float = 0.08  # fraction of edge pixels that signals an obstacle
    obstacle_width_min_frac: float = 0.10  # obstacle must span >= 10% of frame width


@dataclass
class NavigationConfig:
    # Occupancy grid
    grid_resolution_cm: float = 5.0
    grid_size_cells: int = 400
    grid_origin_cell: tuple = (200, 200)

    # Motion — tuned for 6WD
    max_speed: int = 200
    cruise_speed: int = 150
    turn_speed: int = 120
    min_speed: int = 80

    # VFH obstacle avoidance
    vfh_sector_count: int = 36
    vfh_threshold: float = 0.3

    # Obstacle distances — visual-only (no ultrasonic)
    # These are "visual proximity scores" (0–1) not cm
    obstacle_stop_score: float = 0.65
    obstacle_slow_score: float = 0.35

    # Waypoint following
    waypoint_reach_cm: float = 15.0
    stuck_timeout_s: float = 8.0
    max_replan_attempts: int = 3
    control_hz: float = 10.0

    # Left/right correction factor
    # NOTE: physical motor wiring is swapped — drive(right_pwm, left_pwm)
    # This is a calibration scalar for individual motor bias correction.
    left_motor_trim: float = 1.0   # multiply left PWM by this
    right_motor_trim: float = 1.0  # multiply right PWM by this


@dataclass
class PlannerConfig:
    llm_base_url: str = "https://ollama.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gemma4:31b-cloud"
    llm_temperature: float = 0.15
    llm_max_tokens: int = 2048
    llm_timeout_s: float = 45.0
    max_iterations: int = 30        # max tool calls before forced stop
    max_retries: int = 5            # max review-triggered retries
    mission_timeout_s: float = 180  # hard wall

    def __post_init__(self):
        if not self.llm_api_key:
            self.llm_api_key = os.environ.get("NOVA_API_KEY", "ollama")


@dataclass
class MemoryConfig:
    memory_dir: str = ""
    autosave_interval_s: int = 30
    max_task_history: int = 200
    max_object_sightings: int = 500
    hot_memory_max_turns: int = 10   # rolling window of recent actions

    def __post_init__(self):
        if not self.memory_dir:
            self.memory_dir = str(Path.home() / ".nova")


@dataclass
class HardwareConfig:
    serial_port: str = "/dev/ttyUSB0"
    baud_rate: int = 115200
    serial_timeout: float = 1.0



    # Wheel geometry (6WD differential)
    wheel_base_cm: float = 22.0
    wheel_diameter_cm: float = 6.5
    ticks_per_revolution: int = 20

    # Camera mount
    camera_tilt_deg: float = -15.0   # negative = tilted slightly downward


@dataclass
class NovaConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    agent_loop_hz: float = 2.0
    verbose: bool = True
    autonomous_enabled: bool = True
