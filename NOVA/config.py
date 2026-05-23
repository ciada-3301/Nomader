"""
NOVA Configuration
------------------
Central configuration dataclass for all NOVA subsystems.
Loads secrets from .env file in project root.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class VisionConfig:
    """Camera and object detection settings."""
    camera_resolution: tuple = (640, 480)
    camera_fps: int = 15
    detection_model: str = "yolov8n"           # yolov8n, yolov8n-seg, mobilenet_ssd
    detection_confidence: float = 0.45
    detection_iou: float = 0.5
    detection_input_size: int = 320            # YOLO input size (smaller = faster)
    obstacle_floor_fraction: float = 0.6       # bottom 60% of frame for floor obstacles


@dataclass
class NavigationConfig:
    """Path planning and motion control settings."""
    # Occupancy grid
    grid_resolution_cm: float = 5.0            # cm per cell
    grid_size_cells: int = 400                 # 400x400 = 20m x 20m arena
    grid_origin_cell: tuple = (200, 200)       # robot starts at center

    # Motion
    max_speed: int = 200                       # PWM 0-255
    cruise_speed: int = 150                    # normal navigation speed
    turn_speed: int = 120                      # in-place rotation speed
    min_speed: int = 80                        # below this motors stall

    # Obstacle avoidance (VFH)
    vfh_sector_count: int = 36                 # 10° per sector
    vfh_threshold: float = 0.3                 # density threshold for blocked sector
    obstacle_stop_cm: float = 20.0             # emergency stop distance
    obstacle_slow_cm: float = 50.0             # slow down distance

    # Waypoint following
    waypoint_reach_cm: float = 15.0            # close enough to advance to next wp
    stuck_timeout_s: float = 8.0               # re-plan if no progress for this long
    max_replan_attempts: int = 3

    # Control loop
    control_hz: float = 10.0                   # motor command rate


@dataclass
class PlannerConfig:
    """LLM task planner settings."""
    llm_base_url: str = "https://ollama.com/v1"
    llm_api_key: str = ""                      # loaded from env
    llm_model: str = "gemma4:31b-cloud"                # Ollama model name
    llm_temperature: float = 0.1               # low temp for deterministic planning
    llm_max_tokens: int = 1024
    llm_timeout_s: float = 30.0

    def __post_init__(self):
        # Load API key from environment if not set
        if not self.llm_api_key:
            self.llm_api_key = os.environ.get("NOVA_API_KEY", "ollama")


@dataclass
class MemoryConfig:
    """Persistent memory settings."""
    memory_dir: str = ""                       # defaults to ~/.nova/
    autosave_interval_s: int = 30
    max_task_history: int = 200
    max_object_sightings: int = 500

    def __post_init__(self):
        if not self.memory_dir:
            self.memory_dir = str(Path.home() / ".nova")


@dataclass
class HardwareConfig:
    """Serial and sensor settings."""
    serial_port: str = "/dev/ttyUSB0"
    baud_rate: int = 115200
    serial_timeout: float = 1.0

    # Ultrasonic sensor (readings come from Arduino)
    ultrasonic_enabled: bool = True
    ultrasonic_max_range_cm: float = 300.0
    ultrasonic_mount_offset_cm: float = -10.0  # 10cm below camera

    # Wheel geometry (for odometry)
    wheel_base_cm: float = 22.0                # distance between left/right wheel centers
    wheel_diameter_cm: float = 6.5             # wheel diameter
    ticks_per_revolution: int = 20             # encoder ticks (if available)


@dataclass
class NovaConfig:
    """Top-level NOVA configuration."""
    vision: VisionConfig = field(default_factory=VisionConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    # Agent behaviour
    agent_loop_hz: float = 2.0                 # how fast the agent thinks
    verbose: bool = True                       # print debug info
    autonomous_enabled: bool = True            # can be disabled for manual-only mode
