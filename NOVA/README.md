# NOVA v2 — Reactive LangGraph Agent

**N**omader's **O**nboard **V**ision & **A**utonomy — upgraded from a static
plan-execute loop to a fully reactive LangGraph agent inspired by Shifu.

---

## What changed (v1 → v2)

| Area | v1 | v2 |
|---|---|---|
| **Planning** | Fixed JSON plan, executed step-by-step | LLM decides tools & order each turn |
| **Failure recovery** | Stops and reports | Review node retries with feedback |
| **Obstacle detection** | Ultrasonic only | Visual edge-density analysis (no sensor needed) |
| **Obstacle avoidance** | Stop and replan | Reactive dodge + replan with map update |
| **Motor fix** | Swap applied inconsistently | `_send_motors()` handles swap + trim always |
| **Memory** | Cold JSON only | Hot (rolling window) + Cold, injected into LLM |
| **New tools** | — | `spin_search`, `drive_until_clear`, `read_file`, `write_file`, `list_files`, `ask_user` |

---

## Architecture

```
User command
     │
┌────▼────┐    1 LLM call:
│  plan   │    route (SIMPLE/COMPLEX) + sketch plan
└────┬────┘
     │ needs clarification?
┌────▼────────┐   NO
│   clarify   │──────────────────────────┐
└─────────────┘                          │
                                   ┌─────▼──────┐
                                   │  execute   │◄──────────┐
                                   │  (LLM)     │           │
                                   └─────┬──────┘           │
                               tool calls│                   │
                                  ┌──────▼──────┐           │
                                  │ tools_node  │           │
                                  └──────┬──────┘           │
                                         │                   │
                                   ┌─────▼──────┐  RETRY    │
                                   │   review   │───────────┘
                                   └─────┬──────┘
                                      PASS│
                                        END
```

### Background loops (always running)

```
_hardware_loop (10 Hz)
  ├── odometry.update(left_pwm, right_pwm)
  ├── obstacle.update(camera_frame)         ← visual edge detection
  └── driver.update(frame)                  ← reactive steering / replanning
```

---

## Visual Obstacle Detection

No ultrasonic sensor. The `VisualObstacleDetector` analyses the **bottom
N% of the camera frame** (the floor zone):

1. Convert to grayscale → Gaussian blur → Canny edges
2. Compute **edge pixel density** in the floor zone
3. If density > threshold → obstacle present
4. Find the **lateral centroid** of edge pixels → tells us which side to dodge

This approach is inspired by classic desktop robots that used edge analysis
to understand their environment without any range sensors.

---

## Motor Wiring Fix

The Nomader's motor labels are physically swapped on the PCB.

**The fix** is in `NavigationDriver._send_motors(logical_left, logical_right)`:

```python
# Physical swap: robot.motor.drive(physical_right, physical_left)
self.robot.motor.drive(right, left, 0)
```

All tools call `_send_motors()` — no tool ever calls `robot.motor.drive()` directly.
Per-motor trim scalars (`left_motor_trim`, `right_motor_trim` in config) allow
fine-tuning if the rover still pulls to one side.

---

## Hot Memory

Every completed tool call is stored in a rolling JSON window (`.nova/hot_memory.json`).
The last 5 actions are injected into every executor LLM call:

```
══ RECENT ACTIONS ══════════════════════════════════════
  [14:32] navigate_to(kitchen) → Reached kitchen.
  [14:33] scan_for(cup) → Found 'cup' at ~0.8m (conf 0.91).
  [14:33] lock_and_track(red cup) → Reached object.
═══════════════════════════════════════════════════════
```

This gives the LLM instant awareness of what it just did without needing
vector search.

---

## Tools

| Tool | Description |
|---|---|
| `navigate_to` | A* path + obstacle replanning |
| `scan_for` | Pan sweep + YOLO detection |
| `scan_room` | 360° rotation + object catalog |
| `lock_and_track` | VLM locate → LK tracker → approach |
| `drive_raw` | Direct motor PWM for a duration |
| `drive_until_clear` | Drive until obstacle score drops |
| `spin_search` | Rotate to find clear path |
| `remember` | Save location or fact to memory |
| `read_file` | Read text/PDF/DOCX/XLSX |
| `write_file` | Write text to nova_data/ |
| `list_files` | Directory listing |
| `notify` | Send chat message to GUI |
| `ask_user` | Pause and ask human for input |

---

## Config

Key settings in `config.py`:

```python
# Visual obstacle thresholds (no ultrasonic)
edge_density_threshold: float = 0.08   # fraction of edge pixels = obstacle
obstacle_stop_score:    float = 0.65   # stop driving
obstacle_slow_score:    float = 0.35   # slow down + dodge

# Motor calibration
left_motor_trim:  float = 1.0   # adjust if rover pulls left
right_motor_trim: float = 1.0   # adjust if rover pulls right

# LLM
llm_model: str = "nemotron-3-super:cloud"
llm_api_key: str = ""   # loaded from NOVA_API_KEY env var

# Ultrasonic disabled
ultrasonic_enabled: bool = False
```

---

## Installation

```bash
# Unzip into your NOVA folder
unzip nova_v2.zip -d /path/to/Nomader/NOVA/

# Install dependencies
pip install -r NOVA/requirements.txt --break-system-packages

# Set your API key
cp NOVA/.env.example NOVA/.env
# Edit NOVA/.env → set NOVA_API_KEY
```

---

## Integration with existing GUI

`NovaAgent` exposes the same interface as v1:

```python
from NOVA.agent import NovaAgent

agent = NovaAgent(
    robot_interface=robot,
    camera_interface=camera,
    send_ws_callback=send_ws,
)
agent.start()
agent.submit_command("find the cup and bring it to the desk")
```

New: to answer `ask_user` tool calls from the GUI:
```python
agent.receive_user_answer("the blue cup on the shelf")
```
