"""
NOVA Control Tools
------------------
Low-level motor control and memory tools.

All motor commands go through driver._send_motors(logical_left, logical_right)
which applies the physical swap fix internally. Never call robot.motor.drive()
directly from tools.
"""

import asyncio
from .base import Tool, ToolResult


class DriveRaw(Tool):
    name = "drive_raw"
    description = (
        "Drive motors directly for a set duration. "
        "Use logical directions: positive = forward. "
        "Left/right swap is handled automatically."
    )
    parameters = {
        "type": "object",
        "properties": {
            "left_speed":  {"type": "integer", "description": "Left motor PWM (-255 to 255)."},
            "right_speed": {"type": "integer", "description": "Right motor PWM (-255 to 255)."},
            "duration":    {"type": "number",  "description": "Seconds to run motors."},
        },
        "required": ["left_speed", "right_speed", "duration"],
    }

    async def execute(self, left_speed: int, right_speed: int, duration: float) -> ToolResult:
        if not self.context.robot:
            return ToolResult(False, "Robot not connected.")

        self.context.driver._send_motors(left_speed, right_speed)

        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < duration:
            if not self.context.is_running:
                break
            await asyncio.sleep(0.05)

        self.context.robot.motor.halt()
        return ToolResult(True, f"Drove L={left_speed}, R={right_speed} for {duration}s.")


class DriveUntilClear(Tool):
    name = "drive_until_clear"
    description = (
        "Drive forward until the visual obstacle score drops below the slow threshold "
        "(path is clear) or a timeout elapses. Useful for escaping tight spots."
    )
    parameters = {
        "type": "object",
        "properties": {
            "direction": {
                "type":        "string",
                "description": "'forward' or 'backward'",
                "enum":        ["forward", "backward"],
            },
            "timeout_s": {"type": "number", "description": "Max seconds to drive (default 5)."},
        },
        "required": ["direction"],
    }

    async def execute(self, direction: str = "forward", timeout_s: float = 5.0) -> ToolResult:
        if not self.context.robot:
            return ToolResult(False, "Robot not connected.")

        nav   = self.context.config.navigation
        speed = nav.cruise_speed if direction == "forward" else -nav.cruise_speed
        self.context.driver._send_motors(speed, speed)

        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout_s:
            if not self.context.is_running:
                break
            score = self.context.obstacle.get_proximity_score()
            if score < nav.obstacle_slow_score:
                break
            await asyncio.sleep(0.1)

        self.context.robot.motor.halt()
        final_score = self.context.obstacle.get_proximity_score()
        if final_score < nav.obstacle_slow_score:
            return ToolResult(True, f"Path cleared (score {final_score:.2f}).")
        return ToolResult(False, f"Timed out — path still obstructed (score {final_score:.2f}).")


class SpinSearch(Tool):
    name = "spin_search"
    description = (
        "Rotate in place to search for a clear path or an object. "
        "Stops when path ahead is clear or full rotation completed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "direction": {
                "type":        "string",
                "description": "'left' or 'right' (default 'right')",
                "enum":        ["left", "right"],
            },
            "max_degrees": {
                "type":        "number",
                "description": "Maximum rotation in degrees (default 360).",
            },
        },
    }

    async def execute(self, direction: str = "right", max_degrees: float = 360.0) -> ToolResult:
        if not self.context.robot:
            return ToolResult(False, "Robot not connected.")

        nav   = self.context.config.navigation
        speed = nav.turn_speed

        # Spin: one motor forward, other backward
        if direction == "right":
            l_spd, r_spd = speed, -speed
        else:
            l_spd, r_spd = -speed, speed

        self.context.driver._send_motors(l_spd, r_spd)

        # Approximate rotation: 360° at turn_speed ≈ 2–3 seconds for most rovers
        # We stop early if the path clears
        rotation_s   = (max_degrees / 360.0) * 2.5
        stop_score   = nav.obstacle_slow_score
        cleared      = False

        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < rotation_s:
            if not self.context.is_running:
                break
            score = self.context.obstacle.get_proximity_score()
            if score < stop_score:
                cleared = True
                break
            await asyncio.sleep(0.1)

        self.context.robot.motor.halt()
        if cleared:
            return ToolResult(True, f"Found clear path after spinning {direction}.")
        return ToolResult(True, f"Completed {max_degrees}° spin {direction}. Path score: {self.context.obstacle.get_proximity_score():.2f}.")


class Remember(Tool):
    name = "remember"
    description = "Store a fact or save the current position under a name for future navigation."
    parameters = {
        "type": "object",
        "properties": {
            "key":   {"type": "string", "description": "Name of the location or fact."},
            "type":  {"type": "string", "description": "'location' or 'fact'", "enum": ["location", "fact"]},
            "value": {"type": "string", "description": "The fact value. Not needed for 'location'."},
        },
        "required": ["key", "type"],
    }

    async def execute(self, key: str, type: str, value: str = "") -> ToolResult:
        if type == "location":
            pose = self.context.odometry.get_pose()
            self.context.spatial.add_location(key, pose[0], pose[1], "Saved by NOVA")
            return ToolResult(True, f"Saved '{key}' at ({pose[0]:.0f}cm, {pose[1]:.0f}cm).")
        elif type == "fact":
            self.context.memory.remember_fact(key, value)
            return ToolResult(True, f"Remembered fact: {key} = {value}.")
        return ToolResult(False, "Type must be 'location' or 'fact'.")
