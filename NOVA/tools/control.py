import asyncio
from .base import Tool, ToolResult

class DriveRaw(Tool):
    name = "drive_raw"
    description = "Drive the motors directly for a specific duration (in seconds)."
    parameters = {
        "type": "object",
        "properties": {
            "left_speed": {"type": "integer", "description": "Left motor PWM (-255 to 255)."},
            "right_speed": {"type": "integer", "description": "Right motor PWM (-255 to 255)."},
            "duration": {"type": "number", "description": "Duration to run motors in seconds."}
        },
        "required": ["left_speed", "right_speed", "duration"]
    }
    
    async def execute(self, left_speed: int, right_speed: int, duration: float) -> ToolResult:
        if not self.context.robot:
            return ToolResult(False, "Robot not connected.")
            
        self.context.robot.motor.drive(right_speed, left_speed, 0)
        
        # Sleep but allow interruption
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < duration:
            if not self.context.is_running:
                break
            await asyncio.sleep(0.1)
            
        self.context.robot.motor.halt()
        return ToolResult(True, f"Drove motors for {duration} seconds.")

class Remember(Tool):
    name = "remember"
    description = "Store a fact or associate a location name with current coordinates."
    parameters = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "The name of the location or fact (e.g. 'kitchen', 'user name')."},
            "type": {"type": "string", "description": "Type of memory: 'location' or 'fact'"},
            "value": {"type": "string", "description": "The fact string. Leave empty if type is 'location'."}
        },
        "required": ["key", "type"]
    }
    
    async def execute(self, key: str, type: str, value: str = "") -> ToolResult:
        if type == "location":
            pose = self.context.odometry.get_pose()
            self.context.spatial_map.add_location(key, pose[0], pose[1], "Saved by agent")
            return ToolResult(True, f"Remembered '{key}' at coordinates {pose[0]:.0f}, {pose[1]:.0f}.")
        elif type == "fact":
            self.context.memory.remember_fact(key, value)
            return ToolResult(True, f"Remembered fact: {key} = {value}.")
        else:
            return ToolResult(False, "Invalid memory type.")

class DriveUntilObstacle(Tool):
    name = "drive_until_obstacle"
    description = "Drive forward blindly until an obstacle is detected at a certain distance, or a timeout is reached. Useful for commands like 'go to the wall in front of you' or 'approach the object you see'."
    parameters = {
        "type": "object",
        "properties": {
            "stop_distance_cm": {"type": "number", "description": "Stop when an obstacle is this many cm away."},
            "timeout_s": {"type": "number", "description": "Max time to drive in seconds."}
        },
        "required": ["stop_distance_cm"]
    }
    
    async def execute(self, stop_distance_cm: float, timeout_s: float = 10.0) -> ToolResult:
        if not self.context.robot:
            return ToolResult(False, "Robot not connected.")
            
        speed = self.context.config.navigation.cruise_speed
        self.context.robot.motor.drive(speed, speed, 0)
        self.context.send_status(f"Approaching until {stop_distance_cm}cm...")
        
        start_time = asyncio.get_event_loop().time()
        stopped_due_to_obs = False
        
        while asyncio.get_event_loop().time() - start_time < timeout_s:
            if not self.context.is_running:
                break
                
            dist = self.context.obstacle_detector.get_forward_distance_cm()
            if dist > 0 and dist <= stop_distance_cm:
                stopped_due_to_obs = True
                break
                
            await asyncio.sleep(0.1)
            
        self.context.robot.motor.halt()
        
        if stopped_due_to_obs:
            return ToolResult(True, f"Stopped. Obstacle detected at {dist}cm.")
        else:
            return ToolResult(False, f"Timed out after {timeout_s}s. No obstacle reached.")
