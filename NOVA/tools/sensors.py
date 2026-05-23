import asyncio
import os
import time
from .base import Tool, ToolResult

class LogSensors(Tool):
    name = "log_sensors"
    description = "Read current sensors (ultrasonic, odometry, system stats) and log them to a CSV file."
    parameters = {
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "The name of the CSV file to append to."}
        },
        "required": ["filename"]
    }
    
    async def execute(self, filename: str) -> ToolResult:
        # Check for directory traversal attacks just in case
        filename = os.path.basename(filename)
        if not filename.endswith('.csv'):
            filename += '.csv'
            
        filepath = os.path.join(self.context.config.memory.memory_dir, filename)
        
        # Gather stats
        us_dist = getattr(self.context.robot, 'last_distance', -1) if self.context.robot else -1
        pose = self.context.odometry.get_pose()
        
        try:
            from helpers.stats import system
            temp = system.get_cpu_temp()
            usage = system.get_cpu_usage()
        except Exception:
            temp = "N/A"
            usage = "N/A"
            
        row = f"{time.time()},{pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f},{us_dist},{temp},{usage}\n"
        
        file_exists = os.path.exists(filepath)
        
        try:
            with open(filepath, 'a') as f:
                if not file_exists:
                    f.write("timestamp,x_cm,y_cm,heading_rad,ultrasonic_cm,cpu_temp,cpu_usage\n")
                f.write(row)
            return ToolResult(True, f"Successfully logged sensor data to {filename}.")
        except Exception as e:
            return ToolResult(False, f"Failed to write to file: {e}")
