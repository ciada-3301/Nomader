import asyncio
from .base import Tool, ToolResult

class NavigateTo(Tool):
    name = "navigate_to"
    description = "Navigate the rover to a named location or specific coordinates."
    parameters = {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "Name of the location (e.g. 'kitchen', 'desk'). Omit if providing coordinates."},
            "x_cm": {"type": "number", "description": "X coordinate in cm. Use only if location name is not provided."},
            "y_cm": {"type": "number", "description": "Y coordinate in cm. Use only if location name is not provided."}
        }
    }
    
    async def execute(self, location: str = None, x_cm: float = None, y_cm: float = None) -> ToolResult:
        if location:
            loc = self.context.spatial_map.get_location(location)
            if not loc:
                return ToolResult(False, f"I don't know where '{location}' is.")
            target_x, target_y = loc.x_cm, loc.y_cm
        elif x_cm is not None and y_cm is not None:
            target_x, target_y = x_cm, y_cm
        else:
            return ToolResult(False, "Must provide either location name or coordinates.")
            
        curr_pose = self.context.odometry.get_pose()
        start = (curr_pose[0], curr_pose[1])
        goal = (target_x, target_y)
        
        import math
        replans = 0
        max_replans = 3
        
        while replans <= max_replans:
            waypoints = self.context.path_planner.find_path(start, goal)
            if not waypoints:
                return ToolResult(False, "Could not find a clear path to the destination.")
                
            self.context.driver.set_path(waypoints)
            self.context.send_status(f"Navigating to {location or 'coordinates'}...")
            
            # Wait for driver to finish
            while self.context.driver.is_navigating:
                await asyncio.sleep(0.1)
                if not self.context.is_running:
                    self.context.driver.stop()
                    return ToolResult(False, "Navigation interrupted by user.")
                    
            # Determine why it stopped
            curr_pose = self.context.odometry.get_pose()
            start = (curr_pose[0], curr_pose[1])
            dist = ((curr_pose[0] - target_x)**2 + (curr_pose[1] - target_y)**2)**0.5
            
            if dist <= self.context.config.navigation.waypoint_reach_cm * 2:
                return ToolResult(True, f"Successfully navigated to {location or 'destination'}.")
                
            # Check if an obstacle caused the stop
            forward_dist = self.context.obstacle_detector.get_forward_distance_cm()
            stop_thresh = self.context.config.navigation.obstacle_stop_cm
            
            if forward_dist <= stop_thresh + 5.0:
                # We hit an obstacle. Mark it on the occupancy grid.
                obs_x = curr_pose[0] + forward_dist * math.cos(curr_pose[2])
                obs_y = curr_pose[1] + forward_dist * math.sin(curr_pose[2])
                self.context.spatial_map.mark_obstacle(obs_x, obs_y)
                
                self.context.send_status(f"Obstacle detected at {forward_dist}cm. Replanning...")
                replans += 1
            else:
                return ToolResult(False, "Navigation stopped before reaching destination (stuck or blocked).")
                
        return ToolResult(False, f"Failed to navigate around obstacles after {max_replans} attempts.")
