"""
NOVA Navigation Tool
--------------------
Navigates to a named location or coordinates.
Replans around obstacles automatically (up to max_replan_attempts).
"""

import asyncio
import math
from .base import Tool, ToolResult


class NavigateTo(Tool):
    name = "navigate_to"
    description = (
        "Navigate the rover to a named location or (x_cm, y_cm) coordinates. "
        "Automatically replans if blocked by an obstacle."
    )
    parameters = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Name of a saved location (e.g. 'kitchen'). Omit if using coordinates.",
            },
            "x_cm": {"type": "number", "description": "X coordinate in cm."},
            "y_cm": {"type": "number", "description": "Y coordinate in cm."},
        },
    }

    async def execute(
        self,
        location: str = None,
        x_cm: float = None,
        y_cm: float = None,
    ) -> ToolResult:
        if location:
            loc = self.context.spatial.get_location(location)
            if not loc:
                return ToolResult(False, f"Unknown location: '{location}'.")
            target_x, target_y = loc.x_cm, loc.y_cm
        elif x_cm is not None and y_cm is not None:
            target_x, target_y = x_cm, y_cm
        else:
            return ToolResult(False, "Provide either 'location' or 'x_cm'+'y_cm'.")

        max_replans = self.context.config.navigation.max_replan_attempts

        for attempt in range(max_replans + 1):
            pose  = self.context.odometry.get_pose()
            start = (pose[0], pose[1])
            goal  = (target_x, target_y)

            waypoints = self.context.planner_nav.find_path(start, goal)
            if not waypoints:
                return ToolResult(False, "No path found to destination.")

            self.context.driver.set_path(waypoints)
            label = location or f"({target_x:.0f}, {target_y:.0f})"
            self.context.send_status(f"Navigating to {label}… (attempt {attempt+1})")

            # Wait for driver
            while self.context.driver.is_navigating:
                await asyncio.sleep(0.1)
                if not self.context.is_running:
                    self.context.driver.stop()
                    return ToolResult(False, "Navigation interrupted.")

            pose  = self.context.odometry.get_pose()
            dist  = math.hypot(pose[0] - target_x, pose[1] - target_y)
            reach = self.context.config.navigation.waypoint_reach_cm * 2

            if dist <= reach:
                return ToolResult(True, f"Reached {label}.")

            # Mark estimated obstacle position
            obs_score = self.context.obstacle.get_proximity_score()
            if obs_score > self.context.config.navigation.obstacle_stop_score:
                obs_x = pose[0] + 25 * math.cos(pose[2])
                obs_y = pose[1] + 25 * math.sin(pose[2])
                self.context.spatial.mark_obstacle(obs_x, obs_y)
                self.context.send_status(f"Obstacle — replanning ({attempt+1}/{max_replans})…")
            else:
                return ToolResult(False, "Navigation stopped before destination (stuck?).")

        return ToolResult(False, f"Could not navigate to {label} after {max_replans} attempts.")
