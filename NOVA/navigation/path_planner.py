"""
NOVA Path Planner
-----------------
A* algorithm on the occupancy grid.
"""

import math
import heapq
from typing import List, Tuple

class PathPlanner:
    def __init__(self, config, spatial_map):
        self.config = config
        self.map = spatial_map
        
    def find_path(self, start_cm: Tuple[float, float], goal_cm: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        A* pathfinding.
        Returns a list of waypoints in world cm.
        """
        start = self.map.cm_to_cell(start_cm[0], start_cm[1])
        goal = self.map.cm_to_cell(goal_cm[0], goal_cm[1])
        
        # Check if goal is an obstacle
        if self.map.grid[goal[1], goal[0]] == 2:
            print("[NOVA Path Planner] Goal is inside an obstacle!")
            return []
            
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        
        # Heuristic function (Euclidean distance)
        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
            
        f_score = {start: heuristic(start, goal)}
        
        # 8-way movement
        neighbors = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        
        # Costs: 1 for straight, 1.414 for diagonal
        costs = [1, 1, 1, 1, 1.414, 1.414, 1.414, 1.414]
        
        max_size = self.config.navigation.grid_size_cells
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                
                # Convert back to cm
                return [self.map.cell_to_cm(cx, cy) for cx, cy in path]
                
            for idx, (dx, dy) in enumerate(neighbors):
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Bounds check
                if not (0 <= neighbor[0] < max_size and 0 <= neighbor[1] < max_size):
                    continue
                    
                # Obstacle check (value 2)
                if self.map.grid[neighbor[1], neighbor[0]] == 2:
                    continue
                    
                # Add extra cost for unknown cells (0) to encourage exploring free cells (1)
                cell_val = self.map.grid[neighbor[1], neighbor[0]]
                base_cost = costs[idx]
                if cell_val == 0:
                    base_cost *= 1.5
                    
                tentative_g_score = g_score[current] + base_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    
                    # Avoid duplicates in heap
                    if not any(item[1] == neighbor for item in open_set):
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        
        print("[NOVA Path Planner] No path found!")
        return []
