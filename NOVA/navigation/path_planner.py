"""
NOVA Path Planner
-----------------
A* on the occupancy grid. Unchanged from original — logic was sound.
"""

import math
import heapq
from typing import List, Tuple


class PathPlanner:
    def __init__(self, config, spatial_map):
        self.config = config
        self.map    = spatial_map

    def find_path(
        self,
        start_cm: Tuple[float, float],
        goal_cm:  Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        start = self.map.cm_to_cell(*start_cm)
        goal  = self.map.cm_to_cell(*goal_cm)

        if self.map.grid[goal[1], goal[0]] == 2:
            print("[NOVA Path] Goal inside obstacle — aborting.")
            return []

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score   = {start: 0}

        def h(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        f_score  = {start: h(start, goal)}
        moves    = [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        costs    = [1, 1, 1, 1, 1.414, 1.414, 1.414, 1.414]
        max_size = self.config.navigation.grid_size_cells

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return [self.map.cell_to_cm(cx, cy) for cx, cy in path]

            for idx, (dx, dy) in enumerate(moves):
                nb = (current[0] + dx, current[1] + dy)
                if not (0 <= nb[0] < max_size and 0 <= nb[1] < max_size):
                    continue
                if self.map.grid[nb[1], nb[0]] == 2:
                    continue
                cost = costs[idx]
                if self.map.grid[nb[1], nb[0]] == 0:
                    cost *= 1.5  # prefer explored cells
                tg = g_score[current] + cost
                if nb not in g_score or tg < g_score[nb]:
                    came_from[nb] = current
                    g_score[nb]   = tg
                    f_score[nb]   = tg + h(nb, goal)
                    if not any(item[1] == nb for item in open_set):
                        heapq.heappush(open_set, (f_score[nb], nb))

        print("[NOVA Path] No path found.")
        return []
