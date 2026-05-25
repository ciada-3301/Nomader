"""
NOVA Spatial Map
----------------
2D occupancy grid + named location registry. Persisted to disk.
"""

import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class Location:
    name: str
    x_cm: float
    y_cm: float
    description: str = ""


class SpatialMap:
    def __init__(self, config):
        self.config   = config
        size          = config.navigation.grid_size_cells
        self.grid     = np.zeros((size, size), dtype=np.uint8)
        self.origin   = config.navigation.grid_origin_cell
        self.res      = config.navigation.grid_resolution_cm
        self.locations: Dict[str, Location] = {}
        self.map_file = os.path.join(config.memory.memory_dir, "spatial_map.json")
        self.load()

    def load(self):
        if not os.path.exists(self.map_file):
            return
        try:
            with open(self.map_file) as f:
                data = json.load(f)
            for name, d in data.get("locations", {}).items():
                self.locations[name] = Location(**d)
            print(f"[NOVA Map] Loaded {len(self.locations)} locations.")
        except Exception as e:
            print(f"[NOVA Map] Load error: {e}")

    def save(self):
        os.makedirs(os.path.dirname(self.map_file), exist_ok=True)
        try:
            with open(self.map_file, "w") as f:
                json.dump(
                    {"locations": {n: loc.__dict__ for n, loc in self.locations.items()}},
                    f, indent=2,
                )
        except Exception as e:
            print(f"[NOVA Map] Save error: {e}")

    def add_location(self, name: str, x: float, y: float, desc: str = ""):
        self.locations[name.lower()] = Location(name.lower(), x, y, desc)
        self.save()

    def get_location(self, name: str) -> Optional[Location]:
        name = name.lower()
        if name in self.locations:
            return self.locations[name]
        for k, v in self.locations.items():
            if name in k or k in name:
                return v
        return None

    def cm_to_cell(self, x_cm: float, y_cm: float) -> Tuple[int, int]:
        size = self.config.navigation.grid_size_cells
        cx   = max(0, min(size - 1, int(x_cm / self.res) + self.origin[0]))
        cy   = max(0, min(size - 1, int(y_cm / self.res) + self.origin[1]))
        return cx, cy

    def cell_to_cm(self, cx: int, cy: int) -> Tuple[float, float]:
        return (cx - self.origin[0]) * self.res, (cy - self.origin[1]) * self.res

    def mark_obstacle(self, x_cm: float, y_cm: float):
        cx, cy = self.cm_to_cell(x_cm, y_cm)
        self.grid[cy, cx] = 2

    def mark_free(self, x_cm: float, y_cm: float):
        cx, cy = self.cm_to_cell(x_cm, y_cm)
        self.grid[cy, cx] = 1

    def is_obstacle(self, x_cm: float, y_cm: float) -> bool:
        cx, cy = self.cm_to_cell(x_cm, y_cm)
        return self.grid[cy, cx] == 2
