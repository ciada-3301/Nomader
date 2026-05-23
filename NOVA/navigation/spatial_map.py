"""
NOVA Spatial Map
----------------
Maintains a 2D occupancy grid and named locations.
Saves to disk for persistence.
"""

import os
import json
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List

@dataclass
class Location:
    name: str
    x_cm: float
    y_cm: float
    description: str = ""

class SpatialMap:
    def __init__(self, config):
        self.config = config
        
        # The grid: 0=unknown, 1=free, 2=obstacle
        size = self.config.navigation.grid_size_cells
        self.grid = np.zeros((size, size), dtype=np.uint8)
        self.origin = self.config.navigation.grid_origin_cell
        self.res = self.config.navigation.grid_resolution_cm
        
        # Named locations
        self.locations: Dict[str, Location] = {}
        
        # Initialize memory file
        self.map_file = os.path.join(self.config.memory.memory_dir, "spatial_map.json")
        self.load()
        
    def load(self):
        """Load map from disk."""
        if not os.path.exists(self.map_file):
            return
            
        try:
            with open(self.map_file, 'r') as f:
                data = json.load(f)
                
            # Load locations
            if "locations" in data:
                for name, loc_data in data["locations"].items():
                    self.locations[name] = Location(**loc_data)
                    
            print(f"[NOVA Map] Loaded {len(self.locations)} locations from memory.")
            
        except Exception as e:
            print(f"[NOVA Map] Error loading map: {e}")
            
    def save(self):
        """Save map to disk."""
        os.makedirs(os.path.dirname(self.map_file), exist_ok=True)
        
        try:
            data = {
                "locations": {name: loc.__dict__ for name, loc in self.locations.items()}
            }
            with open(self.map_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[NOVA Map] Error saving map: {e}")
            
    def add_location(self, name: str, x: float, y: float, description: str = ""):
        """Add or update a named location."""
        self.locations[name.lower()] = Location(name.lower(), x, y, description)
        self.save()
        
    def get_location(self, name: str) -> Location:
        """Get location by name (fuzzy match)."""
        name = name.lower()
        if name in self.locations:
            return self.locations[name]
            
        # Fuzzy match
        for loc_name, loc in self.locations.items():
            if name in loc_name or loc_name in name:
                return loc
                
        return None
        
    def cm_to_cell(self, x_cm: float, y_cm: float) -> Tuple[int, int]:
        """Convert world cm to grid cell indices."""
        cx = int(x_cm / self.res) + self.origin[0]
        cy = int(y_cm / self.res) + self.origin[1]
        
        size = self.config.navigation.grid_size_cells
        cx = max(0, min(size-1, cx))
        cy = max(0, min(size-1, cy))
        
        return cx, cy
        
    def cell_to_cm(self, cx: int, cy: int) -> Tuple[float, float]:
        """Convert grid cell indices to world cm."""
        x_cm = (cx - self.origin[0]) * self.res
        y_cm = (cy - self.origin[1]) * self.res
        return x_cm, y_cm
        
    def mark_obstacle(self, x_cm: float, y_cm: float):
        """Mark a cell as containing an obstacle."""
        cx, cy = self.cm_to_cell(x_cm, y_cm)
        self.grid[cy, cx] = 2
        
    def mark_free(self, x_cm: float, y_cm: float):
        """Mark a cell as free space."""
        cx, cy = self.cm_to_cell(x_cm, y_cm)
        self.grid[cy, cx] = 1
        
    def is_obstacle(self, x_cm: float, y_cm: float) -> bool:
        """Check if a coordinate contains an obstacle."""
        cx, cy = self.cm_to_cell(x_cm, y_cm)
        return self.grid[cy, cx] == 2
