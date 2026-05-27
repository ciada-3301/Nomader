from dataclasses import dataclass
import time
from typing import List, Optional, Dict, Any

from ..tools.base import Tool, ToolResult

@dataclass
class SemanticObject:
    label: str
    last_seen_bbox: List[int]
    last_seen_frame_idx: int
    estimated_distance: Optional[float]
    vlm_bbox_cache: Optional[List[int]]
    hit_count: int
    timestamp: float

class SemanticMap:
    """
    Lightweight 2D object memory to reduce redundant perception and VLM calls.
    """
    def __init__(self, timeout_s: float = 60.0):
        self.objects: Dict[str, SemanticObject] = {}
        self.vlm_cache: Dict[str, Dict[str, Any]] = {}
        self.timeout_s = timeout_s
        self.frame_counter = 0

    def update(self, label: str, bbox: List[int], distance: Optional[float] = None) -> None:
        self.frame_counter += 1
        label = label.lower()
        if label in self.objects:
            obj = self.objects[label]
            obj.last_seen_bbox = bbox
            obj.last_seen_frame_idx = self.frame_counter
            obj.estimated_distance = distance
            obj.hit_count += 1
            obj.timestamp = time.time()
        else:
            self.objects[label] = SemanticObject(
                label=label,
                last_seen_bbox=bbox,
                last_seen_frame_idx=self.frame_counter,
                estimated_distance=distance,
                vlm_bbox_cache=None,
                hit_count=1,
                timestamp=time.time()
            )

    def query(self, label: str) -> Optional[SemanticObject]:
        label = label.lower()
        obj = self.objects.get(label)
        if obj:
            if time.time() - obj.timestamp <= self.timeout_s:
                return obj
            else:
                # Expired
                del self.objects[label]
        return None

    def invalidate(self, label: str) -> None:
        """Mark object as stale (e.g. after significant robot movement)."""
        label = label.lower()
        if label in self.objects:
            del self.objects[label]
            
    def cache_vlm_result(self, description: str, bbox: List[int]) -> None:
        self.vlm_cache[description.lower()] = {
            "bbox": bbox,
            "timestamp": time.time()
        }
        
    def get_vlm_cache(self, description: str) -> Optional[List[int]]:
        cache = self.vlm_cache.get(description.lower())
        if cache:
            if time.time() - cache["timestamp"] <= self.timeout_s:
                return cache["bbox"]
            else:
                del self.vlm_cache[description.lower()]
        return None

# Singleton
_global_semantic_map = None

def get_semantic_map() -> SemanticMap:
    global _global_semantic_map
    if _global_semantic_map is None:
        _global_semantic_map = SemanticMap()
    return _global_semantic_map

class SemanticMapQueryTool(Tool):
    name = "semantic_map_query"
    description = "Checks the semantic memory for the last known position of an object. Call this BEFORE using any detection or VLM tools to save tokens."
    parameters = {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": "The name of the object to look up."
            }
        },
        "required": ["label"]
    }

    async def execute(self, label: str) -> ToolResult:
        smap = get_semantic_map()
        obj = smap.query(label)
        
        if obj:
            data = {
                "bbox": obj.last_seen_bbox,
                "distance": obj.estimated_distance,
                "hit_count": obj.hit_count,
                "age_s": time.time() - obj.timestamp
            }
            return ToolResult(
                success=True,
                message=f"Found '{label}' in memory at {obj.last_seen_bbox} (age: {data['age_s']:.1f}s).",
                data=data
            )
        else:
            return ToolResult(
                success=False,
                message=f"Object '{label}' not in memory or has expired."
            )
