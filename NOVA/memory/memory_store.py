"""
NOVA Memory Store
-----------------
Persistent memory for facts, locations, and object sightings.
"""

import os
import json
import time

class MemoryStore:
    def __init__(self, config):
        self.config = config
        self.memory_file = os.path.join(self.config.memory.memory_dir, "memory.json")
        
        self.facts = {}
        self.object_sightings = {}
        self.task_history = []
        
        self.load()
        
    def load(self):
        if not os.path.exists(self.memory_file):
            return
            
        try:
            with open(self.memory_file, 'r') as f:
                data = json.load(f)
                self.facts = data.get("facts", {})
                self.object_sightings = data.get("object_sightings", {})
                self.task_history = data.get("task_history", [])
        except Exception as e:
            print(f"[NOVA Memory] Load error: {e}")
            
    def save(self):
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        try:
            with open(self.memory_file, 'w') as f:
                json.dump({
                    "facts": self.facts,
                    "object_sightings": self.object_sightings,
                    "task_history": self.task_history
                }, f, indent=2)
        except Exception as e:
            print(f"[NOVA Memory] Save error: {e}")
            
    def remember_fact(self, key: str, value: str):
        self.facts[key] = {
            "value": value,
            "timestamp": time.time()
        }
        self.save()
        
    def get_fact(self, key: str):
        fact = self.facts.get(key)
        return fact["value"] if fact else None
        
    def log_sighting(self, object_name: str, x: float, y: float, context: str = ""):
        self.object_sightings[object_name] = {
            "x": x,
            "y": y,
            "context": context,
            "timestamp": time.time()
        }
        self.save()
        
    def add_to_history(self, task: str, result: str):
        self.task_history.append({
            "task": task,
            "result": result,
            "timestamp": time.time()
        })
        if len(self.task_history) > self.config.memory.max_task_history:
            self.task_history.pop(0)
        self.save()
