"""
NOVA Memory Store
-----------------
Two-tier memory inspired by Shifu's architecture:

  HOT MEMORY  — lightweight rolling window of recent actions.
                Injected directly into every LLM prompt (zero-latency).
                Survives restarts via JSON file.

  COLD MEMORY — persistent facts, object sightings, task history.
                Loaded on startup, saved on change.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional


# ── Hot Memory ─────────────────────────────────────────────────────────────────

class HotMemory:
    """
    Rolling window of (action, outcome) pairs.
    Provides instant context for the LLM without vector search.
    """

    def __init__(self, path: str, max_turns: int = 10):
        self._path      = Path(path)
        self._max_turns = max_turns
        self._entries: List[Dict] = []
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._entries = data.get("entries", [])[-self._max_turns:]
            except Exception:
                self._entries = []

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._path.write_text(json.dumps({"entries": self._entries}, indent=2))
        except Exception:
            pass  # best-effort

    def store(self, action: str, outcome: str):
        self._entries.append({
            "ts":      time.strftime("%H:%M"),
            "action":  action[:200],
            "outcome": outcome[:200],
        })
        if len(self._entries) > self._max_turns:
            self._entries = self._entries[-self._max_turns:]
        self._save()

    def as_prompt_block(self) -> str:
        if not self._entries:
            return ""
        lines = [
            f"  [{e['ts']}] {e['action'][:80]} → {e['outcome'][:80]}"
            for e in self._entries[-5:]  # last 5 is enough
        ]
        return (
            "══ RECENT ACTIONS (instant context) ══════════════════════════\n"
            + "\n".join(lines)
            + "\n══════════════════════════════════════════════════════════════"
        )

    def clear(self):
        count = len(self._entries)
        self._entries = []
        self._save()
        return count

    def count(self) -> int:
        return len(self._entries)


# ── Cold Memory ────────────────────────────────────────────────────────────────

class MemoryStore:
    def __init__(self, config):
        self.config       = config
        self._dir         = Path(config.memory.memory_dir)
        self._mem_file    = self._dir / "memory.json"
        self.facts:           Dict[str, Any] = {}
        self.object_sightings: Dict[str, Any] = {}
        self.task_history:     List[Dict]     = []

        self.hot = HotMemory(
            str(self._dir / "hot_memory.json"),
            max_turns=config.memory.hot_memory_max_turns,
        )

        self.load()

    def load(self):
        if not self._mem_file.exists():
            return
        try:
            data = json.loads(self._mem_file.read_text())
            self.facts            = data.get("facts", {})
            self.object_sightings = data.get("object_sightings", {})
            self.task_history     = data.get("task_history", [])
        except Exception as e:
            print(f"[NOVA Memory] Load error: {e}")

    def save(self):
        self._dir.mkdir(parents=True, exist_ok=True)
        try:
            self._mem_file.write_text(json.dumps({
                "facts":            self.facts,
                "object_sightings": self.object_sightings,
                "task_history":     self.task_history,
            }, indent=2))
        except Exception as e:
            print(f"[NOVA Memory] Save error: {e}")

    def remember_fact(self, key: str, value: str):
        self.facts[key] = {"value": value, "timestamp": time.time()}
        self.save()

    def get_fact(self, key: str) -> Optional[str]:
        f = self.facts.get(key)
        return f["value"] if f else None

    def log_sighting(self, name: str, x: float, y: float, ctx: str = ""):
        self.object_sightings[name] = {
            "x": x, "y": y, "context": ctx, "timestamp": time.time()
        }
        self.save()

    def add_to_history(self, task: str, result: str):
        self.task_history.append({
            "task": task, "result": result, "timestamp": time.time()
        })
        max_h = self.config.memory.max_task_history
        if len(self.task_history) > max_h:
            self.task_history = self.task_history[-max_h:]
        self.save()
        # Mirror into hot memory
        self.hot.store(task, result)

    def get_recent_history(self, n: int = 5) -> List[Dict]:
        return self.task_history[-n:]
