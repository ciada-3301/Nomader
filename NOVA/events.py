from enum import Enum
from typing import Callable, Dict, List, Any

class RobotEvent(Enum):
    TARGET_LOST = "TargetLost"
    TARGET_FOUND = "TargetFound"
    TASK_STEP_COMPLETE = "TaskStepComplete"
    TASK_COMPLETE = "TaskComplete"
    OBSTACLE_BLOCKED = "ObstacleBlocked"
    TARGET_AMBIGUOUS = "TargetAmbiguous"
    SPIN_SEARCH_FAILED = "SpinSearchFailed"
    VERIFICATION_FAILED = "VerificationFailed"
    REACQUISITION_NEEDED = "ReacquisitionNeeded"

class EventBus:
    """Simple pub/sub event bus for robot subsystem feedback."""
    def __init__(self):
        self._subscribers: Dict[RobotEvent, List[Callable[[Dict[str, Any]], None]]] = {}

    def subscribe(self, event: RobotEvent, callback: Callable[[Dict[str, Any]], None]) -> None:
        if event not in self._subscribers:
            self._subscribers[event] = []
        self._subscribers[event].append(callback)

    def emit(self, event: RobotEvent, payload: Dict[str, Any]) -> None:
        if event in self._subscribers:
            for callback in self._subscribers[event]:
                try:
                    callback(payload)
                except Exception as e:
                    print(f"[EventBus] Error in callback for {event.name}: {e}")

# Global instance if needed, or pass around
global_event_bus = EventBus()
