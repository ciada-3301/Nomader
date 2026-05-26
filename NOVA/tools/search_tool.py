"""
NOVA SearchFor Tool (Robust Browser-Use Style)
----------------------------------------------
Single agentic tool that finds AND approaches an object in one continuous
VLM loop.

Key Improvements:
- Requires bounding box verification on every frame. If the object leaves
  the frame, the VLM cannot hallucinate completion.
- Fixed motor control logic (left/right are now properly mapped).
"""

import asyncio
import base64
import json
import cv2
from .base import Tool, ToolResult
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


_MAX_STEPS       = 30

# ── Timing ─────────────────────────────────────────────────────────────────────
ACTION_DURATION  = 2.0   # seconds each action runs
MID_CALL_OFFSET  = 1.0   # seconds into action when we fire the next VLM call

# ── Motor PWM ──────────────────────────────────────────────────────────────────
_ROTATE_FAST_PWM = 120   # was 80
_ROTATE_SLOW_PWM = 90    # was 55
_DRIVE_PWM       = 150   # was 90
_STEER_TURN_PWM  = 60    # was 40

# ── VLM system prompt ──────────────────────────────────────────────────────────
_SYSTEM = """You are NOVA's vision module controlling a 6WD rover.
You receive a live camera frame each step and choose one action.

Output strict JSON only — no prose, no markdown fences:
{
  "thought": "one sentence: what you see and why",
  "bbox": [x, y, w, h],
  "action":  "<action_name>",
  "duration_s": 2.0
}

Note about bbox: Provide the bounding box of the target object in the image. The coordinates should be scaled 0-1000 where 1000 is the full width/height.
If the object is NOT visible, set bbox to [].

Note about duration_s: You control how long the action runs in seconds (e.g. 0.3 for a tiny micro-correction, 1.0 for a normal rotation, 2.0+ for driving).

═══ ACTIONS ═══
  rotate_right_fast   — coarse rotation right (target is far right or lost)
  rotate_left_fast    — coarse rotation left  (target is far left or lost)
  rotate_right_slow   — fine correction right (target is slightly right)
  rotate_left_slow    — fine correction left  (target is slightly left)
  drive_forward       — target is ANYWHERE roughly in front of you (e.g. middle 60% of the screen). DO NOT try to perfectly center it! Just drive!
  steer_right         — driving forward but target is drifting to the far right edge
  steer_left          — driving forward but target is drifting to the far left edge
  arrived             — target bbox bottom edge is near the bottom of the frame (close to the rover)  arrived             — target bbox bottom edge is near the bottom of the frame (close to the rover)
  lost                — you cannot see the target object in the frame anymore

Rules:
1. ONLY output the JSON object.
2. You MUST include the bbox array. If the target is not in frame, action MUST be 'lost' and bbox MUST be [].
3. You cannot say 'arrived' if the bbox is empty.
4. DO NOT waste time trying to perfectly center the object. If it is somewhat in front of you, use drive_forward immediately!
"""


class SearchFor(Tool):
    name = "search_for"
    description = (
        "Visually search for an object AND drive toward it in one continuous "
        "agentic VLM loop. The AI narrates what it sees each step and issues "
        "micro-actions (rotate, drive, steer) until close to the target. "
        "Use for any task involving finding and approaching an object."
    )
    parameters = {
        "type": "object",
        "properties": {
            "object_description": {
                "type": "string",
                "description": "What to find and approach, e.g. 'cable on the floor', 'trashcan'.",
            },
        },
        "required": ["object_description"],
    }

    async def execute(self, object_description: str) -> ToolResult:
        if not self.context.camera:
            return ToolResult(False, "Camera not connected.")
        if not self.context.robot:
            return ToolResult(False, "Robot not connected.")

        self.context.send_status(f"Searching for '{object_description}'...")

        history: list[dict] = [{
            "role":    "user",
            "content": (
                f"TASK: Find '{object_description}' and drive close to it. "
                "Rotate to find it, then drive towards it."
            ),
        }]

        arrived      = False
        final_thought = ""

        # Prime the pipeline
        first_frame = await self._grab_frame()
        if first_frame is None:
            return ToolResult(False, "Camera returned no frame.")

        frame_h, frame_w = first_frame.shape[:2]
        next_action_obj  = await self._vlm_call(history, first_frame, 0, _MAX_STEPS, None)

        for step in range(_MAX_STEPS):
            if not self.context.is_running:
                self.context.robot.motor.halt()
                break

            action_obj    = next_action_obj
            thought       = action_obj.get("thought", "")
            action        = action_obj.get("action",  "lost")
            bbox          = action_obj.get("bbox", [])
            prev_b64      = action_obj.get("_b64", None)
            final_thought = thought

            self.context.send_status(f"{thought}")
            print(f"[SearchFor {step+1:02d}] {thought[:90]}  → {action}")

            history.append({"role": "assistant", "content": json.dumps(action_obj)})

            # Terminal actions
            if action == "arrived":
                # Verification: did they actually provide a bbox?
                if not bbox or len(bbox) != 4:
                    action = "lost" # Hallucinated completion without bbox
                    history.append({"role": "user", "content": "You said arrived but provided no bbox. You lost it. Spin to find it."})
                else:
                    self.context.robot.motor.halt()
                    arrived = True
                    pose = self.context.odometry.get_pose()
                    self.context.memory.log_sighting(
                        object_description, pose[0], pose[1],
                        f"Approached via agentic loop. {thought}"
                    )
                    break

            if action == "lost":
                history.append({"role": "user", "content": "Target lost. Spin to find it."})

            if bbox and len(bbox) == 4:
                # Bbox is 0-1000 scale
                bx, by, bw, bh = bbox
                self._emit_bbox([bx/1000*frame_w, by/1000*frame_h, bw/1000*frame_w, bh/1000*frame_h], object_description, frame_w, frame_h)

            duration      = action_obj.get("duration_s", 2.0)
            # Constrain duration to be safe
            duration      = max(0.2, min(duration, 5.0))
            
            self._start_motor(action)

            # At the halfway mark (or max 1s), grab a frame and call the VLM in background
            mid_offset = min(duration / 2.0, 1.0)
            mid_task = asyncio.create_task(
                self._mid_pipeline(history, step + 1, _MAX_STEPS, mid_offset, prev_b64)
            )

            # Let the action run for its full duration
            await asyncio.sleep(duration)
            self.context.robot.motor.halt()

            try:
                result = await asyncio.wait_for(mid_task, timeout=30.0)
                if result is not None:
                    next_action_obj = result
                    frame_h = result.get("_fh", frame_h)
                    frame_w = result.get("_fw", frame_w)
                else:
                    next_action_obj = {"thought": "VLM unavailable, lost.", "action": "lost", "bbox": []}
            except asyncio.TimeoutError:
                next_action_obj = {"thought": "VLM timed out, lost.", "action": "lost", "bbox": []}

            await asyncio.sleep(0.05)  # tiny gap for motor to fully stop

        self.context.robot.motor.halt()

        if arrived:
            return ToolResult(True,  f"Arrived at '{object_description}'. {final_thought}")
        return ToolResult(False, f"Search ended without arriving. {final_thought}")


    async def _mid_pipeline(self, history: list, step: int, max_steps: int, offset: float, prev_b64: str = None) -> dict | None:
        await asyncio.sleep(offset)
        frame = await self._grab_frame()
        if frame is None:
            return None
        result = await self._vlm_call(history, frame, step, max_steps, prev_b64)
        fh, fw = frame.shape[:2]
        result["_fh"] = fh
        result["_fw"] = fw
        return result

    async def _vlm_call(self, history: list, frame, step: int, max_steps: int, prev_b64: str = None) -> dict:
        frame_h, frame_w = frame.shape[:2]
        if frame_w > 640:
            scale  = 640 / frame_w
            frame  = cv2.resize(frame, (640, int(frame_h * scale)))
            frame_h, frame_w = frame.shape[:2]

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64    = base64.b64encode(buf).decode()

        content = []
        if prev_b64:
            content.append({"type": "text", "text": "Previous Frame (T-1):"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{prev_b64}"}})
        
        content.append({"type": "text", "text": f"Current Frame (T): step {step+1}/{max_steps} | size {frame_w}x{frame_h}"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        msgs = [
            SystemMessage(content=_SYSTEM),
            *[
                HumanMessage(content=m["content"]) if m["role"] == "user"
                else AIMessage(content=m["content"])
                for m in history
            ],
            HumanMessage(content=content),
        ]

        try:
            resp = await asyncio.to_thread(self.context._llm.invoke, msgs)
            text = resp.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
            result_obj = json.loads(text)
            result_obj["_b64"] = b64
            return result_obj
        except Exception as e:
            print(f"[SearchFor] VLM error step {step+1}: {e}")
            return {"thought": "VLM error.", "action": "lost", "bbox": []}

    async def _grab_frame(self):
        frame = await asyncio.to_thread(self.context.camera.get_frame)
        return frame

    def _start_motor(self, action: str):
        # We now use the standard drive(left_speed, right_speed, turn_bias)
        if action == "rotate_right_fast":
            # Right turn: left forward, right backward
            self.context.robot.motor.drive(_ROTATE_FAST_PWM, -_ROTATE_FAST_PWM, 0)
        elif action == "rotate_left_fast":
            # Left turn: left backward, right forward
            self.context.robot.motor.drive(-_ROTATE_FAST_PWM, _ROTATE_FAST_PWM, 0)
        elif action == "rotate_right_slow":
            self.context.robot.motor.drive(_ROTATE_SLOW_PWM, -_ROTATE_SLOW_PWM, 0)
        elif action == "rotate_left_slow":
            self.context.robot.motor.drive(-_ROTATE_SLOW_PWM, _ROTATE_SLOW_PWM, 0)
        elif action == "drive_forward":
            self.context.robot.motor.drive(_DRIVE_PWM, _DRIVE_PWM, 0)
        elif action == "steer_right":
            # Steer right: left goes faster than right
            self.context.robot.motor.drive(_DRIVE_PWM, _DRIVE_PWM - _STEER_TURN_PWM, 0)
        elif action == "steer_left":
            # Steer left: right goes faster than left
            self.context.robot.motor.drive(_DRIVE_PWM - _STEER_TURN_PWM, _DRIVE_PWM, 0)
        elif action == "lost":
            # If the VLM loses the target or times out, just HALT and re-evaluate. 
            # Blindly spinning often causes us to actually lose the target physically!
            self.context.robot.motor.halt()
        else:
            self.context.robot.motor.halt()

    def _emit_bbox(self, bbox, label: str, frame_w: int, frame_h: int):
        if not bbox or len(bbox) != 4 or not self.context.send_ws_callback:
            return
        x, y, w, h = [int(v) for v in bbox]
        self.context.send_ws_callback("nova_detection", {
            "label": label,
            "x": x / frame_w, "y": y / frame_h,
            "w": w / frame_w, "h": h / frame_h,
        })
