"""
NOVA SearchFor Tool
-------------------
Single agentic tool that finds AND approaches an object in one continuous
VLM loop — no handoff to lock_and_track needed.

Two phases, one loop:
  Phase 1 — SEARCH: VLM rotates the rover until the object is centred.
  Phase 2 — APPROACH: VLM drives forward, steering to stay on target,
             and decides when close enough (bbox bottom >= 75% of frame).

Pipelined execution:
  Each action runs for ACTION_DURATION seconds. At the MID_CALL_OFFSET mark
  (halfway through) the next VLM call is fired in the background. By the time
  the current action finishes, the next decision is already ready — so the
  rover moves continuously without stuttering between steps.

  Timeline per step:
    t=0.0s  action starts
    t=1.0s  VLM call fired in background (takes ~1s to return)
    t=2.0s  action ends, next action starts immediately
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
_ROTATE_FAST_PWM = 80
_ROTATE_SLOW_PWM = 55
_DRIVE_PWM       = 90
_STEER_TURN_PWM  = 40    # how much to reduce the inner wheel when steering

# ── VLM system prompt ──────────────────────────────────────────────────────────
_SYSTEM = """You are NOVA's vision module controlling a 6WD rover.
You receive a live camera frame each step and choose one action.

Output strict JSON only — no prose, no markdown fences:
{
  "thought": "one sentence: what you see and why",
  "action":  "<action_name>",
  "params":  {}
}

═══ PHASE 1 — SEARCH (rotate until object is centred) ═══
  rotate_right_fast   — coarse rotation right (~35 deg)
  rotate_left_fast    — coarse rotation left  (~35 deg)
  rotate_right_slow   — fine correction right (~12 deg)
  rotate_left_slow    — fine correction left  (~12 deg)
  object_centred      — object centre is between 25%-75% of frame width; begin approach

═══ PHASE 2 — APPROACH (drive until close) ═══
  drive_forward       — drive straight toward object
  steer_right         — drive forward curving right (object drifted right)
  steer_left          — drive forward curving left  (object drifted left)
  arrived             — bbox bottom >= 75% of frame height AND object is centred; stop

═══ TERMINAL ═══
  not_found           — completed full 360° search, object is absent

Rules:
1. Declare object_centred only when object centre_x is between 25% and 75% of frame width.
2. In Phase 2, steer when object drifts more than ~15% from frame centre.
3. Declare arrived when bbox bottom >= 75% frame height AND object is centred.
4. One sentence in "thought" — shown live to the user.
5. Output ONLY the JSON object — nothing else.
"""


class SearchFor(Tool):
    name = "search_for"
    description = (
        "Visually search for an object AND drive toward it in one continuous "
        "agentic VLM loop. The AI narrates what it sees each step and issues "
        "micro-actions (rotate, drive, steer) until close to the target. "
        "Replaces spin_search, scan_for, and lock_and_track entirely. "
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

    # ── Public entry point ─────────────────────────────────────────────────────

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
                "Start in Phase 1 (search). Switch to Phase 2 once centred."
            ),
        }]

        arrived      = False
        not_found    = False
        final_thought = ""

        # ── Prime the pipeline: get first VLM decision before moving ──────────
        first_frame = await self._grab_frame()
        if first_frame is None:
            return ToolResult(False, "Camera returned no frame.")

        frame_h, frame_w = first_frame.shape[:2]
        next_action_obj  = await self._vlm_call(history, first_frame, 0, _MAX_STEPS)

        # ── Main pipeline loop ─────────────────────────────────────────────────
        for step in range(_MAX_STEPS):
            if not self.context.is_running:
                self.context.robot.motor.halt()
                break

            action_obj    = next_action_obj
            thought       = action_obj.get("thought", "")
            action        = action_obj.get("action",  "rotate_right_fast")
            params        = action_obj.get("params",  {})
            final_thought = thought

            self.context.send_status(f"{thought}")
            print(f"[SearchFor {step+1:02d}] {thought[:90]}  → {action}")

            history.append({"role": "assistant", "content": json.dumps(action_obj)})

            # ── Terminal actions — no movement, just stop ──────────────────────
            if action == "not_found":
                self.context.robot.motor.halt()
                not_found = True
                break

            if action == "arrived":
                self.context.robot.motor.halt()
                arrived = True
                pose = self.context.odometry.get_pose()
                self.context.memory.log_sighting(
                    object_description, pose[0], pose[1],
                    f"Approached via agentic loop. {thought}"
                )
                break

            if action == "object_centred":
                # No physical movement — inject phase transition and immediately
                # grab a fresh frame for the next VLM call
                history.append({
                    "role":    "user",
                    "content": "Object centred. Now in Phase 2 — drive toward it.",
                })
                self._emit_bbox(params.get("bbox"), object_description, frame_w, frame_h)
                fresh = await self._grab_frame()
                if fresh is not None:
                    frame_h, frame_w = fresh.shape[:2]
                next_action_obj = await self._vlm_call(history, fresh or first_frame, step+1, _MAX_STEPS)
                continue

            # ── Movement actions: start motor, fire VLM at mid-point ───────────
            self._start_motor(action)

            # At the halfway mark, grab a frame and call the VLM in background
            mid_task = asyncio.create_task(
                self._mid_pipeline(history, step + 1, _MAX_STEPS)
            )

            # Let the action run for its full duration
            await asyncio.sleep(ACTION_DURATION)
            self.context.robot.motor.halt()

            # Collect the next decision (should already be ready)
            try:
                result = await asyncio.wait_for(mid_task, timeout=5.0)
                if result is not None:
                    next_action_obj = result
                    frame_h = result.get("_fh", frame_h)
                    frame_w = result.get("_fw", frame_w)
                else:
                    # VLM failed — safe default
                    next_action_obj = {"thought": "VLM unavailable, driving forward.", "action": "drive_forward", "params": {}}
            except asyncio.TimeoutError:
                next_action_obj = {"thought": "VLM timed out, driving forward.", "action": "drive_forward", "params": {}}

            await asyncio.sleep(0.05)  # tiny gap for motor to fully stop

        self.context.robot.motor.halt()

        if arrived:
            return ToolResult(True,  f"Arrived at '{object_description}'. {final_thought}")
        if not_found:
            return ToolResult(False, f"'{object_description}' not found. {final_thought}")
        return ToolResult(False, f"Step limit reached. {final_thought}")

    # ── Pipeline helper: wait MID_CALL_OFFSET then grab frame + call VLM ──────

    async def _mid_pipeline(self, history: list, step: int, max_steps: int) -> dict | None:
        await asyncio.sleep(MID_CALL_OFFSET)
        frame = await self._grab_frame()
        if frame is None:
            return None
        result = await self._vlm_call(history, frame, step, max_steps)
        # Stash frame dims so caller can update its local vars
        fh, fw = frame.shape[:2]
        result["_fh"] = fh
        result["_fw"] = fw
        return result

    # ── VLM call ───────────────────────────────────────────────────────────────

    async def _vlm_call(self, history: list, frame, step: int, max_steps: int) -> dict:
        frame_h, frame_w = frame.shape[:2]
        if frame_w > 640:
            scale  = 640 / frame_w
            frame  = cv2.resize(frame, (640, int(frame_h * scale)))
            frame_h, frame_w = frame.shape[:2]

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64    = base64.b64encode(buf).decode()

        msgs = [
            SystemMessage(content=_SYSTEM),
            *[
                HumanMessage(content=m["content"]) if m["role"] == "user"
                else AIMessage(content=m["content"])
                for m in history
            ],
            HumanMessage(content=[
                {"type": "text", "text": (
                    f"step {step+1}/{max_steps} | frame {frame_w}x{frame_h} | "
                    f"arrived threshold: bbox_bottom >= {int(frame_h * 0.75)}px"
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]),
        ]

        try:
            resp = await asyncio.to_thread(self.context._llm.invoke, msgs)
            text = resp.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
            return json.loads(text)
        except Exception as e:
            print(f"[SearchFor] VLM error step {step+1}: {e}")
            return {"thought": "VLM error — continuing.", "action": "drive_forward", "params": {}}

    # ── Grab frame ─────────────────────────────────────────────────────────────

    async def _grab_frame(self):
        frame = await asyncio.to_thread(self.context.camera.get_frame)
        return frame

    # ── Motor: start and leave running (caller halts after duration) ───────────

    def _start_motor(self, action: str):
        """Start motors for the given action. Caller is responsible for halting."""
        # drive(right, left, bias) — serialcom.py convention
        # Swap note: rotate_right means left wheel forward, right wheel backward
        if action == "rotate_right_fast":
            self.context.robot.motor.drive(-_ROTATE_FAST_PWM,  _ROTATE_FAST_PWM, 0)
        elif action == "rotate_left_fast":
            self.context.robot.motor.drive( _ROTATE_FAST_PWM, -_ROTATE_FAST_PWM, 0)
        elif action == "rotate_right_slow":
            self.context.robot.motor.drive(-_ROTATE_SLOW_PWM,  _ROTATE_SLOW_PWM, 0)
        elif action == "rotate_left_slow":
            self.context.robot.motor.drive( _ROTATE_SLOW_PWM, -_ROTATE_SLOW_PWM, 0)
        elif action == "drive_forward":
            self.context.robot.motor.drive(_DRIVE_PWM, _DRIVE_PWM, 0)
        elif action == "steer_right":
            self.context.robot.motor.drive(_DRIVE_PWM - _STEER_TURN_PWM, _DRIVE_PWM, 0)
        elif action == "steer_left":
            self.context.robot.motor.drive(_DRIVE_PWM, _DRIVE_PWM - _STEER_TURN_PWM, 0)
        else:
            # Unknown — go forward as safe default
            self.context.robot.motor.drive(_DRIVE_PWM, _DRIVE_PWM, 0)

    # ── Emit bbox overlay to webapp ────────────────────────────────────────────

    def _emit_bbox(self, bbox, label: str, frame_w: int, frame_h: int):
        if not bbox or len(bbox) != 4 or not self.context.send_ws_callback:
            return
        x, y, w, h = [int(v) for v in bbox]
        self.context.send_ws_callback("nova_detection", {
            "label": label,
            "x": x / frame_w, "y": y / frame_h,
            "w": w / frame_w, "h": h / frame_h,
        })
