"""
NOVA Agent  (v2 — Reactive LangGraph Architecture)
===================================================
Replaces the static plan-then-execute loop with a Shifu-inspired reactive
graph that adapts to failures, replans dynamically, and never follows a
hardcoded sequence.

Architecture overview
─────────────────────
          ┌─────────┐
  cmd ───►│  plan   │
          └────┬────┘
               │ needs clarification?
          ┌────▼────────┐       no
          │   clarify   │──────────────────────┐
          └─────────────┘                       │
                                          ┌─────▼──────┐
                                          │  execute   │◄──────┐
                                          └─────┬──────┘       │
                                  tool calls?   │               │
                                       ┌────────▼────────┐     │
                                       │   tools_node    │     │
                                       └────────┬────────┘     │
                                                │               │
                                          ┌─────▼──────┐       │
                                          │   review   │ retry │
                                          └─────┬──────┘───────┘
                                             pass│
                                              END

Key differences from v1
────────────────────────
• NO hardcoded plan steps. The executor decides tools and order.
• Review node detects failures and feeds them back — executor adapts.
• Hot memory from previous actions is injected into every turn.
• Visual obstacle avoidance runs in a background hardware loop.
• Motor swap fix applied at driver level, not scattered in tools.
• All tools return ToolResult; executor retries on failure.
"""

import asyncio
import json
import re
import time
import threading
import queue
import uuid
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_openai import ChatOpenAI

from .config import NovaConfig
from .vision.obstacle import VisualObstacleDetector
from .navigation.spatial_map import SpatialMap
from .navigation.odometry import Odometry
from .navigation.path_planner import PathPlanner
from .navigation.driver import NavigationDriver
from .memory.memory_store import MemoryStore

# ── Tool imports ───────────────────────────────────────────────────────────────
from .tools.base import Tool, ToolResult
from .tools.navigation import NavigateTo
from .tools.vision_tools import ScanFor
from .tools.control import DriveRaw, DriveUntilClear, Remember
from .tools.search_tool import SearchFor
from .tools.filesystem import ReadFile, WriteFile, ListFiles
from .tools.comms import Notify, AskUser

# ── New Perception & Skills Tools ──────────────────────────────────────────────
from .perception.detector import GroundingDetectorTool
from .perception.tracker import TrackerLockTool, TrackerStatusTool, TrackerReleaseTool
from .skills.spinsearch import SpinSearch as NewSpinSearch
from .skills.vlm_grounder import VLMGroundTool, VLMVerifyTool
from .memory.semantic_map import SemanticMapQueryTool
from .events import global_event_bus, RobotEvent


# ══════════════════════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════════════════════

class NovaState(TypedDict):
    messages:     list[BaseMessage]   # full message history for this mission
    mission:      str
    plan:         str                 # planner's execution sketch (guidance only)
    route:        str                 # "SIMPLE" | "COMPLEX"
    clarify_q:    str
    clarify_a:    str
    iterations:   int
    verdict:      str                 # "PASS" | "RETRY" | ""
    retry_count:  int
    done:         bool


# ══════════════════════════════════════════════════════════════════════════════
#  NOVA AGENT
# ══════════════════════════════════════════════════════════════════════════════

class NovaAgent:
    """
    Main agent class. Owns all subsystems and the LangGraph execution loop.

    Usage:
        agent = NovaAgent(robot_interface, camera_interface, ws_callback)
        agent.start()
        agent.submit_command("go find the cup and come back")
    """

    DONE_SENTINEL = "<<<DONE>>>"

    def __init__(
        self,
        robot_interface=None,
        camera_interface=None,
        send_ws_callback=None,
    ):
        self.config           = NovaConfig()
        self.robot            = robot_interface
        self.camera           = camera_interface
        self.send_ws_callback = send_ws_callback

        # Agent state
        self.is_running    = False
        self.current_task: Optional[asyncio.Task] = None
        self.status        = "idle"
        self.command_queue = queue.Queue()
        self._notify_sent  = False   # tracks whether notify tool fired during a mission

        # ── Subsystems ─────────────────────────────────────────────────────────
        self.obstacle  = VisualObstacleDetector(self.config)
        self.odometry  = Odometry(self.config)
        self.spatial   = SpatialMap(self.config)
        self.planner_nav = PathPlanner(self.config, self.spatial)
        self.driver    = NavigationDriver(
            self.config, self.robot, self.odometry, self.obstacle
        )
        self.memory    = MemoryStore(self.config)

        # ── Event Subscription ────────────────────────────────────────────────
        global_event_bus.subscribe(RobotEvent.TARGET_LOST, self._on_target_lost)
        global_event_bus.subscribe(RobotEvent.REACQUISITION_NEEDED, self._on_reacquisition_needed)

        # ── LLM ───────────────────────────────────────────────────────────────
        cfg = self.config.planner
        api_key = cfg.llm_api_key if cfg.llm_api_key not in ("", "ollama") else "no-key"
        self._llm = ChatOpenAI(
            model=cfg.llm_model,
            base_url=cfg.llm_base_url,
            api_key=api_key,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
            timeout=cfg.llm_timeout_s,
        )

        # ── Tools ─────────────────────────────────────────────────────────────
        tools_list = [
            NavigateTo(self),
            SearchFor(self),
            ScanFor(self),
            DriveRaw(self),
            DriveUntilClear(self),
            Remember(self),
            ReadFile(self),
            WriteFile(self),
            ListFiles(self),
            Notify(self),
            AskUser(self),
            GroundingDetectorTool(self),
            TrackerLockTool(self),
            TrackerStatusTool(self),
            TrackerReleaseTool(self),
            SemanticMapQueryTool(self),
            NewSpinSearch(self),
            VLMGroundTool(self),
            VLMVerifyTool(self),
        ]
        self.tools = {t.name: t for t in tools_list}
        self._llm_with_tools = self._llm.bind(
            tools=[self._tool_schema(t) for t in tools_list],
            tool_choice="auto",
        )

        self._loop_thread: Optional[threading.Thread] = None
        self._async_loop:  Optional[asyncio.AbstractEventLoop] = None

        # AskUser inter-thread bridge
        self._ask_q: Optional[asyncio.Queue] = None

    # ── Schema builder ─────────────────────────────────────────────────────────

    @staticmethod
    def _tool_schema(tool: Tool) -> dict:
        return {
            "type": "function",
            "function": {
                "name":        tool.name,
                "description": tool.description,
                "parameters":  tool.parameters,
            },
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  Lifecycle
    # ══════════════════════════════════════════════════════════════════════════

    def start(self):
        if self.is_running:
            return
        self.is_running   = True
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()
        print("[NOVA] Agent started.")

    def stop(self):
        self.is_running = False
        self.driver.stop()
        if self._async_loop:
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        print("[NOVA] Agent stopped.")

    def submit_command(self, command: str):
        """Submit a natural-language command (cancels any running task)."""
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            self.driver.stop()
        self.command_queue.put(command)

    # ── Comms helpers (called from tools) ─────────────────────────────────────

    def send_status(self, status: str):
        self.status = status
        print(f"[NOVA] {status}")
        if self.send_ws_callback:
            self.send_ws_callback("nova_status", {"status": status})

    def send_chat(self, message: str, from_notify: bool = False):
        if from_notify:
            self._notify_sent = True
        print(f"[NOVA Chat] {message}")
        if self.send_ws_callback:
            self.send_ws_callback("nova_chat", {"sender": "NOVA", "message": message})

    def _on_target_lost(self, payload: dict):
        self.send_status("Target lost...")
        self.submit_command("Target was lost. Run re-acquisition protocol.")

    def _on_reacquisition_needed(self, payload: dict):
        self.send_status("Re-acquisition needed...")
        self.submit_command("Re-acquire the target.")

    # ── AskUser bridge ─────────────────────────────────────────────────────────

    def ask_user_sync(self, question: str) -> str:
        """Called from AskUser tool (runs in executor thread)."""
        self.send_ws_callback("nova_ask", {"question": question})
        # Block tool thread until GUI answers
        if self._ask_q and self._async_loop:
            fut = asyncio.run_coroutine_threadsafe(
                self._ask_q.get(), self._async_loop
            )
            try:
                return fut.result(timeout=120)
            except Exception:
                return "(no answer — proceeding with best judgment)"
        return "(ask_user unavailable)"

    def receive_user_answer(self, answer: str):
        """Called by GUI when user responds to an ask_user prompt."""
        if self._ask_q and self._async_loop:
            self._async_loop.call_soon_threadsafe(
                self._ask_q.put_nowait, answer
            )

    # ══════════════════════════════════════════════════════════════════════════
    #  Background loops
    # ══════════════════════════════════════════════════════════════════════════

    def _run_loop(self):
        self._async_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._async_loop)
        self._ask_q = asyncio.Queue()

        self._async_loop.create_task(self._hardware_loop())
        self._async_loop.create_task(self._command_processor())

        try:
            self._async_loop.run_forever()
        except Exception as e:
            print(f"[NOVA] Fatal: {e}")
        finally:
            self._async_loop.close()

    async def _hardware_loop(self):
        """10 Hz: update odometry, run driver with latest camera frame."""
        hz  = self.config.navigation.control_hz
        while self.is_running:
            frame = None
            if self.camera:
                frame = self.camera.get_frame()

            if self.robot:
                self.odometry.update(self.robot.left_speed, self.robot.right_speed)

            if self.driver.is_navigating:
                status = self.driver.update(frame)
                if status == "blocked":
                    self.send_status("Obstacle — rerouting…")
                elif status == "stuck":
                    self.send_status("Stuck — please help!")

            await asyncio.sleep(1.0 / hz)

    async def _command_processor(self):
        """Poll the command queue and spawn a mission task."""
        while self.is_running:
            try:
                cmd = self.command_queue.get_nowait()
                self.current_task = asyncio.create_task(self._run_mission(cmd))
            except queue.Empty:
                pass
            await asyncio.sleep(0.1)

    # ══════════════════════════════════════════════════════════════════════════
    #  Mission execution  (Shifu-style reactive loop)
    # ══════════════════════════════════════════════════════════════════════════

    async def _run_mission(self, mission: str):
        """
        Adaptive mission runner.

        Flow:
          1. Planner decides route + sketch plan (1 LLM call)
          2. Executor uses tools freely — no forced sequence
          3. Review node checks for failures → retry or done
          4. Hot memory updated on completion
        """
        try:
            self.send_status(f"Planning: '{mission}'")

            cfg = self.config.planner
            state: NovaState = {
                "messages":    [],
                "mission":     mission,
                "plan":        "",
                "route":       "",
                "clarify_q":   "",
                "clarify_a":   "",
                "iterations":  0,
                "verdict":     "",
                "retry_count": 0,
                "done":        False,
            }

            # ── PHASE 1: plan ─────────────────────────────────────────────────
            state = await self._plan_node(state)
            if state["clarify_q"]:
                state = await self._clarify_node(state)

            self.send_chat(
                f"Understood. Route: {state['route']}. "
                + (f"Plan: {state['plan'][:120]}…" if state["plan"] else "Acting directly.")
            )

            # ── PHASE 2: execute → tools → review loop ─────────────────────
            t_start = time.time()

            for iteration in range(cfg.max_iterations):
                if not self.is_running:
                    break
                if time.time() - t_start > cfg.mission_timeout_s:
                    self.send_chat("Mission timed out.")
                    break

                state["iterations"] = iteration + 1
                state = await self._execute_node(state)

                last = state["messages"][-1] if state["messages"] else None
                done_flag = self.DONE_SENTINEL in (getattr(last, "content", "") or "")

                # Run tools if called
                if last and getattr(last, "tool_calls", None):
                    state = await self._tools_node(state)

                # Check done signal
                if done_flag and not getattr(last, "tool_calls", None):
                    break

                # No tool calls + no done = review
                if not getattr(last, "tool_calls", None):
                    if state["route"] == "COMPLEX":
                        state = await self._review_node(state)
                        if state["verdict"] == "PASS" or state["retry_count"] >= cfg.max_retries:
                            break
                        # Retry: clear messages so executor starts fresh context
                        state["messages"] = []
                    else:
                        break

            # ── PHASE 3: final output ─────────────────────────────────────────
            final = self._extract_final(state)
            clean = final.replace(self.DONE_SENTINEL, "").strip()
            if clean and not self._notify_sent:
                self.send_chat(clean)
            self._notify_sent = False  # reset for next mission

            self.send_status("idle")
            self.memory.add_to_history(mission, "success" if state["verdict"] != "RETRY" else "partial")

        except asyncio.CancelledError:
            self.send_status("idle")
            self.memory.add_to_history(mission, "cancelled")
        except Exception as e:
            self.send_chat(f"Mission failed: {e}")
            self.send_status("idle")
            self.memory.add_to_history(mission, f"error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    #  Graph nodes
    # ══════════════════════════════════════════════════════════════════════════

    async def _plan_node(self, state: NovaState) -> NovaState:
        prompt = self._build_planner_prompt(state["mission"])
        try:
            resp = await asyncio.to_thread(
                self._llm.invoke,
                [SystemMessage(content=self._planner_system()), HumanMessage(content=prompt)]
            )
            parsed = self._parse_plan_response(resp.content)
        except Exception as e:
            print(f"[NOVA Plan] LLM error: {e}")
            parsed = {"route": "SIMPLE", "clarify_q": "", "plan": ""}
        return {**state, **parsed}

    async def _clarify_node(self, state: NovaState) -> NovaState:
        """Ask the GUI/user for clarification, await answer."""
        self.send_ws_callback("nova_ask", {"question": state["clarify_q"]}) if self.send_ws_callback else None
        answer = "(no answer — proceed with best judgment)"
        if self._ask_q:
            try:
                answer = await asyncio.wait_for(self._ask_q.get(), timeout=60)
            except asyncio.TimeoutError:
                pass
        return {**state, "clarify_a": answer}

    async def _execute_node(self, state: NovaState) -> NovaState:
        """One LLM turn — may emit tool calls or a final answer."""
        msgs = self._build_executor_messages(state)
        try:
            resp = await asyncio.to_thread(self._llm_with_tools.invoke, msgs)
        except Exception as e:
            err_msg = AIMessage(content=f"[NOVA] LLM error: {e}. Retrying.")
            return {**state, "messages": state["messages"] + [err_msg]}
        return {**state, "messages": msgs + [resp]}

    async def _tools_node(self, state: NovaState) -> NovaState:
        """Execute all tool calls in the last AI message."""
        last = state["messages"][-1]
        tool_msgs = []

        for tc in last.tool_calls:
            name    = tc.get("name", "")
            args    = {k: v for k, v in tc.get("args", {}).items()}
            call_id = tc.get("id") or str(uuid.uuid4())

            tool = self.tools.get(name)
            if tool is None:
                result = f'{{"status": "error", "message": "Unknown tool: {name}"}}'
            else:
                self.send_status(f"Running: {name}")
                try:
                    tr: ToolResult = await tool.execute(**args)
                    result = json.dumps({
                        "success": tr.success,
                        "message": tr.message,
                        "data":    str(tr.data) if tr.data is not None else None,
                    })
                    # Store in hot memory
                    self.memory.hot.store(f"{name}({args})", tr.message[:100])
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    result = f'{{"status": "error", "message": "{e}"}}'

            tool_msgs.append(ToolMessage(
                content=result, tool_call_id=call_id, name=name
            ))

        return {**state, "messages": state["messages"] + tool_msgs}

    async def _review_node(self, state: NovaState) -> NovaState:
        """Fast-path pass if no errors, otherwise ask LLM for verdict."""
        last_tool = next(
            (m for m in reversed(state["messages"]) if isinstance(m, ToolMessage)), None
        )
        err_kw = ("error", "exception", "failed", "traceback", "timeout",
                  "not found", "permission denied", "refused")

        if last_tool and not any(k in str(last_tool.content).lower() for k in err_kw):
            return {**state, "verdict": "PASS"}

        # Ask reviewer LLM
        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None
        )
        agent_out = (last_ai.content or "") if last_ai else ""
        if last_tool:
            agent_out += "\n[Tool]: " + str(last_tool.content)[:400]

        try:
            resp = await asyncio.to_thread(
                self._llm.invoke,
                [
                    SystemMessage(content=self._reviewer_system()),
                    HumanMessage(content=f"Mission: {state['mission']}\n\nOutput:\n{agent_out}"),
                ]
            )
            verdict = "PASS" if "PASS" in resp.content.upper() else "RETRY"
        except Exception:
            verdict = "PASS"

        new_retry = state["retry_count"] + (1 if verdict == "RETRY" else 0)
        return {**state, "verdict": verdict, "retry_count": new_retry}

    # ══════════════════════════════════════════════════════════════════════════
    #  Prompt builders
    # ══════════════════════════════════════════════════════════════════════════

    def _tools_description(self) -> str:
        lines = []
        for name, tool in self.tools.items():
            lines.append(f"  • **{name}**: {tool.description}")
        return "\n".join(lines)

    def _planner_system(self) -> str:
        return f"""You are NOVA's strategic planner. NOVA is an autonomous 6-wheel-drive robot.

Available tools:
{self._tools_description()}

Reply with ALL THREE sections, in order, using the exact headers shown.

━━ ROUTE ━━
SIMPLE  — one or two tool calls, straight-forward task
COMPLEX — multi-step, requires vision, navigation, or sequencing

━━ CLARIFY ━━
NO_CLARIFICATION
or:
CLARIFY: <one focused question if essential info is missing>

━━ PLAN ━━
A numbered sketch of what you'll do. Not a rigid script — the executor
will adapt. Write PLAN: N/A for SIMPLE tasks.
"""

    def _build_planner_prompt(self, mission: str) -> str:
        return f"Mission: {mission}\n\nPlayground (file ops): ./nova_data/"

    def _parse_plan_response(self, content: str) -> dict:
        route = "SIMPLE"
        m = re.search(r'━━\s*ROUTE\s*━━\s*\n(.+)', content)
        if m:
            w = m.group(1).strip().upper()
            route = "COMPLEX" if "COMPLEX" in w else "SIMPLE"
        elif "COMPLEX" in content.upper():
            route = "COMPLEX"

        clarify_q = ""
        m2 = re.search(r'CLARIFY:\s*(.+)', content, re.IGNORECASE)
        if m2 and "NO_CLARIFICATION" not in content.upper():
            clarify_q = m2.group(1).strip()

        plan = ""
        m3 = re.search(r'━━\s*PLAN\s*━━(.+?)(?=$)', content, re.DOTALL)
        if m3:
            raw = m3.group(1).strip()
            plan = "" if "N/A" in raw.upper() else raw

        return {"route": route, "clarify_q": clarify_q, "plan": plan}

    def _executor_system(self) -> str:
        hot_block = self.memory.hot.as_prompt_block()
        return f"""You are NOVA, an autonomous 6WD rover with a camera.
You act via tools — never describe actions without calling a tool.

{hot_block}

AVAILABLE TOOLS:
{self._tools_description()}

RULES:
1. Use tools to do things. Text responses = thinking aloud only.
2. When fully done, include {self.DONE_SENTINEL} in your final text response (no more tool calls after).
3. If a tool fails, adapt — try a different approach, do not give up immediately.
4. Navigate around obstacles if blocked — call search_for to reorient or drive_raw to back away.
5. Use notify to keep the user updated on progress.
6. ask_user only when you genuinely cannot proceed without human input.

MOTOR NOTE:
The physical motors are wired correctly — just use the logical directions
(forward, left, right). The driver handles internal calibration.

## Operational Rules

### Tool Selection Priority
1. Check semantic_map_query FIRST before any detection or VLM call.
2. Use grounding_detector for concrete object classes (chair, table, cup, person, door).
3. Use vlm_ground ONLY for abstract spatial regions (foot of X, gap under Y, left side of Z) or when grounding_detector fails.
4. Use spin_search when the target is not in current camera view.
5. Use vlm_verify after every navigation task completion before declaring success.

### Token Conservation
- Never call the cloud VLM for a target already in semantic memory (within 60 seconds).
- Always set explicit max_tokens on VLM calls. Spatial grounding: 80 tokens. Verification: 60 tokens. Planning: 300 tokens.
- Never ask the VLM open-ended questions during execution. All VLM calls during task execution must request structured JSON responses.

### Movement Rules
- Always switch to slow_crawl before locking tracker or running perception during approach.
- Never run tracker_update during fast movement.
- After SpinSearch success, tracker_lock is called automatically — do not call it again manually.

### Re-acquisition Protocol
- If tracker_status returns LOST: emit ReacquisitionNeeded event.
- Attempt micro-SpinSearch (±90° from last heading) before full 360° SpinSearch.
- If two consecutive SpinSearches fail: stop, report TargetNotFound to user, request clarification.

### Task Decomposition Template
For any navigation task, decompose as:
1. Identify target (semantic_map_query → grounding_detector → spin_search)
2. Ground target region (grounding_detector or vlm_ground)
3. Lock tracker (tracker_lock)
4. Approach in slow_crawl (move commands)
5. Verify arrival (vlm_verify)
6. Emit TaskComplete or TaskStepComplete
"""

    def _build_executor_messages(self, state: NovaState) -> list[BaseMessage]:
        if state["messages"]:
            return state["messages"]

        content = f"Mission: {state['mission']}"
        if state["route"] == "COMPLEX" and state["plan"]:
            content += f"\n\nExecution sketch:\n{state['plan']}"
        if state["clarify_a"]:
            content += f"\n\nClarification: {state['clarify_a']}"
        if state["retry_count"] > 0:
            content += f"\n\n[RETRY #{state['retry_count']}] Previous attempt failed. Try a different approach."

        return [
            SystemMessage(content=self._executor_system()),
            HumanMessage(content=content),
        ]

    def _reviewer_system(self) -> str:
        return """You are NOVA's quality reviewer.

PASS if:
  • The task was completed (navigation reached, object found, message sent, file written)
  • A clarifying question was asked (valid)
  • The error is non-critical and the agent recovered

RETRY if:
  • A tool returned a hard error AND the task is clearly incomplete
  • The agent produced empty or completely off-topic output

Reply exactly: VERDICT: PASS  or  VERDICT: RETRY — <reason>
"""

    # ── Utility ────────────────────────────────────────────────────────────────

    def _extract_final(self, state: NovaState) -> str:
        msgs = state.get("messages", [])
        last_ai = next(
            (m for m in reversed(msgs) if isinstance(m, AIMessage) and m.content), None
        )
        if last_ai:
            return last_ai.content
        tool_results = [m for m in msgs if isinstance(m, ToolMessage)]
        if tool_results:
            lines = [f"[{getattr(m,'name','tool')}] {str(m.content)[:100]}"
                     for m in tool_results[-3:]]
            return "Done. " + " | ".join(lines)
        return "Mission complete."

    # ── VLM bounding box (used by LockAndTrack tool) ──────────────────────────

    def get_vlm_bounding_box(self, frame, description: str):
        """
        Ask the vision LLM to locate an object. Returns (x,y,w,h) or None.
        Retries up to 3 times on degenerate boxes.
        """
        import base64
        import cv2

        if frame is None:
            return None

        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))
            h, w  = frame.shape[:2]

        _, buf  = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64     = base64.b64encode(buf).decode()

        sys_prompt = (
            f"You are a robotic vision module. Image is {w}x{h} pixels. "
            "Return ONLY JSON: {\"box\": [ymin, xmin, ymax, xmax]} "
            "using 0-1000 scale (1000 = full width/height). "
            "If not visible: {\"box\": []}."
        )

        MIN_SIDE = 20
        MIN_AREA = 0.35

        for attempt in range(3):
            try:
                resp = self._llm.invoke([
                    SystemMessage(content=sys_prompt),
                    HumanMessage(content=[
                        {"type": "text", "text": f"Find: {description}. Use 0-1000 coords."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ]),
                ])
                text = resp.content.strip()
                # Strip fences
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
                data = json.loads(text)
                box  = data.get("box", [])
                if not box:
                    return None
                if len(box) != 4:
                    continue

                ymin, xmin, ymax, xmax = box
                # Auto-detect 0-1 vs 0-1000
                if all(0 <= v <= 1.0 for v in box):
                    px = int(xmin * w); py = int(ymin * h)
                    pw = int((xmax - xmin) * w); ph = int((ymax - ymin) * h)
                else:
                    px = int(xmin / 1000 * w); py = int(ymin / 1000 * h)
                    pw = int((xmax - xmin) / 1000 * w); ph = int((ymax - ymin) / 1000 * h)

                # Clamp
                px = max(0, min(px, w - 1))
                py = max(0, min(py, h - 1))
                pw = max(1, min(pw, w - px))
                ph = max(1, min(ph, h - py))

                if pw < MIN_SIDE or ph < MIN_SIDE:
                    print(f"[NOVA VLM] Box too small ({pw}×{ph}), retry.")
                    continue

                raw_area = (xmax - xmin) * (ymax - ymin)
                if raw_area > 0 and (pw * ph) / (raw_area / 1e6 * w * h) < MIN_AREA:
                    continue

                print(f"[NOVA VLM] Found '{description}' at ({px},{py},{pw},{ph})")
                return (px, py, pw, ph)

            except Exception as e:
                print(f"[NOVA VLM] Attempt {attempt+1} failed: {e}")

        return None
