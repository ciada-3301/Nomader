"""
NOVA Task Planner
-----------------
Interfaces with an LLM (Ollama/OpenAI-compat) to decompose commands into tool actions.
Includes VLM bounding box support for open-vocabulary object detection.

Changes from v1:
  - _box_to_pixels now returns None when clamping destroys the box (out-of-frame coords)
  - get_vlm_bounding_box retries up to MAX_VLM_RETRIES before giving up
  - VLM system prompt explicitly forbids raw pixel coords and is more assertive about format
  - Added _validate_roi to centralise tracker-init sanity checks
  - _parse_json_safe logs the raw text on failure to aid future debugging
  - Minor: constants extracted to class-level; docstrings tightened
"""

import json
import base64
import cv2
from typing import List, Dict, Tuple, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


class TaskPlanner:

    # ── Class-level constants ──────────────────────────────────────────────────
    MAX_VLM_RETRIES   = 3     # how many times to re-ask the VLM before giving up
    MIN_BOX_SIDE_PX   = 20   # both w and h must be at least this many pixels
    MIN_AREA_RETAINED = 0.35  # clamped area must be ≥35 % of the raw area

    def __init__(self, config, tools):
        self.config = config
        self.tools = {tool.name: tool for tool in tools}

        api_key = self.config.planner.llm_api_key
        if not api_key or api_key in ("ollama", ""):
            api_key = "no-key"

        self.llm = ChatOpenAI(
            model=self.config.planner.llm_model,
            base_url=self.config.planner.llm_base_url,
            api_key=api_key,
            temperature=self.config.planner.llm_temperature,
            timeout=self.config.planner.llm_timeout_s,
            max_tokens=4096,
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _strip_fences(self, text: str) -> str:
        """Strip markdown code fences that the model sometimes wraps JSON in."""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]   # drop the ```json line
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        return text.strip()

    def _parse_json_safe(self, text: str) -> Optional[dict]:
        """Try to parse JSON, stripping fences first.  Returns None on failure."""
        cleaned = self._strip_fences(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            print(f"[NOVA Planner] JSON parse failed ({exc}). Raw text was:\n{text}")
            return None

    def _box_to_pixels(
        self, box: list, frame_w: int, frame_h: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Convert [ymin, xmin, ymax, xmax] to (x, y, w, h) in pixel space.

        Auto-detects whether the model returned normalised (0–1) or raw pixel
        coordinates and converts accordingly.  After clamping to frame bounds the
        result is validated: if too little of the original box survives, returns
        None so the caller can retry rather than handing a mangled ROI to the
        tracker.

        Returns:
            (x, y, w, h) on success, or None if the box is degenerate / too far
            outside the frame to be useful.
        """
        ymin, xmin, ymax, xmax = box

        # ── Normalised vs raw pixel detection ─────────────────────────────────
        # Three distinct coordinate spaces are used by different models:
        #
        #   1. 0.0–1.0  fractional  — most OpenAI-compat and open models
        #   2. 0–1000   centipercent — Google models (Gemma, PaliGemma, Gemini)
        #                              return integers in [0, 1000]
        #   3. raw pixels           — some models ignore instructions entirely
        #
        # Detection order matters: check 0–1 first (strictest), then 0–1000
        # (values > 1 but all ≤ 1000), then fall through to raw pixels.
        if all(0.0 <= v <= 1.0 for v in box):
            # Space 1: true fractional normalised
            scale  = 1.0
            norm   = True
        elif all(v <= 1000 for v in box) and any(v > 1.0 for v in box):
            # Space 2: Google 0-1000 scale — divide by 1000 to normalise
            scale  = 1000.0
            norm   = True
            print(
                f"[NOVA Planner] Detected 0-1000 coordinate space (Google model). "
                f"Dividing by 1000. Raw box={box}"
            )
        else:
            norm  = False

        if norm:
            x_raw = int((xmin / scale) * frame_w)
            y_raw = int((ymin / scale) * frame_h)
            w_raw = int(((xmax - xmin) / scale) * frame_w)
            h_raw = int(((ymax - ymin) / scale) * frame_h)
        else:
            # Space 3: raw pixels — use as-is (clamping handles out-of-bounds)
            x_raw = int(xmin)
            y_raw = int(ymin)
            w_raw = int(xmax - xmin)
            h_raw = int(ymax - ymin)

        raw_area = w_raw * h_raw

        # ── Clamp to frame bounds ──────────────────────────────────────────────
        x_cl = max(0, min(x_raw, frame_w - 1))
        y_cl = max(0, min(y_raw, frame_h - 1))
        w_cl = max(1, min(w_raw, frame_w - x_cl))
        h_cl = max(1, min(h_raw, frame_h - y_cl))

        clamped_area  = w_cl * h_cl
        area_fraction = clamped_area / raw_area if raw_area > 0 else 0.0

        # ── Validate ───────────────────────────────────────────────────────────
        if w_cl < self.MIN_BOX_SIDE_PX or h_cl < self.MIN_BOX_SIDE_PX:
            print(
                f"[NOVA Planner] Box rejected — side too small after clamping: "
                f"({w_cl}px × {h_cl}px).  "
                f"Raw box={box}, frame={frame_w}×{frame_h}."
            )
            return None

        if area_fraction < self.MIN_AREA_RETAINED:
            print(
                f"[NOVA Planner] Box rejected — only {area_fraction*100:.0f}% of area "
                f"survives clamping (need ≥{self.MIN_AREA_RETAINED*100:.0f}%).  "
                f"Raw box={box}, frame={frame_w}×{frame_h}."
            )
            return None

        # Explicitly cast to plain Python int.
        # frame.shape[:2] returns numpy.int64, so clamped values can be numpy.int64
        # when the raw coord hits the frame boundary. OpenCV's C++ tracker binding
        # is strict: passing numpy.int64 to tracker.init() returns False on many builds.
        return (int(x_cl), int(y_cl), int(w_cl), int(h_cl))

    def _validate_roi(self, roi: Tuple[int, int, int, int], frame_w: int, frame_h: int) -> bool:
        """
        Final sanity-check before passing an ROI to a tracker.

        Ensures the rectangle is non-degenerate, fully inside the frame, and
        large enough for a tracker to build a meaningful appearance model.
        """
        x, y, w, h = roi

        if w <= 0 or h <= 0:
            print(f"[NOVA Planner] ROI validation failed: non-positive dimensions ({w}×{h}).")
            return False

        if x < 0 or y < 0 or (x + w) > frame_w or (y + h) > frame_h:
            print(
                f"[NOVA Planner] ROI validation failed: out of frame bounds "
                f"({x},{y},{w},{h}) vs frame {frame_w}×{frame_h}."
            )
            return False

        if w < self.MIN_BOX_SIDE_PX or h < self.MIN_BOX_SIDE_PX:
            print(
                f"[NOVA Planner] ROI validation failed: too small "
                f"({w}px × {h}px), need ≥{self.MIN_BOX_SIDE_PX}px each side."
            )
            return False

        return True

    # ──────────────────────────────────────────────────────────────────────────
    #  Plan generation
    # ──────────────────────────────────────────────────────────────────────────

    def generate_plan(self, command: str) -> List[Dict]:
        """Take a natural language command and return a list of tool actions."""
        messages = [
            SystemMessage(content=self._build_system_prompt()),
            HumanMessage(content=command),
        ]

        try:
            structured_llm = self.llm.bind(response_format={"type": "json_object"})
            response = structured_llm.invoke(messages)
            content = response.content

            parsed = self._parse_json_safe(content)
            if parsed is None:
                return self._extract_plan_from_text(content)

            if "plan" in parsed:
                return parsed["plan"]
            if isinstance(parsed, list):
                return parsed

            print(f"[NOVA Planner] Unexpected JSON structure: {parsed}")
            return []

        except Exception as e:
            print(f"[NOVA Planner] LLM call failed: {e}")
            return []

    def _extract_plan_from_text(self, text: str) -> List[Dict]:
        """Last-resort extraction of a plan array from fenced or malformed text."""
        try:
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
                parsed = json.loads(json_str)
                return parsed.get("plan", parsed) if isinstance(parsed, dict) else parsed
        except Exception:
            pass
        return []

    # ──────────────────────────────────────────────────────────────────────────
    #  VLM bounding box
    # ──────────────────────────────────────────────────────────────────────────

    def get_vlm_bounding_box(
        self, frame, object_description: str
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Use the vision model to locate an object in the frame.

        Retries up to MAX_VLM_RETRIES times when the model returns an
        out-of-bounds or otherwise degenerate box.  Each failed attempt logs
        the reason so you can see in the console exactly what the model
        returned and why it was rejected.

        Returns:
            (x, y, w, h) in pixel coords on success, or None if every attempt
            failed.
        """
        if frame is None:
            print("[NOVA Planner VLM] No frame available.")
            return None

        frame_h, frame_w = frame.shape[:2]

        # Downscale to 640-wide before encoding — faster, cheaper, sufficient
        if frame_w > 640:
            scale  = 640 / frame_w
            frame  = cv2.resize(frame, (640, int(frame_h * scale)))
            frame_h, frame_w = frame.shape[:2]

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64_image = base64.b64encode(buffer).decode('utf-8')

        system_prompt = (
            "You are a robotic vision module. "
            f"The image is exactly {frame_w}x{frame_h} pixels. "
            "Your ONLY job is to return a JSON object — no markdown, no explanation. "
            "Format: {\"box\": [ymin, xmin, ymax, xmax]} "
            "using coordinates in the range 0 to 1000, where 0 is the top/left "
            "edge and 1000 is the bottom/right edge. "
            "Example for an object in the centre of the frame: "
            "{\"box\": [350, 350, 650, 650]}. "
            "If the object is not visible, return {\"box\": []}."
        )

        for attempt in range(1, self.MAX_VLM_RETRIES + 1):
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": (
                                f"Find the bounding box for: {object_description}. "
                                "Remember: use 0-1000 integer coordinates."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                        },
                    ]
                ),
            ]

            try:
                vlm      = self.llm.bind(temperature=0.0, response_format={"type": "json_object"})
                response = vlm.invoke(messages)
                raw      = response.content
                print(f"[NOVA VLM RAW] attempt {attempt}/{self.MAX_VLM_RETRIES}: {raw}")

                parsed = self._parse_json_safe(raw)
                if parsed is None:
                    print(f"[NOVA Planner VLM] Attempt {attempt}: response was not valid JSON.")
                    continue

                box = parsed.get("box", [])

                if len(box) == 0:
                    print(f"[NOVA Planner VLM] Attempt {attempt}: model reports object not visible.")
                    return None   # No point retrying — the model can't see it

                if len(box) != 4:
                    print(
                        f"[NOVA Planner VLM] Attempt {attempt}: "
                        f"expected 4 values, got {len(box)}: {box}"
                    )
                    continue

                roi = self._box_to_pixels(box, frame_w, frame_h)
                if roi is None:
                    print(
                        f"[NOVA Planner VLM] Attempt {attempt}: "
                        f"box {box} failed pixel validation — will retry."
                    )
                    continue

                if not self._validate_roi(roi, frame_w, frame_h):
                    print(
                        f"[NOVA Planner VLM] Attempt {attempt}: "
                        f"ROI {roi} failed final validation — will retry."
                    )
                    continue

                print(
                    f"[NOVA Planner VLM] Attempt {attempt}: "
                    f"accepted ROI {roi} for '{object_description}'."
                )
                return roi

            except Exception as e:
                print(f"[NOVA Planner VLM] Attempt {attempt}: exception — {e}")

        print(
            f"[NOVA Planner VLM] All {self.MAX_VLM_RETRIES} attempts failed "
            f"for '{object_description}'."
        )
        return None

    # ──────────────────────────────────────────────────────────────────────────
    #  Summary
    # ──────────────────────────────────────────────────────────────────────────

    def generate_summary(self, command: str, history: List[str]) -> str:
        """Generate a short friendly summary of what was accomplished."""
        history_str = "\n".join(f"- {h}" for h in history)
        prompt = (
            f"The user asked: '{command}'.\n"
            f"Here is what I did:\n{history_str}\n\n"
            "Provide a short, friendly summary (1-2 sentences) of what was accomplished."
        )

        try:
            summary_llm = self.llm.bind(temperature=0.4)
            response    = summary_llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception:
            return "Task execution completed."

    # ──────────────────────────────────────────────────────────────────────────
    #  System prompt
    # ──────────────────────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        tools_desc = []
        for name, tool in self.tools.items():
            params = json.dumps(tool.parameters)
            tools_desc.append(f"- **{name}**: {tool.description}\n  Parameters: {params}")

        tools_str = "\n".join(tools_desc)

        return f"""You are the task planner for NOVA, an autonomous rover.
Your job is to break down the user's command into a sequence of tool calls.

Available Tools:
{tools_str}

Respond with a JSON object containing a "plan" array.
Each item must have an "action" (tool name) and "args" (parameters dict).
Return ONLY the JSON — no markdown fences, no explanation.

Example:
User: "Go to the kitchen and look for a cup, then notify me."
Response:
{{
  "plan": [
    {{"action": "navigate_to", "args": {{"location": "kitchen"}}}},
    {{"action": "scan_for",    "args": {{"object_name": "cup"}}}},
    {{"action": "notify",      "args": {{"message": "Reached kitchen and scanned for a cup."}}}}
  ]
}}
"""