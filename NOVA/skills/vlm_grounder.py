import json
import base64
import cv2
import asyncio
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from ..tools.base import Tool, ToolResult

class VLMGroundTool(Tool):
    name = "vlm_ground"
    description = (
        "Uses the cloud VLM to find the bounding box of abstract spatial targets "
        "(e.g., 'foot of the chair', 'gap under the table'). Use ONLY when grounding_detector fails."
    )
    parameters = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Description of the abstract spatial region."
            }
        },
        "required": ["description"]
    }

    async def execute(self, description: str) -> ToolResult:
        if not self.context.camera:
            return ToolResult(success=False, message="Camera not available")

        # 1. Check semantic memory cache first
        from ..memory.semantic_map import get_semantic_map
        smap = get_semantic_map()
        cached_bbox = smap.get_vlm_cache(description)
        if cached_bbox:
            return ToolResult(
                success=True, 
                message=f"Found cached VLM result for '{description}'.", 
                data={"found": True, "bbox": cached_bbox, "confidence": 1.0, "cached": True}
            )

        # 2. Get sharpest frame
        frame = self.context.camera.get_sharpest_frame(downscale_to=(800, 450))
        if frame is None:
            return ToolResult(success=False, message="Could not acquire frame")

        # 3. Encode image
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf).decode()

        sys_prompt = """You are a spatial grounding assistant for a mobile robot.
You will be given an image and a description of a region of interest.
Respond ONLY with a JSON object. No explanation, no markdown, no preamble.
Format:
{"bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0, "reasoning": "one sentence"}
Where bbox coordinates are pixel values on a 800x450 image.
If the described region is not visible, respond: {"bbox": null, "confidence": 0.0, "reasoning": "not visible"}"""

        msg_content = [
            {"type": "text", "text": f"Find the bounding box of: {description}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]

        try:
            # We enforce token limits by rebinding the llm
            # This works for ChatOpenAI which allows overriding kwargs
            llm_bound = self.context._llm.bind(max_tokens=80)
            
            resp = await asyncio.to_thread(
                llm_bound.invoke,
                [SystemMessage(content=sys_prompt), HumanMessage(content=msg_content)]
            )
            
            text = resp.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
                
            data = json.loads(text)
            
            if data.get("bbox") is None or data.get("confidence", 0.0) < 0.4:
                return ToolResult(success=True, message=f"Target not visible: {data.get('reasoning')}", data={"found": False})
                
            result_data = {
                "found": True,
                "bbox": data["bbox"],
                "confidence": data["confidence"],
                "reasoning": data.get("reasoning", "")
            }
            
            # Cache the result
            smap.cache_vlm_result(description, data["bbox"])
            
            return ToolResult(success=True, message=f"Grounded '{description}' at {data['bbox']}", data=result_data)
            
        except Exception as e:
            return ToolResult(success=False, message=f"VLM call failed: {e}")

class VLMVerifyTool(Tool):
    name = "vlm_verify"
    description = "Uses the cloud VLM to visually verify if the robot is positioned at or near a described target. Call this after completing navigation."
    parameters = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Description of where the robot should be (e.g. 'near the chair', 'left of the table')."
            }
        },
        "required": ["description"]
    }

    async def execute(self, description: str) -> ToolResult:
        if not self.context.camera:
            return ToolResult(success=False, message="Camera not available")

        frame = self.context.camera.get_sharpest_frame(downscale_to=(800, 450))
        if frame is None:
            return ToolResult(success=False, message="Could not acquire frame")

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf).decode()

        sys_prompt = """You are a verification assistant for a mobile robot.
You will be given an image and a description of what the robot should be near or positioned at.
Respond ONLY with a JSON object. No explanation, no markdown, no preamble.
Format:
{"success": true|false, "confidence": 0.0-1.0, "reason": "one sentence"}"""

        msg_content = [
            {"type": "text", "text": f"Is the robot now positioned at or very near: {description}?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]

        try:
            llm_bound = self.context._llm.bind(max_tokens=60)
            
            resp = await asyncio.to_thread(
                llm_bound.invoke,
                [SystemMessage(content=sys_prompt), HumanMessage(content=msg_content)]
            )
            
            text = resp.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
                
            data = json.loads(text)
            
            return ToolResult(
                success=True, 
                message=f"Verification {'successful' if data.get('success') else 'failed'}: {data.get('reason')}", 
                data=data
            )
            
        except Exception as e:
            return ToolResult(success=False, message=f"VLM verification failed: {e}")
