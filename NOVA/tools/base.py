"""NOVA Tool Base"""
from typing import Any, Optional


class ToolResult:
    def __init__(self, success: bool, message: str, data: Any = None):
        self.success = success
        self.message = message
        self.data    = data

    def __repr__(self):
        return f"ToolResult(success={self.success}, message={self.message!r})"


class Tool:
    name:        str  = "base_tool"
    description: str  = "Base tool."
    parameters:  dict = {"type": "object", "properties": {}, "required": []}

    def __init__(self, context):
        # context is the NovaAgent instance
        self.context = context

    async def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError("Tool must implement execute()")
