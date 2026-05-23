import asyncio
from .base import Tool, ToolResult

class Notify(Tool):
    name = "notify"
    description = "Send a message to the user through the chat interface."
    parameters = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "The message to send to the user."}
        },
        "required": ["message"]
    }
    
    async def execute(self, message: str) -> ToolResult:
        self.context.send_chat(message)
        return ToolResult(True, "Message sent to user.")
