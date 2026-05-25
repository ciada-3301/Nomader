"""
NOVA Communication Tools
------------------------
Notify the user and ask clarifying questions mid-mission.
"""

import asyncio
from .base import Tool, ToolResult


class Notify(Tool):
    name = "notify"
    description = "Send a message to the user through the chat interface."
    parameters = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Message to send."},
        },
        "required": ["message"],
    }

    async def execute(self, message: str) -> ToolResult:
        self.context.send_chat(message, from_notify=True)
        return ToolResult(True, "Message sent.")


class AskUser(Tool):
    name = "ask_user"
    description = (
        "Pause and ask the user a clarifying question. "
        "Use only when you genuinely cannot proceed without human input. "
        "One question at a time."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question to ask."},
        },
        "required": ["question"],
    }

    async def execute(self, question: str) -> ToolResult:
        # Delegate to agent's inter-thread ask bridge
        answer = await asyncio.to_thread(self.context.ask_user_sync, question)
        return ToolResult(True, f"User answered: {answer}", data=answer)
