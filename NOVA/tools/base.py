"""
NOVA Tool Base
--------------
Base class for all agent tools.
"""

from typing import Dict, Any

class ToolResult:
    def __init__(self, success: bool, message: str, data: Any = None):
        self.success = success
        self.message = message
        self.data = data
        
class Tool:
    name: str = "base_tool"
    description: str = "Base tool description."
    parameters: dict = {}
    
    def __init__(self, context):
        """
        context is the NovaAgent instance containing references to
        vision, navigation, memory, config, robot, etc.
        """
        self.context = context
        
    async def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError("Tool must implement execute()")
