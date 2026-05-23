"""
NOVA Agent Core
---------------
The main agent loop that ties all subsystems together.
"""

import asyncio
import time
import threading
import queue

from .config import NovaConfig
from .vision.detector import ObjectDetector
from .vision.obstacle import ObstacleDetector
from .navigation.spatial_map import SpatialMap
from .navigation.odometry import Odometry
from .navigation.path_planner import PathPlanner
from .navigation.driver import NavigationDriver
from .memory.memory_store import MemoryStore
from .planner import TaskPlanner

# Tools
from .tools.navigation import NavigateTo
from .tools.vision_tools import ScanFor, ScanRoom, LockAndTrack
from .tools.sensors import LogSensors
from .tools.comms import Notify
from .tools.control import DriveRaw, Remember, DriveUntilObstacle

class NovaAgent:
    def __init__(self, robot_interface=None, camera_interface=None, send_ws_callback=None):
        self.config = NovaConfig()
        
        # Interfaces
        self.robot = robot_interface
        self.camera = camera_interface
        self.send_ws_callback = send_ws_callback # Function to send WebSocket messages
        
        # State
        self.is_running = False
        self.current_task = None
        self.status = "idle"
        self.command_queue = queue.Queue()
        
        # Initialize Subsystems
        self.detector = ObjectDetector(self.config)
        self.obstacle_detector = ObstacleDetector(self.config, self.robot)
        
        self.odometry = Odometry(self.config)
        self.spatial_map = SpatialMap(self.config)
        self.path_planner = PathPlanner(self.config, self.spatial_map)
        self.driver = NavigationDriver(self.config, self.robot, self.odometry, self.obstacle_detector)
        
        self.memory = MemoryStore(self.config)
        
        # Initialize Tools
        tools_list = [
            NavigateTo(self),
            ScanFor(self),
            ScanRoom(self),
            LogSensors(self),
            Notify(self),
            DriveRaw(self),
            Remember(self),
            DriveUntilObstacle(self),
            LockAndTrack(self)
        ]
        self.tools = {tool.name: tool for tool in tools_list}
        
        self.planner = TaskPlanner(self.config, tools_list)
        
        self._loop_thread = None
        self._async_loop = None
        
    def start(self):
        """Start the agent background thread."""
        if self.is_running:
            return
        self.is_running = True
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()
        print("[NOVA] Agent started.")
        
    def stop(self):
        """Stop the agent."""
        self.is_running = False
        if self.driver:
            self.driver.stop()
        if self._async_loop:
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        print("[NOVA] Agent stopped.")
        
    def submit_command(self, command: str):
        """Submit a new command from the user."""
        # Cancel current task if running
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            self.driver.stop()
            
        self.command_queue.put(command)
        
    def send_status(self, status: str):
        """Update agent status and notify GUI."""
        self.status = status
        print(f"[NOVA] {status}")
        if self.send_ws_callback:
            self.send_ws_callback("nova_status", {"status": status})
            
    def send_chat(self, message: str):
        """Send a chat message to the GUI."""
        print(f"[NOVA Chat] {message}")
        if self.send_ws_callback:
            self.send_ws_callback("nova_chat", {"sender": "NOVA", "message": message})
            
    def _run_loop(self):
        """Main thread loop. Creates an asyncio event loop for the tools."""
        self._async_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._async_loop)
        
        # Start the background update task (10Hz)
        self._async_loop.create_task(self._hardware_update_loop())
        
        # Start the command processor task
        self._async_loop.create_task(self._process_commands())
        
        try:
            self._async_loop.run_forever()
        except Exception as e:
            print(f"[NOVA] Fatal error in agent loop: {e}")
        finally:
            self._async_loop.close()
            
    async def _hardware_update_loop(self):
        """Update odometry and run driver 10 times a second."""
        while self.is_running:
            # Update odometry from current PWM
            if self.robot:
                self.odometry.update(self.robot.left_speed, self.robot.right_speed)
                
            # Update obstacle detector
            self.obstacle_detector.update()
            
            # Run driver step
            if self.driver.is_navigating:
                status = self.driver.update()
                if status == 'blocked':
                    self.send_status("Blocked by obstacle!")
                elif status == 'stuck':
                    self.send_status("Stuck. Please help!")
                    
            await asyncio.sleep(1.0 / self.config.navigation.control_hz)
            
    async def _process_commands(self):
        """Process incoming natural language commands."""
        while self.is_running:
            try:
                # Non-blocking get
                command = self.command_queue.get_nowait()
                self.current_task = asyncio.create_task(self._execute_command(command))
            except queue.Empty:
                pass
                
            await asyncio.sleep(0.1)
            
    async def _execute_command(self, command: str):
        try:
            self.send_status(f"Planning: '{command}'")
            self.send_chat(f"Thinking about how to: {command}")
            
            # 1. Plan
            plan = self.planner.generate_plan(command)
            
            if not plan:
                self.send_chat("I couldn't figure out how to do that.")
                self.send_status("idle")
                return
                
            self.send_chat(f"I have a plan with {len(plan)} steps.")
            
            # 2. Execute
            for i, step in enumerate(plan):
                if not self.is_running:
                    break
                    
                action = step.get("action")
                args = step.get("args", {})
                
                if action not in self.tools:
                    self.send_chat(f"Error: I don't know how to use the tool '{action}'.")
                    break
                    
                self.send_status(f"Executing step {i+1}: {action}")
                
                tool = self.tools[action]
                
                try:
                    result = await tool.execute(**args)
                    if not result.success:
                        self.send_chat(f"Step failed: {result.message}")
                        break
                except asyncio.CancelledError:
                    self.send_chat("Task cancelled.")
                    raise
                except Exception as e:
                    self.send_chat(f"An error occurred executing {action}: {e}")
                    break
                    
            # Generate final output
            task_history = [f"Ran {step.get('action')} with success status: {result.success}" for step in plan]
            summary = self.planner.generate_summary(command, task_history)
            self.send_chat(summary)
            
            self.send_status("idle")
            self.memory.add_to_history(command, "success")
            
        except asyncio.CancelledError:
            self.send_status("idle")
            self.memory.add_to_history(command, "cancelled")
