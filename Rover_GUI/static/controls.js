/**
 * THE NOMADER - Control System
 * Real-time rover control with WebSocket communication
 * Updated to use differential drive commands (forward, backward, left, right, arc_left, arc_right)
 */

// ========================================
// GLOBAL STATE
// ========================================

const state = {
    connected: false,
    command: 'halt',
    speed: 0,
    bias: 0,
    pitch: 90,
    yaw: 90,
    light: 0,
    keysPressed: new Set()
};

// Socket.IO connection
const socket = io();

// Control parameters
const MOTOR_MAX = 255;
const MOTOR_STEP = 15;
const GIMBAL_STEP = 5;
const LIGHT_STEP = 25;
const JOYSTICK_DEADZONE = 0.15;
const UPDATE_RATE = 50; // ms (20 updates/sec)
const ARC_THRESHOLD = 0.3; // Threshold for switching from straight to arc movement

// ========================================
// INITIALIZATION
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    initializeSocketIO();
    initializeJoystick();
    initializeGimbal();
    initializeLightControl();
    initializeKeyboard();
    initializeButtons();
    startUpdateLoop();
});

// ========================================
// SOCKET.IO HANDLERS
// ========================================

function initializeSocketIO() {
    socket.on('connect', () => {
        console.log('✓ Connected to rover');
        updateConnectionStatus(true);
    });

    socket.on('disconnect', () => {
        console.log('✗ Disconnected from rover');
        updateConnectionStatus(false);
    });

    socket.on('status', (data) => {
        state.connected = data.connected;
        updateConnectionStatus(data.connected);
    });

    socket.on('state_update', (data) => {
        // Update UI with rover state
        updateDisplays(data);
    });

    socket.on('stats_update', (data) => {
        updateSystemStats(data);
    });
}

function updateConnectionStatus(connected) {
    state.connected = connected;
    const statusEl = document.getElementById('connectionStatus');
    const statusText = statusEl.querySelector('.status-text');
    
    if (connected) {
        statusEl.classList.remove('disconnected');
        statusEl.classList.add('connected');
        statusText.textContent = 'CONNECTED';
    } else {
        statusEl.classList.remove('connected');
        statusEl.classList.add('disconnected');
        statusText.textContent = 'DISCONNECTED';
    }
}

// ========================================
// JOYSTICK CONTROL (DRIVE)
// ========================================

function initializeJoystick() {
    const joystick = document.getElementById('driveJoystick');
    const handle = document.getElementById('driveHandle');
    
    let isDragging = false;
    let centerX, centerY, radius;

    function updateJoystickGeometry() {
        const rect = joystick.getBoundingClientRect();
        centerX = rect.width / 2;
        centerY = rect.height / 2;
        radius = Math.min(centerX, centerY) - 30;
    }

    function handleStart(e) {
        isDragging = true;
        updateJoystickGeometry();
        handleMove(e);
    }

    function handleMove(e) {
        if (!isDragging) return;

        e.preventDefault();
        const touch = e.touches ? e.touches[0] : e;
        const rect = joystick.getBoundingClientRect();
        
        let x = touch.clientX - rect.left - centerX;
        let y = touch.clientY - rect.top - centerY;
        
        // Calculate distance from center
        const distance = Math.sqrt(x * x + y * y);
        
        // Constrain to circle
        if (distance > radius) {
            const angle = Math.atan2(y, x);
            x = Math.cos(angle) * radius;
            y = Math.sin(angle) * radius;
        }
        
        // Apply deadzone
        const normalizedDistance = distance / radius;
        if (normalizedDistance < JOYSTICK_DEADZONE) {
            x = 0;
            y = 0;
        }
        
        // Update handle position
        handle.style.transform = `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))`;
        
        // Convert to drive command
        updateDriveFromJoystick(x / radius, y / radius);
    }

    function handleEnd() {
        if (!isDragging) return;
        isDragging = false;
        
        // Return to center with smooth animation
        handle.style.transition = 'transform 0.3s ease-out';
        handle.style.transform = 'translate(-50%, -50%)';
        
        setTimeout(() => {
            handle.style.transition = '';
        }, 300);
        
        // Stop motors
        state.command = 'halt';
        state.speed = 0;
        state.bias = 0;
        sendDriveCommand();
    }

    // Mouse events
    joystick.addEventListener('mousedown', handleStart);
    document.addEventListener('mousemove', handleMove);
    document.addEventListener('mouseup', handleEnd);
    
    // Touch events
    joystick.addEventListener('touchstart', handleStart, { passive: false });
    document.addEventListener('touchmove', handleMove, { passive: false });
    document.addEventListener('touchend', handleEnd);
    
    // Prevent context menu
    joystick.addEventListener('contextmenu', e => e.preventDefault());
}

function updateDriveFromJoystick(x, y) {
    /**
     * Convert joystick position to drive commands
     * X-axis: turning (left/right)
     * Y-axis: forward/backward (inverted)
     */
    
    const forward = -y; // Invert Y axis (up is forward)
    const turn = x;
    
    // Calculate magnitude for speed
    const magnitude = Math.sqrt(forward * forward + turn * turn);
    const speed = Math.round(magnitude * MOTOR_MAX);
    
    // Determine command based on joystick position
    let command = 'halt';
    let bias = 0;
    
    if (magnitude > 0.1) { // Minimum threshold to register movement
        const turnRatio = Math.abs(turn) / Math.max(magnitude, 0.001);
        
        if (Math.abs(turn) < ARC_THRESHOLD && forward > 0) {
            // Mostly forward
            command = 'forward';
        } else if (Math.abs(turn) < ARC_THRESHOLD && forward < 0) {
            // Mostly backward
            command = 'backward';
        } else if (Math.abs(forward) < ARC_THRESHOLD && turn < 0) {
            // Mostly left turn in place
            command = 'left';
        } else if (Math.abs(forward) < ARC_THRESHOLD && turn > 0) {
            // Mostly right turn in place
            command = 'right';
        } else if (forward > 0 && turn < 0) {
            // Arc left while going forward
            command = 'arc_left';
            bias = Math.abs(turn); // Bias determines how tight the arc is
        } else if (forward > 0 && turn > 0) {
            // Arc right while going forward
            command = 'arc_right';
            bias = Math.abs(turn);
        } else if (forward < 0 && turn < 0) {
            // Arc left while reversing
            command = 'arc_left';
            bias = -Math.abs(turn); // Negative bias for reverse
        } else if (forward < 0 && turn > 0) {
            // Arc right while reversing
            command = 'arc_right';
            bias = -Math.abs(turn);
        }
    }
    
    state.command = command;
    state.speed = speed;
    state.bias = bias;
    
    sendDriveCommand();
}

// ========================================
// GIMBAL CONTROL
// ========================================

function initializeGimbal() {
    const gimbal = document.getElementById('gimbalControl');
    const handle = document.getElementById('gimbalHandle');
    
    let isDragging = false;
    let centerX, centerY, radius;

    function updateGimbalGeometry() {
        const rect = gimbal.getBoundingClientRect();
        centerX = rect.width / 2;
        centerY = rect.height / 2;
        radius = Math.min(centerX, centerY) - 30;
    }

    function handleStart(e) {
        isDragging = true;
        updateGimbalGeometry();
        handleMove(e);
    }

    function handleMove(e) {
        if (!isDragging) return;

        e.preventDefault();
        const touch = e.touches ? e.touches[0] : e;
        const rect = gimbal.getBoundingClientRect();
        
        let x = touch.clientX - rect.left - centerX;
        let y = touch.clientY - rect.top - centerY;
        
        const distance = Math.sqrt(x * x + y * y);
        
        if (distance > radius) {
            const angle = Math.atan2(y, x);
            x = Math.cos(angle) * radius;
            y = Math.sin(angle) * radius;
        }
        
        handle.style.transform = `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))`;
        
        // Convert to gimbal angles
        // X: yaw (0-180, center at 90)
        // Y: pitch (0-180, center at 90)
        state.yaw = Math.round(90 + (x / radius) * 90);
        state.pitch = Math.round(90 - (y / radius) * 90);
        
        sendGimbalCommand();
    }

    function handleEnd() {
        isDragging = false;
    }

    gimbal.addEventListener('mousedown', handleStart);
    document.addEventListener('mousemove', handleMove);
    document.addEventListener('mouseup', handleEnd);
    
    gimbal.addEventListener('touchstart', handleStart, { passive: false });
    document.addEventListener('touchmove', handleMove, { passive: false });
    document.addEventListener('touchend', handleEnd);
    
    gimbal.addEventListener('contextmenu', e => e.preventDefault());
}

// ========================================
// LIGHT CONTROL
// ========================================

function initializeLightControl() {
    const slider = document.getElementById('lightSlider');
    const fill = document.getElementById('lightFill');
    const valueDisplay = document.getElementById('lightValue');
    
    slider.addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        state.light = value;
        
        // Update fill
        const percentage = (value / 255) * 100;
        fill.style.width = `${percentage}%`;
        
        // Update display
        valueDisplay.textContent = value;
        
        sendLightCommand();
    });
}

// ========================================
// KEYBOARD CONTROLS
// ========================================

function initializeKeyboard() {
    document.addEventListener('keydown', (e) => {
        // Prevent default for control keys
        if (['w', 'a', 's', 'd', 'q', 'e', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key.toLowerCase())) {
            e.preventDefault();
        }
        
        state.keysPressed.add(e.key.toLowerCase());
    });

    document.addEventListener('keyup', (e) => {
        state.keysPressed.delete(e.key.toLowerCase());
    });
}

function processKeyboardInput() {
    const keys = state.keysPressed;
    
    // Drive controls (WASD)
    let command = 'halt';
    let speed = MOTOR_MAX;
    let bias = 0.7; // Default arc bias for keyboard
    
    // Determine command based on key combinations
    if (keys.has('w') && keys.has('a')) {
        command = 'arc_left_forward';
    } else if (keys.has('w') && keys.has('d')) {
        command = 'arc_right_forward';
    } else if (keys.has('s') && keys.has('a')) {
        command = 'arc_left_backward';
        bias = -0.7; // Reverse arc
    } else if (keys.has('s') && keys.has('d')) {
        command = 'arc_right_backward';
        bias = -0.7; // Reverse arc
    } else if (keys.has('w')) {
        command = 'forward';
    } else if (keys.has('s')) {
        command = 'backward';
    } else if (keys.has('a')) {
        command = 'left';
    } else if (keys.has('d')) {
        command = 'right';
    }
    
    // Update state and send command if changed
    if (state.command !== command || (command !== 'halt' && state.speed !== speed)) {
        state.command = command;
        state.speed = (command !== 'halt') ? speed : 0;
        state.bias = bias;
        sendDriveCommand();
    }
    
    // Gimbal controls (Arrow keys)
    if (keys.has('arrowup')) {
        state.pitch = Math.max(0, state.pitch - GIMBAL_STEP);
        sendGimbalCommand();
    }
    if (keys.has('arrowdown')) {
        state.pitch = Math.min(180, state.pitch + GIMBAL_STEP);
        sendGimbalCommand();
    }
    if (keys.has('arrowleft')) {
        state.yaw = Math.max(0, state.yaw - GIMBAL_STEP);
        sendGimbalCommand();
    }
    if (keys.has('arrowright')) {
        state.yaw = Math.min(180, state.yaw + GIMBAL_STEP);
        sendGimbalCommand();
    }
    
    // Light controls (Q/E)
    if (keys.has('q')) {
        state.light = Math.max(0, state.light - LIGHT_STEP);
        updateLightUI();
        sendLightCommand();
    }
    if (keys.has('e')) {
        state.light = Math.min(255, state.light + LIGHT_STEP);
        updateLightUI();
        sendLightCommand();
    }
}

// ========================================
// BUTTONS
// ========================================

function initializeButtons() {
    document.getElementById('emergencyStop').addEventListener('click', () => {
        console.log('⚠️ EMERGENCY STOP');
        socket.emit('emergency_stop');
        state.command = 'halt';
        state.speed = 0;
        state.bias = 0;
        state.light = 0;
        updateDisplays(state);
    });
    
    document.getElementById('resetGimbal').addEventListener('click', () => {
        socket.emit('reset_gimbal');
        state.pitch = 90;
        state.yaw = 90;
        updateDisplays(state);
    });
}

// ========================================
// WEBSOCKET COMMANDS
// ========================================

let lastDriveCommand = { command: 'halt', speed: 0, bias: 0 };
let lastGimbalCommand = { pitch: 90, yaw: 90 };
let lastLightCommand = 0;

function sendDriveCommand() {
    // Only send if values changed
    if (lastDriveCommand.command !== state.command || 
        lastDriveCommand.speed !== state.speed || 
        lastDriveCommand.bias !== state.bias) {
        
        socket.emit('drive', {
            command: state.command,
            speed: state.speed,
            bias: state.bias
        });
        
        lastDriveCommand = { 
            command: state.command, 
            speed: state.speed, 
            bias: state.bias 
        };
    }
}

function sendGimbalCommand() {
    if (lastGimbalCommand.pitch !== state.pitch || lastGimbalCommand.yaw !== state.yaw) {
        socket.emit('gimbal', {
            pitch: state.pitch,
            yaw: state.yaw
        });
        lastGimbalCommand = { pitch: state.pitch, yaw: state.yaw };
    }
}

function sendLightCommand() {
    if (lastLightCommand !== state.light) {
        socket.emit('light', {
            brightness: state.light
        });
        lastLightCommand = state.light;
    }
}

// ========================================
// UI UPDATES
// ========================================

function updateDisplays(data) {
    // Command and speed display
    const commandText = data.command || state.command;
    const speedValue = data.speed || state.speed;
    
    document.getElementById('speedDisplay').textContent = speedValue;
    
    // Show current command
    const commandDisplay = document.getElementById('commandDisplay') || createCommandDisplay();
    commandDisplay.textContent = commandText.toUpperCase();
    
    // Gimbal
    document.getElementById('pitchDisplay').textContent = data.pitch || state.pitch;
    document.getElementById('yawDisplay').textContent = data.yaw || state.yaw;
    
    // Light
    if (data.light !== undefined) {
        state.light = data.light;
        updateLightUI();
    }
    
    // Bias indicator for arc movements
    if (data.bias !== undefined && (data.command === 'arc_left' || data.command === 'arc_right')) {
        const biasDisplay = document.getElementById('biasDisplay') || createBiasDisplay();
        biasDisplay.textContent = `Bias: ${(data.bias * 100).toFixed(0)}%`;
    }
}

function createCommandDisplay() {
    const display = document.createElement('div');
    display.id = 'commandDisplay';
    display.style.cssText = 'position: fixed; top: 10px; right: 10px; background: rgba(0,0,0,0.7); color: #0f0; padding: 10px; border-radius: 5px; font-family: monospace;';
    document.body.appendChild(display);
    return display;
}

function createBiasDisplay() {
    const display = document.createElement('div');
    display.id = 'biasDisplay';
    display.style.cssText = 'position: fixed; top: 50px; right: 10px; background: rgba(0,0,0,0.7); color: #0ff; padding: 10px; border-radius: 5px; font-family: monospace;';
    document.body.appendChild(display);
    return display;
}

function updateLightUI() {
    const slider = document.getElementById('lightSlider');
    const fill = document.getElementById('lightFill');
    const valueDisplay = document.getElementById('lightValue');
    
    slider.value = state.light;
    fill.style.width = `${(state.light / 255) * 100}%`;
    valueDisplay.textContent = state.light;
}

function updateSystemStats(data) {
    if (data.cpu_usage !== undefined) {
        document.getElementById('cpuUsage').textContent = `${Math.round(data.cpu_usage)}%`;
    }
    if (data.cpu_temp !== undefined) {
        document.getElementById('cpuTemp').textContent = `${data.cpu_temp}°C`;
    }
    if (data.ram_percent !== undefined) {
        document.getElementById('ramUsage').textContent = `${Math.round(data.ram_percent)}%`;
    }
}

// ========================================
// UPDATE LOOP
// ========================================

function startUpdateLoop() {
    setInterval(() => {
        processKeyboardInput();
    }, UPDATE_RATE);
}

// ========================================
// UTILITY FUNCTIONS
// ========================================

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

// Prevent page scrolling on mobile
document.body.addEventListener('touchmove', (e) => {
    if (e.target.closest('.joystick, .gimbal-control')) {
        e.preventDefault();
    }
}, { passive: false });