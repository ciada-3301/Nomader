/**
 * THE NOMADER - Control System
 * Real-time rover control with WebSocket communication
 */

// ========================================
// GLOBAL STATE
// ========================================

const state = {
    connected: false,
    leftSpeed: 0,
    rightSpeed: 0,
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
        
        // Convert to motor values
        updateMotorsFromJoystick(x / radius, y / radius);
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
        state.leftSpeed = 0;
        state.rightSpeed = 0;
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

function updateMotorsFromJoystick(x, y) {
    // X-axis: turning (left/right)
    // Y-axis: forward/backward (inverted)
    
    const forward = -y; // Invert Y axis
    const turn = x;
    
    // Tank drive mixing
    let left = forward + turn;
    let right = forward - turn;
    
    // Normalize if values exceed [-1, 1]
    const maxValue = Math.max(Math.abs(left), Math.abs(right));
    if (maxValue > 1) {
        left /= maxValue;
        right /= maxValue;
    }
    
    // Scale to motor range
    state.leftSpeed = Math.round(left * MOTOR_MAX);
    state.rightSpeed = Math.round(right * MOTOR_MAX);
    
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
    let forward = 0;
    let turn = 0;
    
    if (keys.has('w')) forward += 1;
    if (keys.has('s')) forward -= 1;
    if (keys.has('a')) turn -= 1;
    if (keys.has('d')) turn += 1;
    
    // Only update motors if keyboard input is active
    if (keys.has('w') || keys.has('s') || keys.has('a') || keys.has('d')) {
        // Tank drive mixing
        let left = forward + turn;
        let right = forward - turn;
        
        // Normalize
        const maxValue = Math.max(Math.abs(left), Math.abs(right));
        if (maxValue > 1) {
            left /= maxValue;
            right /= maxValue;
        }
        
        state.leftSpeed = Math.round(left * MOTOR_MAX);
        state.rightSpeed = Math.round(right * MOTOR_MAX);
        
        sendDriveCommand();
    } else if (!keys.has('w') && !keys.has('s') && !keys.has('a') && !keys.has('d')) {
        // Stop if no drive keys pressed
        if (state.leftSpeed !== 0 || state.rightSpeed !== 0) {
            state.leftSpeed = 0;
            state.rightSpeed = 0;
            sendDriveCommand();
        }
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
        state.leftSpeed = 0;
        state.rightSpeed = 0;
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

let lastDriveCommand = { left: 0, right: 0 };
let lastGimbalCommand = { pitch: 90, yaw: 90 };
let lastLightCommand = 0;

function sendDriveCommand() {
    // Only send if values changed
    if (lastDriveCommand.left !== state.leftSpeed || lastDriveCommand.right !== state.rightSpeed) {
        socket.emit('drive', {
            left: state.leftSpeed,
            right: state.rightSpeed
        });
        lastDriveCommand = { left: state.leftSpeed, right: state.rightSpeed };
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
    // Motor values
    document.getElementById('leftMotor').textContent = data.left_speed || state.leftSpeed;
    document.getElementById('rightMotor').textContent = data.right_speed || state.rightSpeed;
    
    // Gimbal
    document.getElementById('pitchDisplay').textContent = data.pitch || state.pitch;
    document.getElementById('yawDisplay').textContent = data.yaw || state.yaw;
    
    // Light
    if (data.light !== undefined) {
        state.light = data.light;
        updateLightUI();
    }
    
    // Speed (average of both motors)
    const avgSpeed = Math.abs((data.left_speed || state.leftSpeed) + (data.right_speed || state.rightSpeed)) / 2;
    document.getElementById('speedDisplay').textContent = Math.round(avgSpeed);
    
    // Heading (based on differential)
    const diff = (data.right_speed || state.rightSpeed) - (data.left_speed || state.leftSpeed);
    const heading = Math.round((diff / MOTOR_MAX) * 45); // ±45 degrees
    document.getElementById('headingDisplay').textContent = heading > 0 ? `+${heading}°` : `${heading}°`;
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