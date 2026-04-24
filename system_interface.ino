//Arduino Code for the Arduino Mega connected to the Raspberry Pi 4 via USB inside the Rover.

#include <AFMotor.h> 
#include <Servo.h>

// Define motors on the shield ports
AF_DCMotor motorLeft(2);   // M2 port
AF_DCMotor motorRight(3);  // M3 port
AF_DCMotor lightMotor(1);  // M1 port (using motor port for light control)

// Servos
Servo pitchServo;
Servo yawServo;

void setup() {
  Serial.begin(115200);

  // Attach servos (Servo 1 = Pin 10, Servo 2 = Pin 9 on this shield)
  pitchServo.attach(10);
  yawServo.attach(9);

  // Initial positions
  pitchServo.write(90);
  yawServo.write(90);

  Serial.println("Arduino Motor Controller Ready");
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    data.trim();

    // Parse comma-separated values (left,right,pitch,yaw,light)
    int v[5];
    int lastComma = -1;
    for (int i = 0; i < 5; i++) {
      int commaPos = data.indexOf(',', lastComma + 1);
      v[i] = (i < 4) ? data.substring(lastComma + 1, commaPos).toInt() : data.substring(lastComma + 1).toInt();
      lastComma = commaPos;
    }

    // 1. Control Left Motor (M2)
    controlAFMotor(motorLeft, v[0]);

    // 2. Control Right Motor (M3)
    controlAFMotor(motorRight, v[1]);

    // 3. Control Servos
    pitchServo.write(constrain(v[2], 0, 180));
    yawServo.write(constrain(v[3], 0, 180));

    // 4. Control Light (M1) - Simple Forward PWM
    lightMotor.setSpeed(constrain(v[4], 0, 255));
    lightMotor.run(FORWARD);

    Serial.println("OK");
  }
}

// Helper to handle Forward/Backward/Release for AFMotor
void controlAFMotor(AF_DCMotor &m, int speed) {
  if (speed > 0) {
    m.setSpeed(speed);
    m.run(FORWARD);
  } else if (speed < 0) {
    m.setSpeed(abs(speed));
    m.run(BACKWARD);
  } else {
    m.run(RELEASE);
  }
}