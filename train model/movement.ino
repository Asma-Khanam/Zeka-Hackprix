// BCI_EEG_Servo_Control.ino
// Arduino code for BioAmp EEG reading + Servo control
// BioAmp EXG Pill connected to Arduino analog pins

#include <Servo.h>

// Create servo objects
Servo servoX;  // Horizontal servo
Servo servoY;  // Vertical servo

// Pin definitions
const int EEG_CH1_PIN = A0;      // BioAmp Channel 1 (analog pin)
const int EEG_CH2_PIN = A1;      // BioAmp Channel 2 (analog pin)
const int SERVO_X_PIN = 9;       // Servo X (PWM pin)
const int SERVO_Y_PIN = 10;      // Servo Y (PWM pin)
const int LED_PIN = 13;          // Status LED

// EEG sampling parameters
const int SAMPLING_RATE = 250;   // Hz (samples per second)
const int SAMPLE_INTERVAL = 1000 / SAMPLING_RATE;  // 4ms for 250Hz
unsigned long lastSampleTime = 0;
unsigned long sampleCount = 0;

// Servo parameters
int currentX = 90;  // Center position
int currentY = 90;  // Center position
const int MOVE_STEP = 30;
const int MIN_ANGLE = 0;
const int MAX_ANGLE = 180;

// Communication parameters
String lastCommand = "NONE";
bool streamingMode = false;
bool commandMode = true;

// EEG data storage
const int BUFFER_SIZE = 100;
int eegBuffer1[BUFFER_SIZE];
int eegBuffer2[BUFFER_SIZE];
int bufferIndex = 0;

void setup() {
  // Initialize serial communication (high baud rate for EEG data)
  Serial.begin(115200);
  
  // Setup analog pins for EEG input
  pinMode(EEG_CH1_PIN, INPUT);
  pinMode(EEG_CH2_PIN, INPUT);
  
  // Setup servo pins
  servoX.attach(SERVO_X_PIN);
  servoY.attach(SERVO_Y_PIN);
  
  // Setup LED
  pinMode(LED_PIN, OUTPUT);
  
  // Center servos
  centerServos();
  
  // Startup sequence
  startupSequence();
  
  // Ready message
  Serial.println("🧠 Arduino BioAmp BCI System Ready!");
  Serial.println("Commands:");
  Serial.println("  STREAM_ON  - Start EEG streaming");
  Serial.println("  STREAM_OFF - Stop EEG streaming");
  Serial.println("  LEFT/RIGHT/CENTER - Manual servo control");
  Serial.println("  STATUS - Show system status");
  Serial.println("Waiting for commands...");
}

void loop() {
  // Handle EEG sampling
  if (streamingMode && (millis() - lastSampleTime >= SAMPLE_INTERVAL)) {
    readAndSendEEG();
    lastSampleTime = millis();
  }
  
  // Handle incoming commands
  if (Serial.available() > 0) {
    processSerialCommand();
  }
  
  // Small delay for stability
  delayMicroseconds(100);
}

void readAndSendEEG() {
  // Read both EEG channels
  int ch1 = analogRead(EEG_CH1_PIN);
  int ch2 = analogRead(EEG_CH2_PIN);
  
  // Store in buffer (for potential local processing)
  eegBuffer1[bufferIndex] = ch1;
  eegBuffer2[bufferIndex] = ch2;
  bufferIndex = (bufferIndex + 1) % BUFFER_SIZE;
  
  // Send data to computer in CSV format
  Serial.print(millis());          // Timestamp
  Serial.print(",");
  Serial.print(ch1);               // Channel 1 (0-1023)
  Serial.print(",");
  Serial.print(ch2);               // Channel 2 (0-1023)
  Serial.println();                // End of line
  
  sampleCount++;
  
  // Blink LED to show activity
  if (sampleCount % 125 == 0) {  // Every 0.5 seconds at 250Hz
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }
}

void processSerialCommand() {
  String command = Serial.readStringUntil('\n');
  command.trim();
  command.toUpperCase();
  
  if (command.length() > 0) {
    // Turn on LED for command processing
    digitalWrite(LED_PIN, HIGH);
    
    Serial.print("📡 Command received: ");
    Serial.println(command);
    
    if (command == "STREAM_ON") {
      streamingMode = true;
      commandMode = false;
      lastSampleTime = millis();
      sampleCount = 0;
      Serial.println("🟢 EEG streaming started");
      
    } else if (command == "STREAM_OFF") {
      streamingMode = false;
      commandMode = true;
      digitalWrite(LED_PIN, LOW);
      Serial.println("🔴 EEG streaming stopped");
      
    } else if (command == "LEFT") {
      if (commandMode) {
        moveServoX(-MOVE_STEP);
        Serial.println("⬅️ Moving LEFT");
      }
      
    } else if (command == "RIGHT") {
      if (commandMode) {
        moveServoX(MOVE_STEP);
        Serial.println("➡️ Moving RIGHT");
      }
      
    } else if (command == "UP") {
      if (commandMode) {
        moveServoY(-MOVE_STEP);
        Serial.println("⬆️ Moving UP");
      }
      
    } else if (command == "DOWN") {
      if (commandMode) {
        moveServoY(MOVE_STEP);
        Serial.println("⬇️ Moving DOWN");
      }
      
    } else if (command == "CENTER") {
      if (commandMode) {
        centerServos();
        Serial.println("🎯 Servos centered");
      }
      
    } else if (command == "STATUS") {
      printStatus();
      
    } else if (command == "CALIBRATE") {
      calibrateEEG();
      
    } else if (command == "TEST") {
      testServos();
      
    } else {
      Serial.print("❓ Unknown command: ");
      Serial.println(command);
    }
    
    // Store last command
    lastCommand = command;
    
    // Turn off LED
    delay(50);
    digitalWrite(LED_PIN, LOW);
  }
}

void moveServoX(int deltaX) {
  int newX = currentX + deltaX;
  newX = constrain(newX, MIN_ANGLE, MAX_ANGLE);
  
  if (newX != currentX) {
    currentX = newX;
    servoX.write(currentX);
    delay(200);
    
    Serial.print("🔄 X moved to: ");
    Serial.print(currentX);
    Serial.println("°");
  } else {
    Serial.println("⚠️ X at limit");
  }
}

void moveServoY(int deltaY) {
  int newY = currentY + deltaY;
  newY = constrain(newY, MIN_ANGLE, MAX_ANGLE);
  
  if (newY != currentY) {
    currentY = newY;
    servoY.write(currentY);
    delay(200);
    
    Serial.print("🔄 Y moved to: ");
    Serial.print(currentY);
    Serial.println("°");
  } else {
    Serial.println("⚠️ Y at limit");
  }
}

void centerServos() {
  currentX = 90;
  currentY = 90;
  servoX.write(currentX);
  servoY.write(currentY);
  delay(500);
  Serial.println("🎯 Servos centered (90°, 90°)");
}

void startupSequence() {
  // Flash LED
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(200);
    digitalWrite(LED_PIN, LOW);
    delay(200);
  }
  
  // Test servos
  Serial.println("🔧 Testing servos...");
  servoX.write(60);
  delay(300);
  servoX.write(120);
  delay(300);
  servoX.write(90);
  
  servoY.write(60);
  delay(300);
  servoY.write(120);
  delay(300);
  servoY.write(90);
  
  Serial.println("✅ Servo test complete");
}

void printStatus() {
  Serial.println("📊 === SYSTEM STATUS ===");
  Serial.print("EEG Streaming: ");
  Serial.println(streamingMode ? "ON" : "OFF");
  Serial.print("Sample count: ");
  Serial.println(sampleCount);
  Serial.print("Servo X: ");
  Serial.print(currentX);
  Serial.println("°");
  Serial.print("Servo Y: ");
  Serial.print(currentY);
  Serial.println("°");
  Serial.print("Last command: ");
  Serial.println(lastCommand);
  Serial.print("Uptime: ");
  Serial.print(millis() / 1000);
  Serial.println(" seconds");
  
  // Show recent EEG readings
  Serial.println("Recent EEG readings:");
  for (int i = 0; i < 5; i++) {
    int idx = (bufferIndex - 5 + i + BUFFER_SIZE) % BUFFER_SIZE;
    Serial.print("  CH1: ");
    Serial.print(eegBuffer1[idx]);
    Serial.print(", CH2: ");
    Serial.println(eegBuffer2[idx]);
  }
  Serial.println("========================");
}

void calibrateEEG() {
  Serial.println("🔧 EEG Calibration - Keep still for 5 seconds");
  
  long sum1 = 0, sum2 = 0;
  int samples = 0;
  
  for (int i = 0; i < 1250; i++) {  // 5 seconds at 250Hz
    int ch1 = analogRead(EEG_CH1_PIN);
    int ch2 = analogRead(EEG_CH2_PIN);
    
    sum1 += ch1;
    sum2 += ch2;
    samples++;
    
    delay(4);  // 250Hz sampling
    
    if (i % 250 == 0) {
      Serial.print(".");
    }
  }
  
  int baseline1 = sum1 / samples;
  int baseline2 = sum2 / samples;
  
  Serial.println();
  Serial.print("✅ Calibration complete - CH1 baseline: ");
  Serial.print(baseline1);
  Serial.print(", CH2 baseline: ");
  Serial.println(baseline2);
}

void testServos() {
  Serial.println("🔧 Testing servo movements...");
  
  // Test sequence
  String commands[] = {"LEFT", "RIGHT", "CENTER", "UP", "DOWN", "CENTER"};
  
  for (int i = 0; i < 6; i++) {
    Serial.print("Testing: ");
    Serial.println(commands[i]);
    
    if (commands[i] == "LEFT") moveServoX(-30);
    else if (commands[i] == "RIGHT") moveServoX(30);
    else if (commands[i] == "UP") moveServoY(-30);
    else if (commands[i] == "DOWN") moveServoY(30);
    else if (commands[i] == "CENTER") centerServos();
    
    delay(1000);
  }
  
  Serial.println("✅ Servo test complete");
}