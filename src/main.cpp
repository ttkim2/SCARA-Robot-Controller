#include <Arduino.h>
#include <AccelStepper.h>
#include <MultiStepper.h>

/**
 * SCARA Integrated Build - v28.0 (MultiStepper + Velocity Blend)
 *
 * Drawing motion works in two phases:
 *
 *   1. COORDINATED (MultiStepper): both joints always arrive simultaneously,
 *      so the end-effector traces a straight line in Cartesian space between
 *      every pair of waypoints.
 *
 *   2. BLEND: when within BLEND_STEPS of the current waypoint target and the
 *      next waypoint is already queued, coordinated.moveTo() is called with
 *      the new target BEFORE the robot fully stops. MultiStepper recalculates
 *      the proportional speeds instantly and motion continues without
 *      decelerating to zero — eliminating the stop-start jerk at each waypoint.
 *
 * Why this combination:
 *   - Without MultiStepper: joints arrive at different times → curved path +
 *     visible jerk when faster joint hits its target while slower one continues.
 *   - Without blend: MultiStepper stops at each waypoint → one jerk per step.
 *   - Together: straight lines AND no stops between waypoints.
 */

// --- FUNCTION PROTOTYPES ---
void updatePhysics();
void updatePhysicsIfNeeded();
void homeAllWithRamp();
bool isLimitHit(int pin);

// --- PIN CONFIGURATION ---
const int Z_STEP_PIN = 46;
const int Z_DIR_PIN = 48;
const int Z_ENABLE_PIN = 62;
const int SH_STEP_PIN = 36;
const int SH_DIR_PIN = 34;
#define SH_DIR_INVERT true
const int SH_ENABLE_PIN = 30;
const int EL_STEP_PIN = 60;  const int EL_DIR_PIN = 61; const int EL_ENABLE_PIN = 56;
const int Z_MIN_PIN = 18;    const int X_MIN_PIN = 3;   const int EL_MIN_PIN = 14;

// --- PHYSICS CONSTANTS ---
const float I1 = 0.02428;
const float I2 = 0.0035;
const float M2 = 0.35;
const float L1 = 0.220;
const float R2 = 0.116;
const float TORQUE_LIMIT = 0.25;

// --- STEPPER CONSTANTS ---
const float SH_STEPS_PER_DEG = 177.77;
const float EL_STEPS_PER_DEG = 142.222;
const float Z_STEPS_PER_MM = 400.0;

// --- CALIBRATED & SAFETY CONSTANTS ---
const float Z_HOVER_POS       = 137.0;   // full safe height — used for positioning moves
const float Z_NEAR_HOVER_POS  = 145.0;   // 8 mm below full hover — for fast inter-stroke lifts
                                          // still clears the surface but travels 25 mm
                                          // instead of 33 mm, saving ~1.3 s each direction
const float Z_PRESS_POS       = 165;
const float EL_HOME_OFFSET = 158.0;
const float MAX_ROTATION = 330.0;

// --- Z-AXIS SPEED CONSTANTS ---
// Z_HOME_SPEED  : used during homeAllWithRamp() — runSpeed() has NO accel ramp,
//                 so the motor starts at full speed instantly.  Must be kept well
//                 below the motor's stall speed (typically ≤ 1800 steps/s).
// Z_LIFT_SPEED  : used for 'U' (pen up) — moveTo() uses the accel ramp so a
//                 higher speed is safe.  Inter-stroke pen lifts use this.
// Z_PRESS_SPEED : used for 'W' (pen down) — slow approach prevents the pen tip
//                 from skidding on the surface when it makes contact.
const float Z_HOME_SPEED  = 1800.0;
const float Z_LIFT_SPEED  = 2400.0;
const float Z_PRESS_SPEED = 1050.0;

// --- BLEND THRESHOLD ---
// Steps before the current waypoint target at which we redirect the
// coordinated move to the next queued waypoint.
// MultiStepper has NO deceleration ramp (constant speed), so even a small
// value works — the robot changes direction within BLEND_STEPS of each waypoint.
// At DRAW_SPEED and typical step counts, BLEND_STEPS=20 ≈ 0.5 mm at end-effector.
// Increase toward 60 if lines are still wavy; decrease toward 5 if corners are too rounded.
const long BLEND_STEPS = 40;

AccelStepper stepperSH(1, SH_STEP_PIN, SH_DIR_PIN);
AccelStepper stepperEL(1, EL_STEP_PIN, EL_DIR_PIN);
AccelStepper stepperZ(1, Z_STEP_PIN, Z_DIR_PIN);
MultiStepper coordinated;

bool isSweeping = false;
float currentVSpeed = 0;
bool coordinatedMove = false;

// 1-slot lookahead queue
bool hasQueuedMove = false;
float queuedSH = 0, queuedEL = 0;

// Global (not static local) so serial handlers can reset it when new motion starts
bool moveDoneReported = false;

// --- JERK-LIMITED SPEED RAMP ---
// targetDrawSpeed  : speed requested by the last 'S' command.
// currentDrawSpeed : speed actually applied to steppers — ramped toward
//                    targetDrawSpeed at JERK_LIMIT_PER_SEC steps/s².
//                    Prevents instantaneous large speed jumps at waypoint
//                    boundaries that MultiStepper would otherwise apply instantly.
// Initialised to 2000 to match setup() setMaxSpeed calls.
// JERK_LIMIT_PER_SEC = 8000 → ramp 0 → 800 steps/s takes ~100 ms.
const float JERK_LIMIT_PER_SEC = 8000.0;
float targetDrawSpeed  = 2000.0;
float currentDrawSpeed = 2000.0;
unsigned long lastJerkUs = 0;

void setup() {
  Serial.begin(115200);
  pinMode(Z_MIN_PIN, INPUT_PULLUP);
  pinMode(X_MIN_PIN, INPUT_PULLUP);
  pinMode(EL_MIN_PIN, INPUT_PULLUP);

  pinMode(Z_ENABLE_PIN, OUTPUT); digitalWrite(Z_ENABLE_PIN, LOW);
  pinMode(SH_ENABLE_PIN, OUTPUT); digitalWrite(SH_ENABLE_PIN, LOW);
  pinMode(EL_ENABLE_PIN, OUTPUT); digitalWrite(EL_ENABLE_PIN, LOW);

  stepperSH.setMaxSpeed(2000);
  stepperSH.setAcceleration(4500);
  stepperEL.setMaxSpeed(2000);
  stepperEL.setAcceleration(3000);
  stepperZ.setMaxSpeed(Z_LIFT_SPEED);
  stepperZ.setAcceleration(2000);

  coordinated.addStepper(stepperSH);
  coordinated.addStepper(stepperEL);

  lastJerkUs = micros();   // initialise jerk-ramp timer
  updatePhysics();
  Serial.println("SYSTEM_READY");
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();

    if (cmd == 'V') {
      currentVSpeed = Serial.parseFloat();
      isSweeping = true;
    }
    else if (cmd == 'X') { isSweeping = false; stepperSH.stop(); }
    else if (cmd == 'H') homeAllWithRamp();
    else if (cmd == 'U') {
      stepperZ.setMaxSpeed(Z_LIFT_SPEED);   // fast lift — gets pen clear quickly
      stepperZ.moveTo(Z_HOVER_POS * Z_STEPS_PER_MM);
      moveDoneReported = false;
    }
    else if (cmd == 'N') {
      // Near-hover lift — 8 mm below full hover, 25 mm travel vs 33 mm.
      // Use between strokes to save ~1.3 s per lift without sacrificing safety.
      stepperZ.setMaxSpeed(Z_LIFT_SPEED);
      stepperZ.moveTo(Z_NEAR_HOVER_POS * Z_STEPS_PER_MM);
      moveDoneReported = false;
    }
    else if (cmd == 'W') {
      stepperZ.setMaxSpeed(Z_PRESS_SPEED);  // slow approach — prevents skid on contact
      stepperZ.moveTo(Z_PRESS_POS * Z_STEPS_PER_MM);
      moveDoneReported = false;
    }
    else if (cmd == 'P') {
      float shDeg = stepperSH.currentPosition() / SH_STEPS_PER_DEG;
      float elDeg = stepperEL.currentPosition() / EL_STEPS_PER_DEG;
      Serial.print("POS SH:");
      Serial.print(shDeg, 3);
      Serial.print(" EL:");
      Serial.println(elDeg, 3);
    }
    else if (cmd == 'G') {
      float syncAngle = Serial.parseFloat();
      stepperSH.setCurrentPosition(syncAngle * SH_STEPS_PER_DEG);
    }
    else if (cmd == 'S') {
      /* OLD — direct speed assignment (uncomment to revert jerk-limited ramp):
      float spd = Serial.parseFloat();
      stepperSH.setMaxSpeed(spd);
      stepperEL.setMaxSpeed(spd);
      */
      // NEW: store as target; loop() ramps currentDrawSpeed toward it gradually
      targetDrawSpeed = Serial.parseFloat();
    }
    else if (cmd == 'M') {
      isSweeping = false;
      float tSH = Serial.parseFloat();
      float tEL = Serial.parseFloat();

      // 330 DEGREE SAFETY GUARD
      tSH = constrain(tSH, 0.0, MAX_ROTATION);
      tEL = constrain(tEL, 0.0, MAX_ROTATION);

      if (coordinatedMove) {
        // Queue for blend dispatch
        queuedSH = tSH;
        queuedEL = tEL;
        hasQueuedMove = true;
        Serial.println("QUEUED");
      } else {
        long positions[2] = {
          (long)(tSH * SH_STEPS_PER_DEG),
          (long)(tEL * EL_STEPS_PER_DEG)
        };
        coordinated.moveTo(positions);
        coordinatedMove = true;
        moveDoneReported = false;
      }
    }
  }

  if (isSweeping) {
    // 330 DEGREE SWEEP SAFETY
    float currentDeg = stepperSH.currentPosition() / SH_STEPS_PER_DEG;
    if ((currentVSpeed > 0 && currentDeg >= MAX_ROTATION) ||
        (currentVSpeed < 0 && isLimitHit(X_MIN_PIN))) {
      stepperSH.stop();
      isSweeping = false;
    } else {
      stepperSH.setSpeed(currentVSpeed);
      stepperSH.runSpeed();
    }
  } else {
    // JERK-LIMITED SPEED RAMP -----------------------------------------------
    // Blend currentDrawSpeed toward targetDrawSpeed at JERK_LIMIT_PER_SEC
    // (steps/s per second).  MultiStepper uses getMaxSpeed() on every call to
    // run(), so updating setMaxSpeed() here takes effect immediately — the ramp
    // is applied once per loop iteration, which is fast enough to feel smooth.
    {
      unsigned long nowUs = micros();
      float dtSec = (nowUs - lastJerkUs) * 1e-6f;
      if (dtSec > 0.01f) dtSec = 0.01f;   // cap at 10 ms (first call / overflow)
      lastJerkUs = nowUs;
      float maxDelta = JERK_LIMIT_PER_SEC * dtSec;
      if (fabsf(targetDrawSpeed - currentDrawSpeed) <= maxDelta) {
        currentDrawSpeed = targetDrawSpeed;
      } else {
        currentDrawSpeed += (targetDrawSpeed > currentDrawSpeed) ? maxDelta : -maxDelta;
      }
      // Apply immediately; updatePhysics() may further scale SH max speed.
      stepperSH.setMaxSpeed(currentDrawSpeed);
      stepperEL.setMaxSpeed(currentDrawSpeed);
    }
    // -----------------------------------------------------------------------
    updatePhysicsIfNeeded();

    if (coordinatedMove) {
      // BLEND CHECK: when within BLEND_STEPS of the current target and the next
      // waypoint is queued, redirect coordinated motion NOW before fully stopping.
      // coordinated.moveTo() recalculates the speed ratios for the new target
      // instantly — MultiStepper continues at constant speed without ever stopping.
      // This is the key to eliminating stop-start jerk on every waypoint.
      if (hasQueuedMove
          && abs(stepperSH.distanceToGo()) <= BLEND_STEPS
          && abs(stepperEL.distanceToGo()) <= BLEND_STEPS) {
        long positions[2] = {
          (long)(queuedSH * SH_STEPS_PER_DEG),
          (long)(queuedEL * EL_STEPS_PER_DEG)
        };
        coordinated.moveTo(positions);
        hasQueuedMove = false;
        Serial.println("MOVE_DONE");  // previous waypoint reached (within blend threshold)
        // coordinatedMove stays true — motion is ongoing toward new target
      }

      if (!coordinated.run()) {
        coordinatedMove = false;  // reached final target of this stroke
      }
    } else {
      // Only step SH/EL when Z is stationary.
      // This prevents the arm from moving (even a single step) while the
      // pen is going up or down, which would drag the pen across the surface.
      if (stepperZ.distanceToGo() == 0) {
        stepperSH.run();
        stepperEL.run();
      }
    }
    stepperZ.run();

    // Fire MOVE_DONE when ALL axes are truly at rest (no blend pending).
    // Handles: final waypoint of stroke, pen up/down Z settle, travel moves.
    if (!coordinatedMove
        && stepperSH.distanceToGo() == 0
        && stepperEL.distanceToGo() == 0
        && stepperZ.distanceToGo() == 0) {
      if (!moveDoneReported) {
        if (hasQueuedMove) {
          // Edge case: queued move arrived after a very fast or zero-distance move
          long positions[2] = {
            (long)(queuedSH * SH_STEPS_PER_DEG),
            (long)(queuedEL * EL_STEPS_PER_DEG)
          };
          coordinated.moveTo(positions);
          coordinatedMove = true;
          hasQueuedMove = false;
        }
        Serial.println("MOVE_DONE");
        moveDoneReported = true;
      }
    } else {
      moveDoneReported = false;
    }
  }
}

void updatePhysics() {
  float theta2 = (stepperEL.currentPosition() / EL_STEPS_PER_DEG) * DEG_TO_RAD;
  float effective_I = I1 + I2 + M2 * (sq(L1) + sq(R2) + 2.0 * L1 * R2 * cos(theta2));
  float maxAlpha = TORQUE_LIMIT / effective_I;
  float stepsPerRad = (SH_STEPS_PER_DEG * 360.0) / (2.0 * PI);

  /* OLD — acceleration only (uncomment to revert physics setMaxSpeed scaling):
  stepperSH.setAcceleration(maxAlpha * stepsPerRad);
  stepperEL.setAcceleration(3000);
  */

  // Physics-adaptive acceleration (torque budget / effective inertia)
  stepperSH.setAcceleration(maxAlpha * stepsPerRad);
  stepperEL.setAcceleration(3000);

  // Physics-adaptive max speed: only applied during drawing (targetDrawSpeed ≤ 1500).
  // During travel (targetDrawSpeed > 1500) the arm uses full currentDrawSpeed so
  // positioning moves are fast.  During drawing, scale by sqrt(I_MIN / I_eff) to
  // keep required deceleration torque within TORQUE_LIMIT at each waypoint.
  // Both steppers must share the same maxSpeed — MultiStepper computes each axis
  // speed as  maxSpeed × distanceToGo / longestDistance,  so mismatched maxSpeeds
  // break proportionality and cause one joint to arrive before the other.
  const float I_MIN = I1 + I2 + M2 * sq(L1 - R2);   // min inertia (arm fully folded)
  float physMaxSpd;
  if (targetDrawSpeed <= 1500.0f) {
    // DRAW mode: reduce max speed by inertia ratio (speedScale ≈ 0.68–1.0)
    float speedScale = sqrtf(I_MIN / effective_I);
    physMaxSpd = currentDrawSpeed * speedScale;
    physMaxSpd = constrain(physMaxSpd, currentDrawSpeed * 0.4f, currentDrawSpeed);
  } else {
    // TRAVEL mode: full speed — no physics cap on positioning moves
    physMaxSpd = currentDrawSpeed;
  }
  stepperSH.setMaxSpeed(physMaxSpd);
  stepperEL.setMaxSpeed(physMaxSpd);   // must match SH for simultaneous arrival
}

void updatePhysicsIfNeeded() {
  static float lastElDeg = -999.0;
  float elDeg = stepperEL.currentPosition() / EL_STEPS_PER_DEG;
  if (abs(elDeg - lastElDeg) > 1.0) {
    updatePhysics();
    lastElDeg = elDeg;
  }
}

void homeAllWithRamp() {
  isSweeping = false;
  coordinatedMove = false;
  hasQueuedMove = false;

  // Reset maxSpeeds so setSpeed() is not clamped by whatever drawing left them at.
  stepperSH.setMaxSpeed(3500);
  stepperEL.setMaxSpeed(3500);
  // Use Z_HOME_SPEED for homing — runSpeed() has no accel ramp so the motor
  // must start at a safe constant speed to avoid stalling.
  stepperZ.setMaxSpeed(Z_HOME_SPEED);

  stepperZ.setSpeed(-Z_HOME_SPEED);
  while (!isLimitHit(Z_MIN_PIN)) { stepperZ.runSpeed(); }
  stepperZ.stop();
  stepperZ.setCurrentPosition(0);
  delay(500);

  stepperSH.setSpeed(-3500);
  stepperEL.setSpeed(-3500);
  bool shH = false, elH = false;
  while (!shH || !elH) {
    if (!shH) { if (isLimitHit(X_MIN_PIN)) { stepperSH.stop(); shH = true; } else stepperSH.runSpeed(); }
    if (!elH) { if (isLimitHit(EL_MIN_PIN)) { stepperEL.stop(); elH = true; } else stepperEL.runSpeed(); }
  }
  stepperSH.setCurrentPosition(0);
  stepperEL.setCurrentPosition(0);
  moveDoneReported = false;

  Serial.println("CALIBRATED");
}

bool isLimitHit(int pin) {
  if (digitalRead(pin) == HIGH) {
    delay(20);
    if (digitalRead(pin) == HIGH) return true;
  }
  return false;
}