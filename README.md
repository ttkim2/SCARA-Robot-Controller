Project Title: 3-DOF SCARA Robotic Arm & Kinematics

Status: Active (Firmware & Electronics Verified)

Overview: A custom-designed SCARA robot driven by an Arduino Mega. Unlike standard XY plotters, this project required implementing non-linear kinematic equations to translate G-code coordinates into joint angles.

Key Achievements:

Firmware: Modified Marlin 2.0 to support MP_SCARA kinematics, tuning acceleration for high-inertia rotational arms.

Simulation: Built a "Digital Twin" in Python (scara_visualizer.py) using NumPy/Matplotlib to optimize arm lengths for A4 paper coverage.

Electronics: Configured TMC2209 drivers in standalone mode for silent operation, verifying torque and wiring phases via a custom C++ test bench.
