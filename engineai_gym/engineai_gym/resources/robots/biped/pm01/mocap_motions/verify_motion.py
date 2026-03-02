"""
Verify exported motion JSON file matches PM01 data format.

Usage:
    python verify_motion.py <path_to_json>

Example:
    python verify_motion.py dance1.json
"""

import json
import sys
import numpy as np

EXPECTED_VALUES_PER_FRAME = 73

COMPONENTS = [
    ("root_pos",        0,  3),
    ("root_rot",        3,  7),
    ("dof_pos",         7,  31),
    ("foot_pos_local",  31, 37),
    ("root_lin_vel",    37, 40),
    ("root_ang_vel",    40, 43),
    ("dof_vel",         43, 67),
    ("foot_vel_local",  67, 73),
]

PM01_JOINT_NAMES = [
    "j00_hip_pitch_l", "j01_hip_roll_l", "j02_hip_yaw_l",
    "j03_knee_pitch_l", "j04_ankle_pitch_l", "j05_ankle_roll_l",
    "j06_hip_pitch_r", "j07_hip_roll_r", "j08_hip_yaw_r",
    "j09_knee_pitch_r", "j10_ankle_pitch_r", "j11_ankle_roll_r",
    "j12_waist_yaw",
    "j13_shoulder_pitch_l", "j14_shoulder_roll_l", "j15_shoulder_yaw_l",
    "j16_elbow_pitch_l", "j17_elbow_yaw_l",
    "j18_shoulder_pitch_r", "j19_shoulder_roll_r", "j20_shoulder_yaw_r",
    "j21_elbow_pitch_r", "j22_elbow_yaw_r",
    "j23_head_yaw",
]

PM01_JOINT_LIMITS = [
    (-3.141, 2.443), (-0.436, 2.094), (-1.57, 4.014),
    (-0.3491, 2.3911), (-0.6807, 0.7243), (-0.2618, 0.2618),
    (-3.141, 2.443), (-2.094, 0.436), (-4.014, 1.57),
    (-0.3491, 2.3911), (-0.6807, 0.7243), (-0.2618, 0.2618),
    (-4.014, 1.57),
    (-2.9671, 2.7925), (-0.6108, 2.3562), (-2.618, 2.618),
    (-2.1948, 0.7374), (-2.618, 2.618),
    (-2.9671, 2.7925), (-2.3562, 0.6108), (-2.618, 2.618),
    (-2.1948, 0.7374), (-2.618, 2.618),
    (-0.6109, 0.6109),
]


def verify(filepath):
    with open(filepath) as f:
        data = json.load(f)

    print(f"File: {filepath}")
    print(f"LoopMode: {data.get('LoopMode', 'N/A')}")
    print(f"FrameDuration: {data.get('FrameDuration', 'N/A')}")
    print(f"MotionWeight: {data.get('MotionWeight', 'N/A')}")

    frames = np.array(data["Frames"])
    num_frames, num_values = frames.shape
    duration = num_frames * data["FrameDuration"]

    print(f"\nFrames: {num_frames}")
    print(f"Values per frame: {num_values}")
    print(f"Duration: {duration:.2f}s")

    # Check frame size
    if num_values != EXPECTED_VALUES_PER_FRAME:
        print(f"\nERROR: Expected {EXPECTED_VALUES_PER_FRAME} values per frame, got {num_values}")
        return False

    print(f"\n{'Component':<20} {'Indices':<10} {'Min':>10} {'Max':>10} {'Mean':>10}")
    print("-" * 65)
    for name, start, end in COMPONENTS:
        vals = frames[:, start:end]
        print(f"{name:<20} [{start}:{end}]    {vals.min():>10.4f} {vals.max():>10.4f} {vals.mean():>10.4f}")

    # Check quaternion normalization
    quats = frames[:, 3:7]
    quat_norms = np.linalg.norm(quats, axis=1)
    print(f"\nQuaternion norm range: [{quat_norms.min():.4f}, {quat_norms.max():.4f}] (should be ~1.0)")

    # Check root height
    root_z = frames[:, 2]
    print(f"Root Z range: [{root_z.min():.4f}, {root_z.max():.4f}] (PM01 standing ~0.9m)")

    # Check joint limits
    dof_pos = frames[:, 7:31]
    violations = 0
    for i, (jname, (lower, upper)) in enumerate(zip(PM01_JOINT_NAMES, PM01_JOINT_LIMITS)):
        joint_vals = dof_pos[:, i]
        if joint_vals.min() < lower - 0.01 or joint_vals.max() > upper + 0.01:
            violations += 1
            print(f"  WARNING: {jname} out of limits [{lower:.3f}, {upper:.3f}], "
                  f"got [{joint_vals.min():.3f}, {joint_vals.max():.3f}]")

    if violations == 0:
        print("\nAll joint angles within limits.")

    # Check for all-zero joints (may indicate mapping issues)
    zero_joints = []
    for i, jname in enumerate(PM01_JOINT_NAMES):
        if np.allclose(dof_pos[:, i], 0.0, atol=1e-6):
            zero_joints.append(jname)
    if zero_joints:
        print(f"\nWARNING: These joints are always zero (may need better mapping):")
        for jname in zero_joints:
            print(f"  - {jname}")

    print("\nVerification complete!")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_motion.py <path_to_json>")
        sys.exit(1)
    verify(sys.argv[1])
