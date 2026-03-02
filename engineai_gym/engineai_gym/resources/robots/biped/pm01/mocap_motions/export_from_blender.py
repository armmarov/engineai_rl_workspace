"""
Blender Python script to export Mixamo animation as PM01 reference motion JSON.

Usage:
    1. Open Blender
    2. Import Mixamo FBX (File -> Import -> FBX)
    3. Open Scripting tab
    4. Open/paste this script
    5. Update ARMATURE_NAME and OUTPUT_PATH below
    6. Run script

Frame format (73 values per frame):
    [0:3]   root_pos        - Base position (x, y, z)
    [3:7]   root_rot        - Base rotation quaternion (qx, qy, qz, qw)
    [7:31]  dof_pos         - 24 joint angles
    [31:37] foot_pos_local  - 2 feet local positions (left, right)
    [37:40] root_lin_vel    - Base linear velocity
    [40:43] root_ang_vel    - Base angular velocity
    [43:67] dof_vel         - 24 joint velocities
    [67:73] foot_vel_local  - 2 feet local velocities
"""

import bpy
import json
import math
from mathutils import Vector, Quaternion, Matrix

# ============================================================
# CONFIGURATION - UPDATE THESE
# ============================================================
ARMATURE_NAME = "Armature"  # Name of Mixamo armature in Blender Outliner
OUTPUT_PATH = "/tmp/pm01_dance_motion.json"  # Output file path
FPS = 60

# ============================================================
# PM01 JOINT DEFINITIONS (from pm01.urdf)
# ============================================================
PM01_JOINTS = [
    # name,                    axis (from URDF),              lower,   upper
    ("j00_hip_pitch_l",        (0, 0.96593, -0.25882),       -3.141,  2.443),
    ("j01_hip_roll_l",         (1, 0, 0),                    -0.436,  2.094),
    ("j02_hip_yaw_l",          (0, 0, 1),                    -1.57,   4.014),
    ("j03_knee_pitch_l",       (0, 1, 0),                    -0.3491, 2.3911),
    ("j04_ankle_pitch_l",      (0, 1, 0),                    -0.6807, 0.7243),
    ("j05_ankle_roll_l",       (1, 0, 0),                    -0.2618, 0.2618),
    ("j06_hip_pitch_r",        (0, 0.96593, 0.25882),        -3.141,  2.443),
    ("j07_hip_roll_r",         (1, 0, 0),                    -2.094,  0.436),
    ("j08_hip_yaw_r",          (0, 0, 1),                    -4.014,  1.57),
    ("j09_knee_pitch_r",       (0, 1, 0),                    -0.3491, 2.3911),
    ("j10_ankle_pitch_r",      (0, 1, 0),                    -0.6807, 0.7243),
    ("j11_ankle_roll_r",       (1, 0, 0),                    -0.2618, 0.2618),
    ("j12_waist_yaw",          (0, 0, 1),                    -4.014,  1.57),
    ("j13_shoulder_pitch_l",   (0, 0.99803, 0.062791),       -2.9671, 2.7925),
    ("j14_shoulder_roll_l",    (1, 0, 0),                    -0.6108, 2.3562),
    ("j15_shoulder_yaw_l",     (0, -0.062803, 0.99803),      -2.618,  2.618),
    ("j16_elbow_pitch_l",      (0.0027243, 0.99802, 0.062803), -2.1948, 0.7374),
    ("j17_elbow_yaw_l",        (-0.21479, -0.061921, 0.9747), -2.618, 2.618),
    ("j18_shoulder_pitch_r",   (0, 0.99803, -0.062791),      -2.9671, 2.7925),
    ("j19_shoulder_roll_r",    (1, 0, 0),                    -2.3562, 0.6108),
    ("j20_shoulder_yaw_r",     (0, 0.062791, 0.99803),       -2.618,  2.618),
    ("j21_elbow_pitch_r",      (0.0027243, 0.99802, -0.06279), -2.1948, 0.7374),
    ("j22_elbow_yaw_r",        (-0.21479, 0.061909, 0.9747), -2.618,  2.618),
    ("j23_head_yaw",           (0, 0, 1),                    -0.6109, 0.6109),
]

# Mixamo bone -> PM01 joints mapping
# One Mixamo bone may control multiple PM01 joints (e.g. hip has pitch, roll, yaw)
MIXAMO_TO_PM01 = {
    "mixamorig:LeftUpLeg":   ["j00_hip_pitch_l", "j01_hip_roll_l", "j02_hip_yaw_l"],
    "mixamorig:LeftLeg":     ["j03_knee_pitch_l"],
    "mixamorig:LeftFoot":    ["j04_ankle_pitch_l", "j05_ankle_roll_l"],
    "mixamorig:RightUpLeg":  ["j06_hip_pitch_r", "j07_hip_roll_r", "j08_hip_yaw_r"],
    "mixamorig:RightLeg":    ["j09_knee_pitch_r"],
    "mixamorig:RightFoot":   ["j10_ankle_pitch_r", "j11_ankle_roll_r"],
    "mixamorig:Spine":       ["j12_waist_yaw"],
    "mixamorig:LeftArm":     ["j13_shoulder_pitch_l", "j14_shoulder_roll_l", "j15_shoulder_yaw_l"],
    "mixamorig:LeftForeArm": ["j16_elbow_pitch_l", "j17_elbow_yaw_l"],
    "mixamorig:RightArm":    ["j18_shoulder_pitch_r", "j19_shoulder_roll_r", "j20_shoulder_yaw_r"],
    "mixamorig:RightForeArm":["j21_elbow_pitch_r", "j22_elbow_yaw_r"],
    "mixamorig:Head":        ["j23_head_yaw"],
}

# Mixamo foot bones for foot position tracking
FOOT_BONES = ["mixamorig:LeftFoot", "mixamorig:RightFoot"]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

# Build lookup: joint_name -> (axis, lower, upper, index)
JOINT_LOOKUP = {}
for idx, (name, axis, lower, upper) in enumerate(PM01_JOINTS):
    JOINT_LOOKUP[name] = {"axis": Vector(axis).normalized(), "lower": lower, "upper": upper, "index": idx}

# Build reverse lookup: joint_name -> mixamo_bone
JOINT_TO_MIXAMO = {}
for mixamo_bone, pm01_joints in MIXAMO_TO_PM01.items():
    for jname in pm01_joints:
        JOINT_TO_MIXAMO[jname] = mixamo_bone


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def get_bone_rotation_quaternion(bone):
    """Get bone's local rotation as quaternion."""
    if bone.rotation_mode == 'QUATERNION':
        return bone.rotation_quaternion.copy()
    else:
        return bone.rotation_euler.to_quaternion()


def project_rotation_to_axis(bone, joint_axis):
    """
    Project a bone's rotation onto a specific joint axis.
    Returns the angle of rotation around that axis.
    """
    quat = get_bone_rotation_quaternion(bone)

    # For small rotations, we can decompose the quaternion
    # into rotation around the target axis
    # Blender returns (Vector axis, float angle)
    rot_axis, angle = quat.to_axis_angle()

    if rot_axis.length < 1e-6:
        return 0.0

    rot_axis.normalize()

    # Project: how much of the rotation is around our target axis
    projection = rot_axis.dot(joint_axis)
    projected_angle = angle * projection

    return projected_angle


def compute_angular_velocity(quat_prev, quat_curr, dt):
    """Compute angular velocity from two quaternions."""
    # q_diff = q_curr * q_prev^(-1)
    q_diff = quat_curr @ quat_prev.inverted()
    # Blender returns (Vector axis, float angle)
    axis, angle = q_diff.to_axis_angle()

    if axis.length < 1e-6:
        return [0.0, 0.0, 0.0]

    axis.normalize()
    omega = [(angle / dt) * axis.x, (angle / dt) * axis.y, (angle / dt) * axis.z]
    return omega


# ============================================================
# MAIN EXPORT
# ============================================================
def export_motion():
    armature = bpy.data.objects[ARMATURE_NAME]
    scene = bpy.context.scene
    frame_duration = 1.0 / FPS
    num_joints = len(PM01_JOINTS)

    all_frames = []
    prev_root_pos = None
    prev_root_quat = None
    prev_dof_pos = None
    prev_foot_pos = None

    print(f"Exporting frames {scene.frame_start} to {scene.frame_end}...")

    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)

        # --- Root position & rotation ---
        hips = armature.pose.bones["mixamorig:Hips"]
        root_world = armature.matrix_world @ hips.head
        root_pos = [root_world.x, root_world.y, root_world.z]

        root_mat = armature.matrix_world @ hips.matrix
        root_quat = root_mat.to_quaternion()
        root_rot = [root_quat.x, root_quat.y, root_quat.z, root_quat.w]

        # --- Joint angles (24 DOFs) ---
        dof_pos = [0.0] * num_joints
        for jname, jinfo in JOINT_LOOKUP.items():
            mixamo_bone_name = JOINT_TO_MIXAMO.get(jname)
            if mixamo_bone_name is None:
                continue
            try:
                bone = armature.pose.bones[mixamo_bone_name]
                angle = project_rotation_to_axis(bone, jinfo["axis"])
                angle = clamp(angle, jinfo["lower"], jinfo["upper"])
                dof_pos[jinfo["index"]] = angle
            except KeyError:
                pass

        # --- Foot positions (local to root) ---
        foot_pos = []
        for fname in FOOT_BONES:
            try:
                bone = armature.pose.bones[fname]
                foot_world = armature.matrix_world @ bone.head
                foot_local = foot_world - root_world
                foot_pos.extend([foot_local.x, foot_local.y, foot_local.z])
            except KeyError:
                foot_pos.extend([0.0, 0.0, 0.0])

        # --- Velocities (finite differences) ---
        if prev_root_pos is not None:
            dt = frame_duration
            root_lin_vel = [(root_pos[i] - prev_root_pos[i]) / dt for i in range(3)]
            root_ang_vel = compute_angular_velocity(prev_root_quat, root_quat, dt)
            dof_vel = [(dof_pos[i] - prev_dof_pos[i]) / dt for i in range(num_joints)]
            foot_vel = [(foot_pos[i] - prev_foot_pos[i]) / dt for i in range(len(foot_pos))]
        else:
            root_lin_vel = [0.0, 0.0, 0.0]
            root_ang_vel = [0.0, 0.0, 0.0]
            dof_vel = [0.0] * num_joints
            foot_vel = [0.0] * len(foot_pos)

        # --- Build flat frame (73 values) ---
        flat = (
            root_pos +          # [0:3]   3 values
            root_rot +          # [3:7]   4 values
            dof_pos +           # [7:31]  24 values
            foot_pos +          # [31:37] 6 values
            root_lin_vel +      # [37:40] 3 values
            root_ang_vel +      # [40:43] 3 values
            dof_vel +           # [43:67] 24 values
            foot_vel            # [67:73] 6 values
        )

        assert len(flat) == 73, f"Expected 73 values, got {len(flat)}"
        all_frames.append(flat)

        prev_root_pos = root_pos
        prev_root_quat = root_quat
        prev_dof_pos = dof_pos
        prev_foot_pos = foot_pos

    # --- Save JSON ---
    output = {
        "LoopMode": "Wrap",
        "FrameDuration": frame_duration,
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": True,
        "MotionWeight": 1,
        "Frames": all_frames,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f)

    duration = len(all_frames) * frame_duration
    print(f"Export complete!")
    print(f"  Frames: {len(all_frames)}")
    print(f"  Values per frame: {len(all_frames[0])}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  FPS: {FPS}")
    print(f"  Saved to: {OUTPUT_PATH}")


# Run
export_motion()
