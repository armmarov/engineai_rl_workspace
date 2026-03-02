"""
Blender script to preview exported PM01 motion JSON.

Creates a simple stick figure armature and plays back the motion data.

Usage:
    1. Open Blender (new scene, delete default cube)
    2. Open Scripting tab
    3. Paste/open this script
    4. Update JSON_PATH below
    5. Run script
    6. Press Space in viewport to play animation
"""

import bpy
import json
import math
from mathutils import Vector, Quaternion, Euler, Matrix

# ============================================================
# CONFIGURATION
# ============================================================
JSON_PATH = "/home/armmarov/work/robot/engineai/engineai_rl_workspace/engineai_gym/engineai_gym/resources/robots/biped/pm01/mocap_motions/dance1.json"

# PM01 joint names (same order as in data.py indices [7:31])
JOINT_NAMES = [
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

# PM01 joint axes (from URDF)
JOINT_AXES = [
    (0, 0.96593, -0.25882),  # j00
    (1, 0, 0),               # j01
    (0, 0, 1),               # j02
    (0, 1, 0),               # j03
    (0, 1, 0),               # j04
    (1, 0, 0),               # j05
    (0, 0.96593, 0.25882),   # j06
    (1, 0, 0),               # j07
    (0, 0, 1),               # j08
    (0, 1, 0),               # j09
    (0, 1, 0),               # j10
    (1, 0, 0),               # j11
    (0, 0, 1),               # j12
    (0, 0.99803, 0.062791),  # j13
    (1, 0, 0),               # j14
    (0, -0.062803, 0.99803), # j15
    (0.0027243, 0.99802, 0.062803),  # j16
    (-0.21479, -0.061921, 0.9747),   # j17
    (0, 0.99803, -0.062791), # j18
    (1, 0, 0),               # j19
    (0, 0.062791, 0.99803),  # j20
    (0.0027243, 0.99802, -0.06279),  # j21
    (-0.21479, 0.061909, 0.9747),    # j22
    (0, 0, 1),               # j23
]

# PM01 skeleton: parent-child with offsets from URDF
# (joint_name, parent_joint, offset_xyz)
SKELETON = [
    ("base",               None,               (0, 0, 0)),
    ("j00_hip_pitch_l",    "base",             (0.01541, 0.076141, -0.061208)),
    ("j01_hip_roll_l",     "j00_hip_pitch_l",  (0.048, 0.049359, -0.013226)),
    ("j02_hip_yaw_l",      "j01_hip_roll_l",   (-0.03139, -0.0015951, -0.086016)),
    ("j03_knee_pitch_l",   "j02_hip_yaw_l",    (-0.02602, -2.8566e-05, -0.23655)),
    ("j04_ankle_pitch_l",  "j03_knee_pitch_l",  (-0.026756, 0.00041994, -0.36305)),
    ("j05_ankle_roll_l",   "j04_ankle_pitch_l", (0, 0, -0.015)),
    ("j06_hip_pitch_r",    "base",             (0.01541, -0.076141, -0.061208)),
    ("j07_hip_roll_r",     "j06_hip_pitch_r",  (0.048, -0.04936, -0.013226)),
    ("j08_hip_yaw_r",      "j07_hip_roll_r",   (-0.03139, 0.0015966, -0.086016)),
    ("j09_knee_pitch_r",   "j08_hip_yaw_r",    (-0.02602, 2.8566e-05, -0.23655)),
    ("j10_ankle_pitch_r",  "j09_knee_pitch_r",  (-0.026756, -0.00041994, -0.36305)),
    ("j11_ankle_roll_r",   "j10_ankle_pitch_r", (0, 0, -0.015)),
    ("j12_waist_yaw",      "base",             (0.01216, 0, 0.0809)),
    ("j13_shoulder_pitch_l","j12_waist_yaw",    (-0.027105, 0.12916, 0.21549)),
    ("j14_shoulder_roll_l", "j13_shoulder_pitch_l", (-0.0371, 0.066941, -0.020838)),
    ("j15_shoulder_yaw_l",  "j14_shoulder_roll_l",  (0.0371, 0.017645, -0.070132)),
    ("j16_elbow_pitch_l",   "j15_shoulder_yaw_l",   (0, 0.0065994, -0.10487)),
    ("j17_elbow_yaw_l",     "j16_elbow_pitch_l",    (0.013817, 0.0097723, -0.1547)),
    ("j18_shoulder_pitch_r","j12_waist_yaw",    (-0.027105, -0.12916, 0.21549)),
    ("j19_shoulder_roll_r", "j18_shoulder_pitch_r", (-0.0371, -0.066941, -0.020838)),
    ("j20_shoulder_yaw_r",  "j19_shoulder_roll_r",  (0.0371, -0.017644, -0.070132)),
    ("j21_elbow_pitch_r",   "j20_shoulder_yaw_r",   (0, -0.006598, -0.10487)),
    ("j22_elbow_yaw_r",     "j21_elbow_pitch_r",    (0.013817, -0.0097704, -0.1547)),
    ("j23_head_yaw",        "j12_waist_yaw",    (-0.017638, 0, 0.2961)),
]


def create_armature():
    """Create PM01 armature from skeleton definition."""
    bpy.ops.object.armature_add(enter_editmode=True)
    armature_obj = bpy.context.object
    armature_obj.name = "PM01_Preview"
    armature = armature_obj.data
    armature.name = "PM01_Armature"

    # Remove default bone
    for bone in armature.edit_bones:
        armature.edit_bones.remove(bone)

    # Create bones
    bone_map = {}
    for name, parent_name, offset in SKELETON:
        bone = armature.edit_bones.new(name)
        bone.length = 0.05  # default length

        if parent_name and parent_name in bone_map:
            parent_bone = bone_map[parent_name]
            bone.parent = parent_bone
            bone.head = parent_bone.head + Vector(offset)
        else:
            bone.head = Vector((0, 0, 0.9))  # PM01 standing height

        bone.tail = bone.head + Vector((0, 0, 0.03))
        bone_map[name] = bone

    bpy.ops.object.mode_set(mode='POSE')
    return armature_obj


def apply_motion(armature_obj, frames, fps):
    """Apply motion data as keyframes on the armature."""
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = len(frames)
    scene.render.fps = fps

    # Build joint name to index mapping
    joint_idx = {name: i for i, name in enumerate(JOINT_NAMES)}
    joint_axis_map = {name: Vector(axis).normalized() for name, axis in zip(JOINT_NAMES, JOINT_AXES)}

    for frame_num, frame_data in enumerate(frames):
        frame = frame_num + 1  # Blender frames start at 1
        scene.frame_set(frame)

        # Root position
        root_pos = Vector((frame_data[0], frame_data[1], frame_data[2]))
        root_rot = Quaternion((frame_data[6], frame_data[3], frame_data[4], frame_data[5]))  # Blender: wxyz

        # Set root bone
        base_bone = armature_obj.pose.bones.get("base")
        if base_bone:
            base_bone.location = root_pos - Vector((0, 0, 0.9))  # offset from rest
            base_bone.rotation_mode = 'QUATERNION'
            base_bone.rotation_quaternion = root_rot
            base_bone.keyframe_insert(data_path="location", frame=frame)
            base_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # Set joint angles
        dof_pos = frame_data[7:31]
        for jname in JOINT_NAMES:
            bone = armature_obj.pose.bones.get(jname)
            if bone is None:
                continue

            idx = joint_idx[jname]
            angle = dof_pos[idx]
            axis = joint_axis_map[jname]

            # Create rotation quaternion from axis-angle
            bone.rotation_mode = 'QUATERNION'
            bone.rotation_quaternion = Quaternion(axis, angle)
            bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    print(f"Applied {len(frames)} frames of motion data.")


def main():
    # Load JSON
    print(f"Loading: {JSON_PATH}")
    with open(JSON_PATH) as f:
        data = json.load(f)

    frames = data["Frames"]
    fps = int(round(1.0 / data["FrameDuration"]))

    print(f"Frames: {len(frames)}, FPS: {fps}, Duration: {len(frames) * data['FrameDuration']:.2f}s")

    # Create armature and apply motion
    armature_obj = create_armature()
    apply_motion(armature_obj, frames, fps)

    # Go to frame 1
    bpy.context.scene.frame_set(1)
    print("Done! Press Space in the viewport to play the animation.")


main()
