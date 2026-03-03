from engineai_gym.envs.robots.biped.pm01.rough.config_pm01_rough import (
    ConfigPm01Rough,
)
from engineai_gym.envs.base.config_legged_robot_ref import ConfigLeggedRobotRef


class ConfigPm01BodyBlock(ConfigPm01Rough):
    class env(ConfigPm01Rough.env):
        env_spacing = 1.5
        # Full body: 12 leg joints + 1 waist + 10 arm joints + 1 head = 24 joints
        action_joints = [
            "j00_hip_pitch_l",
            "j01_hip_roll_l",
            "j02_hip_yaw_l",
            "j03_knee_pitch_l",
            "j04_ankle_pitch_l",
            "j05_ankle_roll_l",
            "j06_hip_pitch_r",
            "j07_hip_roll_r",
            "j08_hip_yaw_r",
            "j09_knee_pitch_r",
            "j10_ankle_pitch_r",
            "j11_ankle_roll_r",
            "j12_waist_yaw",
            "j13_shoulder_pitch_l",
            "j14_shoulder_roll_l",
            "j15_shoulder_yaw_l",
            "j16_elbow_pitch_l",
            "j17_elbow_yaw_l",
            "j18_shoulder_pitch_r",
            "j19_shoulder_roll_r",
            "j20_shoulder_yaw_r",
            "j21_elbow_pitch_r",
            "j22_elbow_yaw_r",
            "j23_head_yaw",
        ]

        obs_list = [
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "dof_pos",
            "absolute_dof_pos",
            "dof_vel",
            "actions",
            "foot_pos",
            "base_z_pos",
        ]
        goal_list = ["commands"]
        use_ref_actions = False
        episode_length_s = 20

    class terrain(ConfigPm01Rough.terrain):
        mesh_type = "plane"
        measure_heights = False

    class init_state(ConfigPm01Rough.init_state):
        pos = [0.0, 0.0, 0.9]

        default_joint_angles = {
            # Legs - same as walking
            "j00_hip_pitch_l": -0.24,
            "j01_hip_roll_l": 0.0,
            "j02_hip_yaw_l": 0.0,
            "j03_knee_pitch_l": 0.48,
            "j04_ankle_pitch_l": -0.24,
            "j05_ankle_roll_l": 0.0,
            "j06_hip_pitch_r": -0.24,
            "j07_hip_roll_r": 0.0,
            "j08_hip_yaw_r": 0.0,
            "j09_knee_pitch_r": 0.48,
            "j10_ankle_pitch_r": -0.24,
            "j11_ankle_roll_r": 0.0,
            # Upper body - default standing pose
            "j12_waist_yaw": 0.0,
            "j13_shoulder_pitch_l": 0.0,
            "j14_shoulder_roll_l": 0.0,
            "j15_shoulder_yaw_l": 0.0,
            "j16_elbow_pitch_l": 0.0,
            "j17_elbow_yaw_l": 0.0,
            "j18_shoulder_pitch_r": 0.0,
            "j19_shoulder_roll_r": 0.0,
            "j20_shoulder_yaw_r": 0.0,
            "j21_elbow_pitch_r": 0.0,
            "j22_elbow_yaw_r": 0.0,
            "j23_head_yaw": 0.0,
        }

    class control(ConfigPm01Rough.control):
        control_type = "P"

        stiffness = {
            "hip_pitch": 70,
            "hip_roll": 50,
            "hip_yaw": 50,
            "knee_pitch": 70,
            "ankle_pitch": 20,
            "ankle_roll": 20,
            "waist_yaw": 50,
            "shoulder_pitch": 40,
            "shoulder_roll": 40,
            "shoulder_yaw": 40,
            "elbow_pitch": 30,
            "elbow_yaw": 30,
            "head_yaw": 20,
        }

        damping = {
            "hip_pitch": 7.0,
            "hip_roll": 5.0,
            "hip_yaw": 5.0,
            "knee_pitch": 7.0,
            "ankle_pitch": 0.2,
            "ankle_roll": 0.2,
            "waist_yaw": 5.0,
            "shoulder_pitch": 4.0,
            "shoulder_roll": 4.0,
            "shoulder_yaw": 4.0,
            "elbow_pitch": 3.0,
            "elbow_yaw": 3.0,
            "head_yaw": 2.0,
        }

        action_scales = {
            "hip_pitch": 0.5,
            "hip_roll": 0.5,
            "hip_yaw": 0.5,
            "knee_pitch": 0.5,
            "ankle_pitch": 0.5,
            "ankle_roll": 0.5,
            "waist_yaw": 0.5,
            "shoulder_pitch": 0.5,
            "shoulder_roll": 0.5,
            "shoulder_yaw": 0.5,
            "elbow_pitch": 0.5,
            "elbow_yaw": 0.5,
            "head_yaw": 0.3,
        }

        decimation = 10  # 100hz

    class ref_state(ConfigLeggedRobotRef.ref_state):
        ref_state_loader = True
        ref_state_init = True
        motion_files_path = (
            "{ENGINEAI_GYM_PACKAGE_DIR}/resources/robots/biped/pm01/mocap_motions"
        )
        ref_state_init_prob = 0.85
        data_mapping = {
            "base_pos": "root_pos",
            "base_z_pos": "root_z_pos",
            "base_rot": "root_rot",
            "absolute_dof_pos": "dof_pos",
            "dof_vel": "dof_vel",
            "foot_pos": "foot_pos_local",
            "base_lin_vel": "root_lin_vel",
            "base_ang_vel": "root_ang_vel",
        }

    class tester(ConfigPm01Rough.tester):
        config_path = "{ENGINEAI_GYM_PACKAGE_DIR}/envs/robots/biped/pm01/body_block/tester_config.yaml"

    class asset(ConfigPm01Rough.asset):
        # Full body URDF with all joints movable
        file = "{ENGINEAI_GYM_PACKAGE_DIR}/resources/robots/biped/pm01/urdf/pm01.urdf"

        terminate_after_contacts_on = [
            "link_base",
            "link_knee_pitch",
        ]
        penalize_contacts_on = ["link_base"]
        self_collisions = 0

    class domain_rands(ConfigPm01Rough.domain_rands):
        class rigid_shape(ConfigPm01Rough.domain_rands.rigid_shape):
            friction_range = [0.25, 1.75]

        class rigid_body(ConfigPm01Rough.domain_rands.rigid_body):
            randomize_base_mass = True
            added_mass_range = [-1.0, 1.0]

        class dof(ConfigPm01Rough.domain_rands.dof):
            randomize_gains = True
            stiffness_multi_range = [0.9, 1.1]
            damping_multi_range = [0.9, 1.1]
            randomize_joint_friction = False
            randomize_joint_armature = False

    class commands(ConfigPm01Rough.commands):
        curriculum = False
        yaw_from_heading_target = False
        still_ratio = 0

        class ranges(ConfigPm01Rough.commands.ranges):
            lin_vel_x = [-0.5, 0.5]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.5, 0.5]

    class rewards(ConfigPm01Rough.rewards):
        class params(ConfigPm01Rough.rewards.params):
            base_height_target = 0.8132
            soft_dof_pos_limit_multi = 0.9

        class scales(ConfigPm01Rough.rewards.scales):
            # AMP discriminator provides the main motion reward
            # Task rewards focus on balance and safety
            termination = -0.0
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            lin_vel_z = -0.0
            ang_vel_xy = -0.0
            orientation = -0.0
            torques = -0.0
            dof_vel = -0.0
            dof_acc = -0.0
            base_height = -0.0
            feet_air_time = 0.0
            collision = -0.0
            action_rate = -0.0
            stand_still = -0.0
            feet_distance = 0.0
            foot_slip = -0.0
            base_acc = 0.0
            vel_mismatch_exp = 0.0
            track_vel_hard = 0.0
            default_joint_pos = 0.0
            low_speed = 0.0
            action_smoothness = -0.0
            feet_contact_number = 0.0
            feet_clearance = 0.0
            dof_ref_pos_diff = 0.0
            knee_distance = 0.0
