from engineai_rl_lib.ref_state.data_base import DataBase
from engineai_rl_lib.ref_state import pose3d, ref_state_util
from engineai_rl_lib.math import interpolate, slerp


class Data(DataBase):
    """
    PM01 full body motion data format.
    Each frame contains 73 values:
        [0:3]     root_pos        - Base position (x, y, z)
        [3:7]     root_rot        - Base rotation quaternion (qx, qy, qz, qw)
        [7:31]    dof_pos         - 24 joint positions
        [31:37]   foot_pos_local  - 2 feet x 3D local positions
        [37:40]   root_lin_vel    - Base linear velocity
        [40:43]   root_ang_vel    - Base angular velocity
        [43:67]   dof_vel         - 24 joint velocities
        [67:73]   foot_vel_local  - 2 feet x 3D local velocities
    """

    def component_root_pos(self, trajectory):
        return trajectory[:, :3]

    def component_root_z_pos(self, trajectory):
        return trajectory[:, 2:3]

    def component_root_rot(self, trajectory):
        root_rot = trajectory[:, 3:7]
        root_rot = pose3d.quaternion_normalize(root_rot)
        root_rot = ref_state_util.standardize_quaternion(root_rot)
        return root_rot

    def component_dof_pos(self, trajectory):
        return trajectory[:, 7:31]

    def component_foot_pos_local(self, trajectory):
        return trajectory[:, 31:37]

    def component_root_lin_vel(self, trajectory):
        return trajectory[:, 37:40]

    def component_root_ang_vel(self, trajectory):
        return trajectory[:, 40:43]

    def component_dof_vel(self, trajectory):
        return trajectory[:, 43:67]

    def component_foot_vel_local(self, trajectory):
        return trajectory[:, 67:73]

    def blend_root_rot(self, frame_start, frame_end, blend):
        root_rot = slerp(frame_start, frame_end, blend)
        root_rot = ref_state_util.standardize_quaternion(root_rot)
        return root_rot
