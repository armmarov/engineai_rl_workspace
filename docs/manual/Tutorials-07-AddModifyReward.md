# Tutorials

## Add/Modify Reward

### Reward Class

#### Base Reward

Go to [engineai_gym/engineai_gym/envs/base/rewards](../../engineai_gym/engineai_gym/envs/base/rewards) find the corresponding rewards_type and add a new rewards to it. If there is no corresponding rewards_type, create one and inherit `RewardsBase`

##### Example

```python
class RewardsTypeAction(RewardsBase):

    def reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)

    def reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.env.last_actions - self.env.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.env.actions + self.env.last_last_actions - 2 * self.env.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.env.actions), dim=1)
        return term_1 + term_2 + term_3
```

##### Specific Reward

If you need to override the base reward for a specific reward, enter
- `engineai_gym/engineai_gym/envs/robots/{robot type}`, create `rewards_{robot_type}.py`
- Or `engineai_gym/engineai_gym/envs/robots/{robot type}/{robot}`, create `rewards_{robot}.py`
- Or `engineai_gym/engineai_gym/envs/robots/{robot type}/{robot}/{task}`, create `rewards_{task}.py`

Then add new rewards

##### Example

```python
class RewardsBiped(Rewards):
    def reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.env.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.env.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes.
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error = torch.sum(torch.square(
            self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.env.cfg.rewards.tracking_sigma)
```

#### Reward Config

In `config_{robot_type}/{robot}_[{task}]`, under `rewards` class, add `params` and `scales`

##### Example

```python
from engineai_rl_lib.base_config import BaseConfig


class ConfigLeggedRobot(BaseConfig):
    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.0
            torques = -1e-05
            dof_vel = -0.0
            dof_acc = -2.5e-07
            base_height = -0.0
            feet_air_time = 1.0
            collision = -1.0
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.0
            feet_distance = 0.2
            foot_slip = -0.1
            base_acc = 0.2
            vel_mismatch_exp = 0.5
            track_vel_hard = 0.5
            default_joint_pos = 0.8
            low_speed = 0.2
            action_smoothness = -0.003
    
        class params:
            # if true negative total rewards are clipped at zero (avoids early termination problems)
            only_positive_rewards = True
            # tracking reward = exp(-error^2/sigma)
            tracking_sigma = 0.25
            # multiplier of URDF limits, values above this limit are penalized
            soft_dof_pos_limit_multi = 1.0
            soft_dof_vel_limit_multi = 1.0
            soft_torque_limit_multi = {
                "joint": 1.0,
            }
            base_height_target = 1.0
            # forces above this value are penalized
            max_contact_force = 100.0
            min_feet_dist = 0.15
            max_feet_dist = 0.8
            target_feet_height = 0.2
```