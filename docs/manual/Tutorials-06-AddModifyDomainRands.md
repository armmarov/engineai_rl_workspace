# Tutorials

## Add/Modify Domain Rands

### Domain Rands Class

#### Base domain rands

Go to [engineai_gym/engineai_gym/envs/base/domain_rands ](../../engineai_gym/engineai_gym/envs/base/domain_rands) find the corresponding domain_rands_type and add a new domain_rands to it. If there is no corresponding domain_rands, create one and inherit `DomainRandsBase`

##### Example

```python
class DomainRandsTypeRigidShape(DomainRandsBase):
    def init_rand_vec_on_create_env(self):
        # prepare friction randomization
        friction_range = self.env.cfg.domain_rand.rigid_shape.friction_range
        num_buckets = 64
        bucket_ids = torch.randint(0, num_buckets, (self.env.num_envs, 1))
        friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device='cpu')
        self.friction_coeffs = friction_buckets[bucket_ids]

    def process_on_create_env(self, props, env_id):
        """ randomize the rigid shape properties of each environment.
        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id
        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.env.cfg.domain_rand.rigid_shape.randomize_friction:
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
            self.env.env_frictions[env_id] = self.friction_coeffs[env_id]
        return props
```

##### Specific domain rands

If you need to cover the base domain rands for specific domain rands, enter
- `engineai_gym/engineai_gym/envs/robots/{robot type}`, create `domain_rands_type_{domain_rands_type}_{robot_type}.py` and `domain_rands_{robot_type}.py`
- Or `/engineai_gym/envs/robots/{robot type}/{robot}`,  create `domain_rands_type_{domain_rands_type}_{robot}.py` and `domain_rands_{robot}.py`
- `engineai_gym /engineai_gym/envs/robots/{robot type}/{robot}/{task}`, create `domain_rands_type_{domain_rands_type}_{task}.py` and `domain_rands_{task}.py`

In `domain_rands_type_{...}`, override the target domain_rands method, and in `domain_rands_{...}.py`, create an instance, the name must be `self.domain_rands_type_{...}`

##### Example

```python
from engineai_gym.envs.base.domain_rands.domain_rands_type_dof import DomainRandsTypeDof

class DomainRandsTypeDofAnymalC(DomainRandsTypeDof):
    pass
```

```python
from engineai_gym.envs.base.domain_rands.domain_rands import DomainRands
from .domain_rands_type_dof_anymal_c import DomainRandsTypeDofAnymalC


class DomainRandsAnymalC(DomainRands):
    def __init__(self, env):
        super().__init__(env)
        self.domain_rands_type_dof = DomainRandsTypeDofAnymalC(env)
```

#### Domain Rands Config

In `config_{robot_type}/{robot}_[{task}]`, under `domain_rands` class, add domain rands config

##### Example

```python
from engineai_rl_lib.base_config import BaseConfig


class ConfigLeggedRobot(BaseConfig):
    class domain_rands:
        class rigid_shape:
            randomize_friction = True
            friction_range = [0.5, 1.25]
            randomize_restitution = True
            restitution_range = [0.0, 0.4]
    
        class rigid_body:
            randomize_base_mass = False
            added_mass_range = [-2.5, 2.5]
            randomize_com = False
            com_displacement_range = [-0.05, 0.05]
            randomize_link_mass = False
            link_mass_multi_range = [0.9, 1.1]
    
        class dof:
            randomize_gains = False
            stiffness_multi_range = [0.8, 1.2]
            damping_multi_range = [0.8, 1.2]
            randomize_torque = False
            torque_multip_range = [0.8, 1.2]
            randomize_motor_offset = False
            motor_offset_range = [-0.035, 0.035]
            randomize_joint_friction = False
            randomize_joint_friction_each_joint = False
            joint_friction_multi_range = [0.01, 1.15]
            joint_friction_multi_range_each_joint = {
                "joint_a": [0.01, 1.15],
                "joint_b": [0.5, 1.3],
            }
            randomize_joint_armature = False
            randomize_joint_armature_each_joint = False
            joint_armature_multi_range = [-0.03, 0.03]
            joint_armature_multi_range_each_joint = {
                "joint_a": [-0.03, 0.03],
                "joint_b": [-0.03, 0.03],
            }
            randomize_coulomb_friction = False
            joint_coulomb_range = [0.1, 0.9]
            joint_viscous_range = [0.05, 0.1]
    
        class action_lag:
            # if action_lag_timesteps > 1 and randomize_action_lag_timesteps = False, the action_lag_timesteps is fixed.
            # if randomize_action_lag_timesteps = True and randomize_action_lag_timesteps_perstep = False, action_lag_timesteps will be randomized on reset
            # if randomize_action_lag_timesteps = True and randomize_action_lag_timesteps_perstep = True, action_lag_timesteps will be randomized on each step
            action_lag_timesteps = 0
            randomize_action_lag_timesteps = False
            randomize_action_lag_timesteps_perstep = False
            action_lag_timesteps_range = [2, 4]
    
        class obs_lag:
            # if motor_lag_timesteps > 1 and randomize_motor_lag_timesteps = False, the motor_lag_timesteps is fixed.
            # if randomize_motor_lag_timesteps = True and randomize_motor_lag_timesteps_perstep = False, motor_lag_timesteps will be randomized on reset
            # if randomize_motor_lag_timesteps = True and randomize_motor_lag_timesteps_perstep = True, motor_lag_timesteps will be randomized on each step
            motor_lag_timesteps = 0
            randomize_motor_lag_timesteps = False
            randomize_motor_lag_timesteps_perstep = False
            motor_lag_timesteps_range = [2, 10]
            # if imu_lag_timesteps > 1 and randomize_imu_lag_timesteps = False, the imu_lag_timesteps is fixed.
            # if randomize_imu_lag_timesteps = True and randomize_imu_lag_timesteps_perstep = False, imu_lag_timesteps will be randomized on reset
            # if randomize_imu_lag_timesteps = True and randomize_imu_lag_timesteps_perstep = True, imu_lag_timesteps will be randomized on each step
            imu_lag_timesteps = 0
            randomize_imu_lag_timesteps = False
            randomize_imu_lag_timesteps_perstep = False
            imu_lag_timesteps_range = [1, 2]
    
        class disturbance:
            push_robots = True
            push_interval_s = 15
            max_push_vel_xy = 1.0
            max_push_ang_vel = 0.6
```