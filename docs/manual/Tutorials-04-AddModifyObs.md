# Tutorials

## Add/Modify Obs

### Obs Class

#### Base obs

Go to [engineai_gym/engineai_gym/envs/base/obs/obs.py](../../engineai_gym/engineai_gym/envs/base/obs/obs.py) , add the new obs

```python
def dof_pos(self):
    try:
        return self.env.domain_rands.domain_rands_type_obs_lag.lagged_dof_pos - self.env.default_dof_pos
    except:
        return self.env.dof_pos - self.env.default_dof_pos
```

##### Specific obs (robot type/robots/tasks)

If you need to override base obs for specific obs, enter
- `engineai_gym/engineai_gym/envs/robots/{robot type}`, create `obs_ {robot_type}.py`
- Or `engineai_gym/engineai_gym/envs/robots/{robot type}/{robot}`, create `obs_ {robot}.py`
- Or `engineai_gym/engineai_gym/envs/robots/{robot type}/{robot}/{task}`, create `obs_ {task}.py`

Then add the new obs.

##### Example

```python
from engineai_gym.envs.base.obs.obs import Obs


class ObsAnymalC(Obs):
    pass
```

#### Obs Config

In `config_{robot_type}/{robot}_[{task}]`, under `env` class, add `obs_list` 

##### Example

```python
from engineai_rl_lib.base_config import BaseConfig


class ConfigLeggedRobot(BaseConfig):
    class env:
        # obs to save in obs_dict
        obs_list = [
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "dof_pos",
            "dof_vel",
            "actions",
            "height_measurements",
        ]
```