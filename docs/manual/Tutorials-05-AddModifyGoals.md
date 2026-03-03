# Tutorials

## Add/Modify Goals

### Goals Class

#### Base Goals

Go to [engineai_gym/engineai_gym/envs/base/goals/goals.py](../../engineai_gym/engineai_gym/envs/base/goals/goals.py) , add the new goals

```python
def pos_phase(self):
    phase = self.env.get_phase()
    sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
    cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)
    return torch.cat((sin_pos, cos_pos), dim=1)
```

##### Specific goals (robot type/robots/tasks)

If you need to override the base goals for specific goals, enter
- `engineai_gym/engineai_gym/envs/robots/{robot type}`, create `goals_ {robot_type}.py`
- Or `engineai_gym/engineai_gym/envs/robots/{robot type}/{robot}`, create `goals_ {robot}.py`
- Or `engineai_gym/engineai_gym/envs/robots/{robot type}/{robot}/{task}`, create `goals_ {task}.py`

Then add the new goals.

##### Example

```python
from engineai_gym.envs.base.goals.goals import Goals
import torch


class GoalsBiped(Goals):
    def pos_phase(self):
        phase = self.env.get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)
        return torch.cat((sin_pos, cos_pos), dim=1)
```

#### Goals Config

In `config_{robot_type}/{robot}_[{task}]`, under `env` class, add `goal_list` 

##### Example

```python
from engineai_rl_lib.base_config import BaseConfig


class ConfigLeggedRobot(BaseConfig):
    class env:
        # goals to save in goal_dict
        goal_list = ["commands"]
```