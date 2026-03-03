# Tutorials

## Testers & Loggers

### Overview

- Testers are used for testing a policy with specific setting.
- Loggers are used for logging data for a tester.

### Add a Tester

Enter `engineai_rl_workspace/tester/testers`, create `tester_{tester_name}.py`, write `TesterClass` here and inherit from `TesterBase`

#### Example

```python
from .tester_base import TesterTypeBase
from engineai_rl_workspace.utils import convert_to_visible_commands

class TesterBackwardCommandsOnly(TesterTypeBase):
    def set_commands(self) -> None:
        convert_to_visible_commands(self.env.commands)
        self.env.commands[:, 0] = -abs(self.env.commands[:, 0])
        self.env.commands[:, 1:] = 0
```

Register the tester in the `.yaml` file and specify the location of the yaml file in the Env config

#### Example

```yaml
testers:
  tester_normal_commands:
    loggers:
      [ logger_type_position,
        logger_type_torque,
        logger_type_vel,
        logger_type_force ]
```

```python
class ConfigLeggedRobot(BaseConfig):
    class env:
        tester_config_path = '{ENGINEAI_GYM_PACKAGE_DIR}/tester/tester_config.yaml'
```

### Add a Logger

Go to `engineai_gym/tester/loggers`, create `logger_{logger_name}.py`, write `LoggerClass` here and inherit from `LoggerBase`

#### Example

```python
from engineai_gym.tester.loggers.logger_base import LoggerBase


class LoggerTypeBaseVel(LoggerBase):

    def log_data(self, args) -> dict:
        self.add_data_to_log({
                                 'command_x': self.env.commands[args["robot_index"], 0].item(),
                                 'command_y': self.env.commands[args["robot_index"], 1].item(),
                                 'command_yaw': self.env.commands[args["robot_index"], 2].item(),
                                 'base_vel_x': self.env.base_lin_vel[args["robot_index"], 0].item(),
                                 'base_vel_y': self.env.base_lin_vel[args["robot_index"], 1].item(),
                                 'base_vel_yaw': self.env.base_ang_vel[args["robot_index"], 2].item(), })

    def retrieve_data(self):
        self.save_to_csv("Base velocity x", [self.log["base_vel_x"], self.log["command_x"]], "base lin vel [m/s]",
                         ["base_vel_x", "command_x"], ["measured", "target"])
        self.save_to_csv("Base velocity y", [self.log["base_vel_y"], self.log["command_y"]], "base lin vel [m/s]",
                         ["base_vel_y", "command_y"], ["measured", "target"])
        self.save_to_csv("Base velocity yaw", [self.log["base_vel_yaw"], self.log["command_yaw"]],
                         "base ang vel [rad/s]", ["base_vel_yaw", "command_yaw"], ["measured", "target"])
```

Register the logger to a tester in the `.yaml` file

#### Example

```yaml
testers:
  tester_normal_commands:
    loggers:
      [ logger_type_position,
        logger_type_torque,
        logger_type_vel,
        logger_type_force ]
```

