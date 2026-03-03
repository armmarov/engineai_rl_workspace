# Tutorials

## Add/Modify Network

### Network Class

Go to `engineai_rl/engineai_rl/modules/networks`, create `{network_name}.py,` add network class inherited from `NetworkBase`

#### Example

```python
from engineai_rl.utils import get_activation
import numpy as np
from .network_base import NetworkBase  
import torch.nn as nn


class Mlp(NetworkBase):
    def __init__(
        self,
        num_input_dim,
        num_output_dim,
        hidden_dims=[256, 256],
        activation="elu",
        orthogonal_init=False,
        normalizer=None,
    ):
        super().__init__(num_input_dim, num_output_dim, orthogonal_init, normalizer)

        activation = get_activation(activation)

        # MLP
        mlp_trunk_layers = []
        mlp_trunk_layers.append(nn.Linear(num_input_dim, hidden_dims[0]))
        if self.orthogonal_init:
            nn.init.orthogonal_(mlp_trunk_layers[-1].weight, np.sqrt(2))
        mlp_trunk_layers.append(activation)
        for layer in range(len(hidden_dims)):
            if layer == len(hidden_dims) - 1:
                self.head = nn.Linear(hidden_dims[layer], num_output_dim)
                if self.orthogonal_init:
                    nn.init.orthogonal_(self.head.weight, 0.01)
                    nn.init.constant_(self.head.bias, 0.0)
            else:
                mlp_trunk_layers.append(
                    nn.Linear(hidden_dims[layer], hidden_dims[layer + 1])
                )
                if self.orthogonal_init:
                    nn.init.orthogonal_(mlp_trunk_layers[-1].weight, np.sqrt(2))
                    nn.init.constant_(mlp_trunk_layers[-1].bias, 0.0)
                mlp_trunk_layers.append(activation)
        self.trunk = nn.Sequential(*mlp_trunk_layers)

    def pure_forward(self, x):
        return self.head(self.trunk(x))
```

### Config

In `config_[{task}]_{algo}.py`, under `networks` class, add network config. 

#### Example

```python
from engineai_rl_lib.base_config import BaseConfig


class ConfigAlgoBase(ConfigPpo):
    class networks:
    # networks used for training
    training = ["actor", "critic"]
    # networks used for inference
    inference = ["actor"]

    class actor:
        # network class
        class_name = "Mlp"
        # args related to network input size.
        # format: {network_arg_names: input_name | int} or {arg_name: list(input_type: input_name | int)}
        # when it's a single value, the size is as it is. when it's a list, the size is all sizes added up
        # when it's input_name, the size is the corresponding input, when it's an int, the size is the int
        input_infos = {"num_input_dim": "actor"}
        output_infos = {"num_output_dim": "action"}
        # other network args
        hidden_dims = [512, 256, 128]
        # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"

    class critic:
        # network class
        class_name = "Mlp"
        # args related to network output size.
        # format: {network_arg_names: output_type | int} or {arg_name: list(output_type | int)}
        # when it's a single value, the size is as it is. when it's a list, the size is all sizes added up
        # when it's output_type, it can be "action" (num_actions) and value (1)

        input_infos = {"num_input_dim": "critic"}
        output_infos = {"num_output_dim": "value"}
        hidden_dims = [512, 256, 128]
        # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"
```