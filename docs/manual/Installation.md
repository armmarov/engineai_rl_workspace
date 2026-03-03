# Installation

## Install Linux dependencies

```shell
sudo apt update
sudo apt install redis-server
```

## Create Virtual Env

Generate a new Python virtual environment with Python 3.8

```shell
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
apt-cache policy python3.8
sudo apt install python3.8 python3.8-dev python3.8-venv
python3.8 -m venv engineai_rl_ws
```

## Install Isaac Gym

- Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
  - `cd isaacgym/python && pip install -e .`
- Adapt to latest numpy `sed -i 's/np.float/float/' isaacgym/torch_utils.py`
- Run an example with `cd examples && python 1080_balls_of_solitude.py`.
- Consult `isaacgym/docs/index.html` for troubleshooting.

## Install engineai_rl_workspace

1. Clone engineai_rl_workspace: `git clone https://github.com/engineai-robotics/engineai_rl_workspace.git`
2. Install engineai_gym: `cd engineai_gym && pip install -e .`
3. Install engineai_rl: `cd engineai_rl && pip install -e .`
4. Install engineai_rl_lib: `cd engineai_rl_lib && pip install -e .`
5. Install engineai_rl_workspace: `pip install -e .`



