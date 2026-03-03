# Tutorials

## Start/Resume Training

### Args

- `--exp_name EXP_NAME`: Experiment name.
- `--sub_exp_name SUB_EXP_NAME`: Name of the sub-experiment to run or load, default is default.
- `--run_name RUN_NAME`: Name of the run, default is current time %Y-%m-%d_%H-%M-%S.
- `--log_roo`: Path of log_root, default is engineai_rl_workspace/logs/{exp_name}/{sub_exp_name}.
- `--load_run LOAD_RUN`: Name of the run to load when resume=True, default is -1. If -1`: will load the last run.
- `--checkpoint CHECKPOINT`: Saved model checkpoint number, default is -1. If -1`: will load the last checkpoint.
- `--resume`: Resume training from a checkpoint
- `--run_exist`: Run training from an existing run with its config.json.
- `--debug`: In debug mode, no logs will be saved.
- `--num_envs NUM_ENVS`: Number of environments to create.
- `--seed SEED`: Random seed.
- `--max_iterations MAX_ITERATIONS`: Maximum number of training iterations.
- `--logger LOGGER`: Logger module to use. Choice`: tensorboard, wandb, neptune.
- `--upload_model`: upload models to wandb or neptune.
- `--sim_device SIM_DEVICE`: Device used by the simulator, (cpu, gpu, cuda:0, cuda:1 etc..), default is cuda:0.
- `--rl_device RL_DEVICE`: Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..), default is cuda:0.
- `--sim_devices SIM_DEVICES`: For multi-GPU training, devices used by the simulator, (cpu, gpu, cuda:0, cuda:1 etc..)
- `--rl_devices RL_DEVICES`: For multi-GPU training, devices used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)
- `--video`: Record video during training. Headless mode also works.
- `--record_length RECORD_LENGTH`: The number of steps to record for videos, default is 200.
- `--record_interval RECORD_INTERVAL`: The number of step as interval to record a video.
- `--fps FPS`: The fps of recorded videos, default is 50.
- `--frame_size FRAME_SIZE`: The size of recorded frame, default is (1280, 720).
- `--camera_offset CAMERA_OFFSET`: The offset of the video filming camera, default is (0, -2, 0).
- `--camera_rotation CAMREA_ROTATION`: The rotation of the video filming camera, default is (0, 0, 90).
- `--env_idx_record ENV_IDX_RECORD`: The env idx to record, default is 0.
- `--actor_idx_record ACTOR_IDX_RECORD`: The actor idx to record, default is 0.
- `--rigid_body_idx_record RIGID_BODY_IDX_RECORD`: The rigid_body idx to record, default is 0.

### Examples

#### From Scratch

Files required to resume the run will be saved for resume or play, which will work even when the code is changed.

```shell
# basic
python engineai_rl_workspace/scripts/train.py --exp_name pm01_rough_ppo
# headless
python engineai_rl_workspace/scripts/train.py --exp_name pm01_rough_ppo --headless
# use specific logger
python engineai_rl_workspace/scripts/train.py --exp_name pm01_rough_ppo --headless --logger wandb
# run with params overriden
python engineai_rl_workspace/scripts/train.py --exp_name pm01_rough_ppo --headless --num_envs 4096 --max_iterations 30000 --seed 1
# multi-GPU Traning
MASTER_PORT=12346 torchrun --standalone --nnodes=1 --nproc_per_node=2 engineai_rl_workspace/scripts/train.py --exp_name pm01_rough_ppo --headless --sim_devices cuda:0,cuda:1 --rl_devices cuda:0,cuda:1
```

#### Video Recording

```shell
# default setting
python engineai_rl_workspace/scripts/train.py --exp_name pm01_rough_ppo --headless --video
# custom setting
python engineai_rl_workspace/scripts/train.py --exp_name pm01_rough_ppo --headless --video --record_length 500 --record_interval 100 --fps 100 --frame_size=1920,1080 --camera_offset=-2,0,0 --camera_rotation=0,0,90 --env_idx_record 1 --actor_idx_record 1 --rigid_body_idx_record
```

#### From `.json` Config from Scratch

Since a config is saved for each, if you want to start a new run with modification of the `.json` config of a old run, you can create a new folder copying the old config, modify the config, and run a training from it. The Algos config files will be converted to `.py` config files, and used for training.

```shell
# from a default sub_exp_name
python engineai_rl_workspace/scripts/train.py --exp_name pm01_rough_ppo --headless --run_exist --load_run 2025-06-03_12-00-00
# from a specific sub_exp_name
python engineai_rl_workspace/scripts/train.py --exp_name pm01_rough_ppo --headless --run_exist --sub_exp_name fixed_std --load_run 2025-06-03_12-00-00 
```

#### Using a Specific Logger (Tensorboard, Wandb, Neptune)

```shell
python engineai_rl_workspace/scripts/train.py --exp_name pm01_rough_ppo --headless --logger wandb
```

#### Resume a Run

```shell
# resume from default log root
python engineai_rl_workspace/scripts/train.py --exp_name pm01_rough_ppo --headless --resume --load_run 2025-06-03_12-00-00
# resume from a specific log root
python engineai_rl_workspace/scripts/train.py --exp_name pm01_rough_ppo --headless --resume --log_root ~/server/engineai_rl_workspace/logs/pm01_rough_ppo/default --load_run 2025-06-03_12-00-00
# resume from a specific checkpoint
python engineai_rl_workspace/scripts/train.py --exp_name pm01_rough_ppo --headless --resume  --load_run 2025-06-03_12-00-00 --checkpoint 30000
```

#### Debug Mode

Training won't save log files in debug mode, so user can maintain a clean log directory

```shell
# debug mode
python engineai_rl_workspace/scripts/train.py --exp_name pm01_rough_ppo --headless --debug
```