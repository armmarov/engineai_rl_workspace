# Tutorials

## Start Play

### Args

- `--exp_name EXP_NAME`: Experiment name.
- `--sub_exp_name SUB_EXP_NAME`: Name of the sub-experiment to run or load, default is default.
- `--run_name RUN_NAME`: Name of the run, default is current time %Y-%m-%d_%H-%M-%S.
- `--log_root`：Path of log_root, default is engineai_rl_workspace/logs/{exp_name}/{sub_exp_name}.
- `--load_run LOAD_RUN`: Name of the run to load when resume=True, default is -1. If -1`: will load the last run.
- `--checkpoint CHECKPOINT`: Saved model checkpoint number, default is -1. If -1`: will load the last checkpoint.
- `--num_envs NUM_ENVS`: Number of environments to create.
- `--seed SEED`: Random seed.
- `--sim_device SIM_DEVICE`: Device used by the simulator, (cpu, gpu, cuda:0, cuda:1 etc..), default is cuda:0.
- `--rl_device RL_DEVICE`: Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..), default is cuda:0.
- `--use_joystick`: Use joystick in play mode.
- `--joystick_scale`: Scale of joystick, only useful when use_joystick is True.
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
- `--test_length TEST_LENGTH`: Number of iteration for each tester, default is 500.

### Examples

#### Play with a Tester

```shell
# default setting
python engineai_rl_workspace/scripts/play.py --exp_name pm01_rough_ppo --load_run 2025-06-03_12-00-00
# custom test_length
python engineai_rl_workspace/scripts/play.py --exp_name pm01_rough_ppo --load_run 2025-06-03_12-00-00 --test_length 500
# from a custom log root
python engineai_rl_workspace/scripts/play.py --exp_name pm01_rough_ppo --log_root ~/server/engineai_rl_workspace/logs/pm01_rough_ppo/default --load_run 2025-06-03_12-00-00
# recording video
python engineai_rl_workspace/scripts/play.py --exp_name pm01_rough_ppo --load_run 2025-06-03_12-00-00 --video
# run with params overriden
python engineai_rl_workspace/scripts/play.py --exp_name pm01_rough_ppo --headless --num_envs 32 --seed 1
```

#### Play with Joystick

```shell
python engineai_rl_workspace/scripts/play.py --exp_name pm01_rough_ppo --load_run 2025-06-03_12-00-00 --use_joystick
```