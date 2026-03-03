# Tutorials

## Export Policy

### Args

- `--exp_name EXP_NAME`: Experiment name.
- `--sub_exp_name SUB_EXP_NAME`: Name of the sub-experiment to run or load, default is default.
- `--run_name RUN_NAME`: Name of the run, default is current time %Y-%m-%d_%H-%M-%S.
- `--log_root`：Path of log_root, default is engineai_rl_workspace/logs/{exp_name}/{sub_exp_name}.
- `--load_run LOAD_RUN`: Name of the run to load when resume=True. If -1: will load the last run.
- `--checkpoint CHECKPOINT`: Saved model checkpoint number. If -1: will load the last checkpoint.

### Examples

```
# basic
python engineai_rl_workspace/scripts/export_policy.py --exp_name pm01_rough_ppo --load_run 2025-06-03_12-00-00
# specific checkpoint
python engineai_rl_workspace/scripts/export_policy.py --exp_name pm01_rough_ppo --load_run 2025-06-03_12-00-00 --checkpoint 10000
# from a custom log root
python engineai_rl_workspace/scripts/export_policy.py --exp_name pm01_rough_ppo --log_root ~/server/engineai_rl_workspace/logs/pm01_rough_ppo/default --load_run 2025-06-03_12-00-00
```

