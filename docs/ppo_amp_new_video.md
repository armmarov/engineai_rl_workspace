# Training with PPO-AMP Using a New Video

This guide explains how to use the PPO-AMP (Adversarial Motion Priors) pipeline
to train a robot policy that imitates motion extracted from a video.

---

## What is PPO-AMP?

Standard PPO trains the robot purely from reward functions (e.g. track velocity,
stay upright). The robot learns to walk but the movement can look unnatural.

PPO-AMP adds a **discriminator network** that has seen expert motion data. It
rewards the robot for producing transitions that look like the expert, blended
with the task reward:

```
total_reward = 0.3 * task_reward + 0.7 * amp_reward
```

The discriminator is trained simultaneously to tell apart:
- **Expert** transitions (from your motion data) → target score +1
- **Policy** transitions (what the robot actually does) → target score -1

Over time the robot learns to move like the expert while still achieving the task.

---

## Architecture Overview

```
Video
  └─► Pose Estimation (WHAM / HMR2)
        └─► SMPL pose sequence
              └─► Motion Retargeting
                    └─► Robot joint angles per frame
                          └─► your_motion.json
                                └─► RefStateLoader
                                      └─► PPO-AMP Training
                                            ├─► Actor (policy)
                                            ├─► Critic (value)
                                            └─► Discriminator (AMP)
                                                  └─► Export ONNX / MNN
                                                        └─► Deploy on robot
```

---

## Key Components

| Component | Location | Purpose |
|---|---|---|
| `PpoAmp` | `engineai_rl/algos/ppo/ppo_amp/ppo_amp.py` | Main AMP algorithm |
| `AmpDiscriminator` | `engineai_rl/algos/ppo/ppo_amp/amp_discriminator.py` | Discriminator network |
| `ConfigPpoAmp` | `engineai_rl/algos/ppo/ppo_amp/config_ppo_amp.py` | AMP hyperparameters |
| `RefStateLoader` | `engineai_rl_lib/ref_state/ref_state_loader.py` | Loads motion JSON files |
| `LeggedRobotRef` | `engineai_gym/envs/base/legged_robot_ref.py` | Env with reference state resets |
| Motion files | `engineai_gym/resources/robots/.../mocap_motions/` | JSON motion data |

---

## Motion Data Format

Each motion is a JSON file with this structure:

```json
{
  "LoopMode": "Wrap",
  "FrameDuration": 0.01677,
  "EnableCycleOffsetPosition": true,
  "EnableCycleOffsetRotation": true,
  "MotionWeight": 1,
  "Frames": [
    [f0, f1, f2, ...],
    ...
  ]
}
```

Each frame contains **61 floats**:

| Index | Size | Field |
|---|---|---|
| 0:3 | 3 | `root_pos` (x, y, z) |
| 3:7 | 4 | `root_rot` (quaternion: qx, qy, qz, qw) |
| 7:19 | 12 | `dof_pos` (joint angles) |
| 19:31 | 12 | `foot_pos_local` (4 feet × 3) |
| 31:34 | 3 | `root_lin_vel` |
| 34:37 | 3 | `root_ang_vel` |
| 37:49 | 12 | `dof_vel` (joint velocities) |
| 49:61 | 12 | `foot_vel_local` (4 feet × 3) |

`MotionWeight` controls sampling probability when multiple motion files are used.
Higher weight = sampled more often during training.

The AMP observations used by the discriminator are:

```python
obs_list = [
    "absolute_dof_pos",
    "foot_pos",
    "base_lin_vel",
    "base_ang_vel",
    "dof_vel",
    "base_z_pos",
]
```

---

## Step-by-Step Guide

### Step 1: Extract Pose from Video

Use a pose estimation tool to get 3D joint data from your video.

**Recommended tools:**

| Tool | Link | Notes |
|---|---|---|
| WHAM | https://github.com/yohanshin/WHAM | Best for full body + root motion |
| HMR2 / 4D-Humans | https://github.com/shubham-goel/4D-Humans | Strong per-frame accuracy |
| MotionBERT | https://github.com/Walter0807/MotionBERT | Good for 3D pose from 2D video |

```bash
# Example: WHAM
git clone https://github.com/yohanshin/WHAM
cd WHAM
python demo.py --video /path/to/your_video.mp4 --output_path ./output
```

Output: SMPL pose parameters (body shape + joint angles) + root translation per frame.

---

### Step 2: Retarget Motion to Robot Skeleton

Human skeleton proportions differ from the robot. Retargeting maps human joint
angles to the robot's joint space while respecting joint limits.

**What to do:**
1. Load SMPL output (body pose per frame)
2. Map relevant joints to robot DOFs:
   - Human hip → robot hip (roll/yaw/pitch)
   - Human knee → robot knee
   - Human ankle → robot ankle inner/outer
3. Scale and clip to robot joint limits defined in the URDF

**Existing motion files for reference:**
```
engineai_gym/resources/robots/quadruped/a1/mocap_motions/
├── trot0.json
├── trot1.json
├── pace0.json
├── canter0.json
└── ...
```

Study these files to understand the expected value ranges before writing your
conversion script.

---

### Step 3: Convert to AMP JSON Format

A CSV-to-JSON conversion utility is provided:

```
engineai_rl_lib/engineai_rl_lib/convert_mocap_files_from_csv2json.py
```

**CSV format expected:**
- Line 1: `frame_duration` (e.g. `0.01677`)
- Remaining lines: comma-separated floats, one frame per line (61 values)

Run conversion:

```bash
python engineai_rl_lib/engineai_rl_lib/convert_mocap_files_from_csv2json.py \
    --input your_motion.csv \
    --output your_motion.json
```

Place the output JSON in your robot's motion folder:

```
engineai_gym/resources/robots/biped/sa01/motions/your_motion.json
```

---

### Step 4: Create Data Mapping Class

Create a `data.py` for your robot (copy from A1 and adjust indices if DOF count differs):

```python
# engineai_gym/resources/robots/biped/sa01/motions/data.py
from engineai_rl_lib.ref_state.data_base import DataBase
from engineai_rl_lib.ref_state import ref_state_util
from engineai_rl_lib.utils import pose3d
from scipy.spatial.transform import Rotation, Slerp

class Data(DataBase):
    def component_root_pos(self, trajectory):
        return trajectory[:, :3]

    def component_root_rot(self, trajectory):
        root_rot = trajectory[:, 3:7]
        root_rot = pose3d.quaternion_normalize(root_rot)
        root_rot = ref_state_util.standardize_quaternion(root_rot)
        return root_rot

    def component_dof_pos(self, trajectory):
        return trajectory[:, 7:19]

    def component_foot_pos_local(self, trajectory):
        return trajectory[:, 19:31]

    def component_root_lin_vel(self, trajectory):
        return trajectory[:, 31:34]

    def component_root_ang_vel(self, trajectory):
        return trajectory[:, 34:37]

    def component_dof_vel(self, trajectory):
        return trajectory[:, 37:49]

    def component_foot_vel_local(self, trajectory):
        return trajectory[:, 49:61]

    def blend_root_rot(self, frame_start, frame_end, blend):
        # SLERP interpolation for quaternions
        key_rots = Rotation.from_quat(
            [frame_start.cpu().numpy(), frame_end.cpu().numpy()]
        )
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        import torch
        result = torch.tensor(slerp([blend]).as_quat()[0], dtype=torch.float32)
        return ref_state_util.standardize_quaternion(result)
```

---

### Step 5: Create Environment Config

```python
# engineai_gym/envs/robots/biped/sa01/config_sa01_flat_ref_state.py
from engineai_gym.envs.robots.biped.sa01.config_sa01_flat import ConfigSA01Flat

class ConfigSA01FlatRefState(ConfigSA01Flat):
    class ref_state:
        ref_state_loader = True
        ref_state_init = True
        motion_files_path = (
            "{ENGINEAI_GYM_PACKAGE_DIR}/resources/robots/biped/sa01/motions"
        )
        ref_state_init_prob = 0.85  # 85% of episode resets use reference motion

        data_mapping = {
            "base_pos":          "root_pos",
            "base_z_pos":        "root_z_pos",
            "base_rot":          "root_rot",
            "absolute_dof_pos":  "dof_pos",
            "dof_vel":           "dof_vel",
            "foot_pos":          "foot_pos_local",
            "base_lin_vel":      "root_lin_vel",
            "base_ang_vel":      "root_ang_vel",
        }
```

---

### Step 6: Create Training Config

```python
# engineai_rl/exps/biped/sa01/config_sa01_ppo_amp.py
from engineai_rl.algos.ppo.ppo_amp.config_ppo_amp import ConfigPpoAmp

class ConfigSA01PpoAmp(ConfigPpoAmp):
    class params(ConfigPpoAmp.params):
        entropy_coef = 0.01
        amp_reward_coef = 2.0        # Scale of AMP reward
        amp_task_reward_lerp = 0.3   # 30% task, 70% AMP

    class runner(ConfigPpoAmp.runner):
        max_iterations = 50000
```

---

### Step 7: Register the Experiment

```python
# engineai_rl_workspace/exps/biped/sa01.py
from engineai_gym.envs.base.legged_robot_ref import LeggedRobotRef
from engineai_rl.algos.ppo.ppo_amp.ppo_amp import PpoAmp
from engineai_gym.envs.robots.biped.sa01.config_sa01_flat_ref_state import ConfigSA01FlatRefState
from engineai_rl.exps.biped.sa01.config_sa01_ppo_amp import ConfigSA01PpoAmp
from engineai_rl_workspace.utils.exp_registry import exp_registry

exp_registry.register(
    name="sa01_flat_ppo_amp",
    task_class=LeggedRobotRef,
    env_cfg=ConfigSA01FlatRefState(),
    algo_class=PpoAmp,
    algo_cfg=ConfigSA01PpoAmp(),
)
```

---

### Step 8: Run Training

```bash
cd /path/to/engineai_rl_workspace
python engineai_rl_workspace/scripts/train.py --config sa01_flat_ppo_amp
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir logs/
```

Or with WandB (set in config):

```bash
wandb login
# then run train.py as normal
```

---

## Key Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `amp_reward_coef` | 2.0 | Scales discriminator reward magnitude |
| `amp_task_reward_lerp` | 0.3 | 0.0 = pure AMP, 1.0 = pure task reward |
| `amp_replay_buffer_size` | 100000 | Policy transition buffer size |
| `preload_batches` | True | Cache expert data in RAM for faster training |
| `num_preload_batches` | 2000000 | How many batches to preload |
| `ref_state_init_prob` | 0.85 | Fraction of episode resets from reference motion |
| `max_iterations` | 50000 | Total training iterations |

---

## Troubleshooting

**Discriminator loss not decreasing**
- Check that motion JSON values are in the correct ranges
- Verify `data_mapping` keys match observation names in the env

**Robot falls immediately**
- Lower `amp_task_reward_lerp` closer to 0 to reduce task reward pressure early
- Increase `ref_state_init_prob` so more resets start from valid poses

**Motion looks nothing like the reference**
- Check retargeting — joint angle signs and axis conventions may differ between
  SMPL and the robot URDF
- Visualize the JSON motion in MuJoCo before training

**Device mismatch errors**
- See fix in `rsl_rl/algorithms/ppo.py` — permutation matrices must use
  `.to(self.device)` not `.cuda()`

---

## Extending to Full Body (Legs + Arms)

The existing AMP setup covers **legs only** (12 DOF). To extend to full body:

1. Update `data.py` to include arm joint indices in `component_dof_pos`
2. Expand `obs_list` in `ConfigPpoAmp.input.components.amp` to include arm observations
3. Update the robot URDF to include arm joints
4. Retarget arm motion from SMPL upper body joints
5. Adjust `min_normalized_std` in config to match total DOF count

Full-body whole-body training is significantly harder to stabilize. The recommended
approach is to train a locomotion policy first, then train a separate upper-body
policy and combine them with a whole-body controller.
