NUM_ENVS ?= 512
HEADLESS ?= --headless

# ============================================================
# Training
# ============================================================

train-flat:
	python3 engineai_rl_workspace/scripts/train.py --exp_name pm01_flat_ppo --num_envs $(NUM_ENVS) $(HEADLESS)

train-rough:
	python3 engineai_rl_workspace/scripts/train.py --exp_name pm01_rough_ppo --num_envs $(NUM_ENVS) $(HEADLESS)

train-body-block:
	python3 engineai_rl_workspace/scripts/train.py --exp_name pm01_body_block_ppo_amp --num_envs $(NUM_ENVS) $(HEADLESS)

# ============================================================
# Play / Test
# ============================================================

play-flat:
	python3 engineai_rl_workspace/scripts/play.py --exp_name pm01_flat_ppo

play-rough:
	python3 engineai_rl_workspace/scripts/play.py --exp_name pm01_rough_ppo

play-body-block:
	python3 engineai_rl_workspace/scripts/play.py --exp_name pm01_body_block_ppo_amp

# ============================================================
# Export Policy
# ============================================================

export-flat:
	python3 engineai_rl_workspace/scripts/export_policy.py --exp_name pm01_flat_ppo

export-rough:
	python3 engineai_rl_workspace/scripts/export_policy.py --exp_name pm01_rough_ppo

export-body-block:
	python3 engineai_rl_workspace/scripts/export_policy.py --exp_name pm01_body_block_ppo_amp

# ============================================================
# Verify Motion Data
# ============================================================

verify-motion:
	python3 engineai_gym/engineai_gym/resources/robots/biped/pm01/mocap_motions/verify_motion.py \
		engineai_gym/engineai_gym/resources/robots/biped/pm01/mocap_motions/body_block_pose.json

# ============================================================
# Help
# ============================================================

help:
	@echo "Usage:"
	@echo "  make train-flat              Train PM01 flat terrain (PPO)"
	@echo "  make train-rough             Train PM01 rough terrain (PPO)"
	@echo "  make train-body-block        Train PM01 body block (PPO-AMP)"
	@echo ""
	@echo "  make play-flat               Test flat terrain policy"
	@echo "  make play-rough              Test rough terrain policy"
	@echo "  make play-body-block         Test body block policy"
	@echo ""
	@echo "  make export-flat             Export flat policy to ONNX/MNN"
	@echo "  make export-rough            Export rough policy to ONNX/MNN"
	@echo "  make export-body-block       Export body block policy to ONNX/MNN"
	@echo ""
	@echo "  make verify-motion           Verify motion JSON data"
	@echo ""
	@echo "Options:"
	@echo "  NUM_ENVS=1024               Number of environments (default: 512)"
	@echo "  HEADLESS=                   Show GUI (default: --headless)"
	@echo ""
	@echo "Examples:"
	@echo "  make train-flat NUM_ENVS=256          Train with 256 envs"
	@echo "  make train-body-block HEADLESS=       Train with GUI visible"

.PHONY: train-flat train-rough train-body-block \
        play-flat play-rough play-body-block \
        export-flat export-rough export-body-block \
        verify-motion help
