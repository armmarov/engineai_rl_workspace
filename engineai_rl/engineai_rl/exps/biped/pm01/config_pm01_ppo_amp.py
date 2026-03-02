from engineai_rl.algos.ppo.ppo_amp.config_ppo_amp import ConfigPpoAmp


class ConfigPm01PpoAmp(ConfigPpoAmp):
    class params(ConfigPpoAmp.params):
        entropy_coef = 0.01
        learning_rate = 5e-5
        num_learning_epochs = 5
        gamma = 0.99
        lam = 0.95
        amp_reward_coef = 2.0
        amp_task_reward_lerp = 0.3
        min_normalized_std = [0.01, 0.01, 0.01] * 8  # 24 joints

    class runner(ConfigPpoAmp.runner):
        seed = 5
        num_steps_per_env = 24
        max_iterations = 50000

    class networks(ConfigPpoAmp.networks):
        class critic(ConfigPpoAmp.networks.critic):
            hidden_dims = [768, 256, 128]

        class discriminator(ConfigPpoAmp.networks.discriminator):
            hidden_dims = [1024, 512]

    class input(ConfigPpoAmp.input):
        class components(ConfigPpoAmp.input.components):
            goal_list = ["commands"]

            class actor(ConfigPpoAmp.input.components.actor):
                obs_list = [
                    "projected_gravity",
                    "dof_pos",
                    "dof_vel",
                    "actions",
                ]
                obs_history_length = 1
                obs_with_goals = True

            class critic(ConfigPpoAmp.input.components.critic):
                obs_list = [
                    "base_lin_vel",
                    "base_ang_vel",
                    "projected_gravity",
                    "dof_pos",
                    "dof_vel",
                    "actions",
                ]
                obs_history_length = 1
                obs_with_goals = True

            class amp(ConfigPpoAmp.input.components.amp):
                obs_list = [
                    "absolute_dof_pos",
                    "foot_pos",
                    "base_lin_vel",
                    "base_ang_vel",
                    "dof_vel",
                    "base_z_pos",
                ]

        class obs_noise(ConfigPpoAmp.input.obs_noise):
            noise_level = 1.0

            class scales(ConfigPpoAmp.input.obs_noise.scales):
                actor = {
                    "projected_gravity": 0.05,
                    "dof_pos": 0.02,
                    "dof_vel": 1.5,
                }
                critic = {
                    "base_lin_vel": 0.1,
                    "base_ang_vel": 0.3,
                    "projected_gravity": 0.05,
                    "dof_pos": 0.02,
                    "dof_vel": 1.5,
                }
