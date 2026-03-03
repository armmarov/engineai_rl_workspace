from engineai_gym.tester.loggers.logger_base import LoggerBase


class LoggerTypeAmpReward(LoggerBase):
    def log_data(self, extra_data):
        if "amp_reward" in extra_data:
            self.add_data_to_log(
                {
                    "amp_reward": extra_data["amp_reward"][
                        self.extra_args["robot_index"]
                    ].item(),
                }
            )

    def retrieve_data(self):
        if "amp_reward" in self.log:
            self.save_to_csv(
                "AMP Reward",
                [self.log["amp_reward"]],
                "Reward",
                ["amp_reward"],
                ["amp_reward"],
            )
