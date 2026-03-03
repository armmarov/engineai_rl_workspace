from engineai_gym.tester.loggers.logger_base import LoggerBase


class LoggerTypeAmpDiscriminator(LoggerBase):
    def log_data(self, extra_data):
        if "amp_disc_pred" in extra_data:
            self.add_data_to_log(
                {
                    "amp_disc_pred": extra_data["amp_disc_pred"][
                        self.extra_args["robot_index"]
                    ].item(),
                }
            )

    def retrieve_data(self):
        if "amp_disc_pred" in self.log:
            self.save_to_csv(
                "AMP Discriminator Prediction",
                [self.log["amp_disc_pred"]],
                "Discriminator Score",
                ["amp_disc_pred"],
                ["amp_disc_pred"],
            )
