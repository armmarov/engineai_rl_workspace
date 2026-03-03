from engineai_gym.tester.loggers.logger_base import LoggerBase


UPPER_BODY_KEYWORDS = ["waist", "shoulder", "elbow", "head"]


class LoggerTypeUpperBodyPosition(LoggerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upper_body_indices = []
        self.upper_body_names = []
        for idx, name in enumerate(self.env.dof_names):
            if any(keyword in name for keyword in UPPER_BODY_KEYWORDS):
                self.upper_body_indices.append(idx)
                self.upper_body_names.append(name)

    def log_data(self, extra_data):
        for i, idx in enumerate(self.upper_body_indices):
            self.add_data_to_log(
                {
                    f"upper_body_pos[{self.upper_body_names[i]}]": self.env.dof_pos[
                        self.extra_args["robot_index"], idx
                    ].item(),
                }
            )

    def retrieve_data(self):
        for i, name in enumerate(self.upper_body_names):
            self.save_to_csv(
                f"Upper Body Position[{name}]",
                [self.log[f"upper_body_pos[{name}]"]],
                "Position [rad]",
                [f"upper_body_pos[{name}]"],
                ["measured"],
            )
