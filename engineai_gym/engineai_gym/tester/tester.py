import numpy as np
import os
import yaml
from engineai_gym import ENGINEAI_GYM_ROOT_DIR
from engineai_gym.tester.testers.tester_type_base import TesterTypeBase
from engineai_rl_lib.class_operations import (
    instance_name_to_class_name,
    add_space_to_class_name,
)
from engineai_gym.wrapper.record_video_wrapper import RecordVideoWrapper
from engineai_rl_lib.files_and_dirs import import_modules_of_specific_type_from_path


current_file_directory = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "testers"
)
imported_classes = import_modules_of_specific_type_from_path(
    ENGINEAI_GYM_ROOT_DIR, current_file_directory, TesterTypeBase
)


class Tester:
    def __init__(
        self,
        env,
        length,
        dt,
        save_path,
        tester_config_path,
        record_video=False,
        extra_args=None,
    ):
        self.env = env
        self.record_video = record_video
        if record_video:
            self.env_supports_video_recording = isinstance(env, RecordVideoWrapper)
            if self.env_supports_video_recording:
                if self.env.record_length > length:
                    raise ValueError(
                        "Video record length must be less than the tester length!"
                    )
            else:
                raise ValueError("Env doesn't support video recording!")
        self.length = length
        time = np.linspace(0, length * dt, length)
        self.save_path = save_path
        with open(tester_config_path) as file:
            self.tester_config = yaml.safe_load(file)
        if extra_args is None:
            extra_args = {}
        self.testers = self._get_all_testers(env, time, save_path, extra_args)
        self.tester_names = list(self.testers.keys())

    def _get_all_testers(self, env, time, save_path, extra_args):
        testers = {}
        for key, value in self.tester_config["testers"].items():
            loggers = {}
            for logger in value["loggers"]:
                loggers[logger.replace("logger_type_", "")] = logger
            tester_class = imported_classes[key]
            name = add_space_to_class_name(instance_name_to_class_name(key))
            testers[key] = tester_class(
                name, loggers, env, time, os.path.join(save_path, "data"), extra_args
            )
        return testers

    def retrieve_data_from_loggers(self, idx):
        tester = self.get_current_tester(idx)
        if (idx + 1) % self.length == 0:
            for logger in tester.loggers:
                logger.retrieve_data()
                logger.log.clear()

    def add_data_for_testers_to_log(self, idx, extra_data):
        tester = self.get_current_tester(idx)
        for logger in tester.loggers:
            logger.log_data(extra_data)

    def set_goals(self, idx):
        return self.get_current_tester(idx).set_goals()

    def get_current_tester(self, idx):
        if idx // self.length < self.num_testers:
            return self.testers[self.tester_names[idx // self.length]]
        else:
            raise RuntimeError("tester is not found!")

    def process_record_video(self, idx):
        tester = self.get_current_tester(idx)
        if idx % self.length == 0:
            tester.start_record_video()
        elif (idx + 1) % self.env.record_length == 0 and self.env.is_recording_video:
            tester.end_and_save_recording_video()

    def step(self, idx, extra_data):
        if idx % self.length == 0:
            print(f"\nStart tester: {self.tester_names[idx // self.length]}")
        self.add_data_for_testers_to_log(idx, extra_data)
        self.retrieve_data_from_loggers(idx)
        if self.record_video:
            self.process_record_video(idx)

    @property
    def num_testers(self):
        return len(self.testers)
