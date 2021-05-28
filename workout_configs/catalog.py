from utils import read_cfg_file
import os


# TODO add _check_config (if_file_exist, if_all_conditions_have_same_length)
class WorkoutDataLoader:
    def __init__(self, config):
        self._set_config(config)

    def _set_config(self, config):
        cfg_dict = read_cfg_file(config)
        self.WINDOW_NAME = cfg_dict["WINDOW_NAME"]
        self.ANGLES = cfg_dict["ANGLES"]
        self.HIGH_LOW = cfg_dict["HIGH_LOW"]
        self.ERRORS = cfg_dict["ERRORS"]
        self.DESCRIPTION = cfg_dict["DESCRIPTION"]
        self.IMAGE_PATH = cfg_dict["IMAGE"]
        self.MET = cfg_dict["MET"]

    @staticmethod
    def _get_path(path):
        if __name__ == "__main__":
            folder = ""
        else:
            folder = "workout_configs"
        return os.path.join(folder, path)

    @classmethod
    def dumbbell_shoulder_press(cls):
        config = "dumbbell_shoulder_press.yaml"
        path = cls._get_path(config)
        return cls(path)

    @classmethod
    def dumbbell_front_raise(cls):
        config = "dumbbell_front_raise.yaml"
        path = cls._get_path(config)
        return cls(path)

    @classmethod
    def dumbbell_lateral_raise(cls):
        config = "dumbbell_lateral_raise.yaml"
        path = cls._get_path(config)
        return cls(path)

    @classmethod
    def dumbbell_bent_over_lateral_raise(cls):
        config = "dumbbell_bent_over_lateral_raise.yaml"
        path = cls._get_path(config)
        return cls(path)

    @classmethod
    def dumbbell_upright_row(cls):
        config = "dumbbell_upright_row.yaml"
        path = cls._get_path(config)
        return cls(path)


if __name__ == "__main__":
    w = WorkoutDataLoader.dumbbell_shoulder_press()
    print(w.ERRORS, type(w.ERRORS))
