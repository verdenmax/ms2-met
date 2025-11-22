class ConstantsClass(type):

    def __setattr__(self, name, value):
        raise TypeError("Constants class cannot be modified")

    def get_values(cls):
        """Get all user-defined string values of the class."""
        return [
            value
            for key, value in cls.__dict__.items()
            if not key.startswith("__") and isinstance(value, str)
        ]


class ConfigKeys(metaclass=ConstantsClass):
    """String constants for accessing the config."""

    INPUT = "input"
    RAW_NUM = "raw_num"
    RAW_PATH = "raw_path"
    LIGHT_RESULT_PATH = "light_result_file"
    SEARCH_ENGINE_TYPE = "search_engine_type"

    GENERAL = "general"
    WORK_DIRECTORY = "work_directory"
    MASS_TOL_PPM = "mass_tol_ppm"
    XIC_CYCLE_WINDOW = "xic_cycle_window"
