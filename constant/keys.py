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

    # pfind 特有配置
    PFIND_QVALUE_THRESHOLD = "pfind_qvalue_threshold"

    GENERAL = "general"
    WORK_DIRECTORY = "work_directory"
    MASS_TOL_PPM = "mass_tol_ppm"
    XIC_CYCLE_WINDOW = "xic_cycle_window"
    RESULT_FILE = "result_file"
    FEATURE_TYPE = "feature_type"
    RANDOM_SEED = "random_seed"

    # mzML centroiding (loaded by manager/data_manager.py)
    CENTROID_ENABLED = "centroid_enabled"
    CENTROID_REL_THRESHOLD = "centroid_rel_threshold"

    # speclib predicted-intensity features (Phase 2)
    SPECLIB = "speclib"
    SPECLIB_DIR = "speclib_dir"
    SPECLIB_FASTA = "speclib_fasta"
    SPECLIB_MOD = "speclib_mod"
    PRED_TOP_K = "pred_top_k"
