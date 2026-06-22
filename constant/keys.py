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
    # 轻标【碎片】峰形特征开关（默认 on）。off 时跳过 all_light_* 碎片缺陷列，
    # 便于消融。母离子轻标形状不受此开关影响。见
    # docs/superpowers/specs/2026-06-20-xic-apex-shape-penalty-design.md
    LIGHT_FRAGMENT_SHAPE = "light_fragment_shape"
    # Drop PSMs whose heavy precursor fell outside the acquisition range
    # (heavy_out_of_range == 1) for BOTH classes; default on. See
    # workflows/feature_postfilter.py.
    FILTER_HEAVY_OUT_OF_RANGE = "filter_heavy_out_of_range"

    # mzML centroiding (loaded by manager/data_manager.py)
    CENTROID_ENABLED = "centroid_enabled"
    CENTROID_REL_THRESHOLD = "centroid_rel_threshold"

    # speclib predicted-intensity features (Phase 2)
    SPECLIB = "speclib"
    SPECLIB_DIR = "speclib_dir"
    SPECLIB_FASTA = "speclib_fasta"
    SPECLIB_MOD = "speclib_mod"
    PRED_TOP_K = "pred_top_k"
    PRED_PRESENCE_FLOOR = "pred_presence_floor"
    PRED_SIGNAL_ALPHA = "pred_signal_alpha"
