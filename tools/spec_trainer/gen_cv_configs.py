"""Generate cv_*.yaml variants from the existing in_*/cross_test_* configs.

Each cv variant adds CV keys (group_col / cv_folds / cv_seed / inner valid_size
/ audit) and rewrites output paths to cv-specific files (so CV results never
collide with the single-holdout main.py outputs). Run once and commit the
generated files; re-run to resync after the source configs change. Idempotent.
"""
import copy
import glob
import os

import yaml

_CFG_DIR = os.path.join(os.path.dirname(__file__), "config")


def to_cv_config(src, name):
    """Transform an in_/cross_test source config dict into its cv_ variant.

    name: source basename (e.g. 'in_2da_clean'); the variant is 'cv_' + name.
    Does NOT mutate src (deep-copies). train_files/test_files/model/target_col
    are preserved; CV keys + cv output paths are set; figures_dir is dropped.
    """
    cfg = copy.deepcopy(src)
    cfg["data"]["group_col"] = "sequence"
    cfg["training"]["cv_folds"] = 5
    cfg["training"]["cv_seed"] = 42
    cfg["training"]["valid_size"] = 0.15
    cfg["audit"] = {"suspect_threshold": 0.9, "suspect_top_n": 200}
    cv_name = "cv_" + name
    cfg["output"] = {
        "model_path": f"runs/spec_trainer/models/{cv_name}.txt",
        "result_path": f"runs/spec_trainer/results/{cv_name}.cv.json",
    }
    return cfg


def main():
    srcs = sorted(glob.glob(os.path.join(_CFG_DIR, "in_*.yaml"))
                  + glob.glob(os.path.join(_CFG_DIR, "cross_test_*.yaml")))
    for path in srcs:
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            src = yaml.safe_load(f)
        cv = to_cv_config(src, name)
        out = os.path.join(_CFG_DIR, f"cv_{name}.yaml")
        with open(out, "w") as f:
            yaml.safe_dump(cv, f, sort_keys=False, allow_unicode=True)
    print(f"generated {len(srcs)} cv configs in {_CFG_DIR}")


if __name__ == "__main__":
    main()
