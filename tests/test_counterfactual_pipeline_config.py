import configparser
from pathlib import Path

import pandas as pd

from tools.counterfactual_negatives import load_job as load_negative_job
from tools.counterfactual_parents import load_job as load_parent_job


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "config" / "counterfactual"
FEATURE_CONFIG = (
    ROOT / "runs" / "counterfactual_2da_label_dev_train" / "config.ini")
BASE_FEATURE_CONFIG = ROOT / "runs" / "baseline_2da_clean" / "config.ini"


def test_2da_counterfactual_configs_have_consistent_stage_handoffs():
    parent = load_parent_job(str(
        CONFIG_ROOT / "2da_label_dev_train.parents.ini"))
    negative = load_negative_job(str(
        CONFIG_ROOT / "2da_label_dev_train.negatives.ini"))
    feature = configparser.ConfigParser()
    assert feature.read(FEATURE_CONFIG)

    assert parent.output_psms == negative.parents
    assert negative.output_psms == feature["input"]["light_result_file"]
    assert parent.prepare.dataset_split == "label_dev_train"
    assert negative.build.dataset_split == "label_dev_train"
    assert negative.workers == 8
    assert negative.worker_chunk_size == 128
    assert negative.max_parents == 5000
    assert negative.target_fasta == str(
        Path.home()
        / "share/2026_06_07_kongweisa_guangshan_puku"
        / "uniprotkb_proteome_UP000005640_2026_06_10.fasta")


def test_counterfactual_feature_config_inherits_2da_config_ini():
    baseline = configparser.ConfigParser()
    feature = configparser.ConfigParser()
    assert baseline.read(BASE_FEATURE_CONFIG)
    assert feature.read(FEATURE_CONFIG)

    for key, value in baseline["input"].items():
        if key != "light_result_file":
            assert feature["input"][key] == value
    for key, value in baseline["general"].items():
        if key != "result_file":
            assert feature["general"][key] == value
    assert dict(feature["speclib"]) == dict(baseline["speclib"])


def test_2da_split_keeps_all_windows_from_each_rep_together():
    split = pd.read_csv(CONFIG_ROOT / "2da_raw_split.csv")
    feature = configparser.ConfigParser()
    assert feature.read(FEATURE_CONFIG)

    assert not split["raw_title"].duplicated().any()
    assert split.groupby(
        split["raw_title"].str.extract(r"(Rep\d+)$", expand=False)
    )["dataset_split"].nunique().eq(1).all()
    assert split["dataset_split"].value_counts().to_dict() == {
        "label_dev_train": 9,
    }

    configured_raws = {
        Path(feature["input"][f"raw_path_{index}"]).stem
        for index in range(1, feature.getint("input", "raw_num") + 1)
    }
    expected_training_raws = set(split.loc[
        split["dataset_split"] == "label_dev_train", "raw_title"])
    assert configured_raws == expected_training_raws
