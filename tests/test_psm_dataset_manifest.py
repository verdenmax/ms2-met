import json

import numpy as np
import pytest

from spectrum.psm_dataset_manifest import (
    MANIFEST_SCHEMA,
    manifest_path,
    validate_manifest,
)
from spectrum.psm_info import PSMInfo
from tools.extract_common import write_psms_to_json


def _psm(label_type="positive", modify=None):
    return PSMInfo(
        sequence="PEPTMDEK",
        charge=2,
        modify=modify or [],
        rt=np.float32(10),
        precursor_mz=np.float32(500),
        raw_title="run1",
        protein_names="P1",
        label_type=label_type,
    )


def test_writer_keeps_array_json_and_adds_chemistry_manifest(tmp_path):
    output = tmp_path / "psms.json"
    config = tmp_path / "extract.ini"
    config.write_text("[extract]\nlabeling = c13\n", encoding="utf-8")
    psms = [_psm(), _psm("negative", [(4, 35)])]

    write_psms_to_json(
        psms, str(output), labeling="c13",
        source_config_path=str(config),
    )

    assert isinstance(json.loads(output.read_text()), list)
    manifest = json.loads(
        (tmp_path / "psms.json.manifest.json").read_text())
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["labeling"] == "c13"
    assert manifest["isotope_model"] == "ideal_full_label_v1"
    assert manifest["dataset"]["n_psms"] == 2
    assert manifest["dataset"]["n_modified_psms"] == 1
    assert manifest["dataset"]["counts_by_label_type"] == {
        "negative": 1, "positive": 1,
    }
    assert len(manifest["source_config"]["sha256"]) == 64


def test_manifest_rejects_labeling_mismatch_and_dataset_mutation(tmp_path):
    output = tmp_path / "psms.json"
    write_psms_to_json([_psm()], str(output), labeling="c13")

    with pytest.raises(ValueError, match="标记类型.*不一致"):
        validate_manifest(str(output), "n15", require=True)

    output.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="摘要不一致"):
        validate_manifest(str(output), "c13", require=True)


def test_missing_manifest_is_only_allowed_for_legacy_silac(tmp_path, caplog):
    output = tmp_path / "legacy.json"
    output.write_text("[]\n", encoding="utf-8")
    assert validate_manifest(str(output), "silac", require=False) is None
    assert "legacy SILAC" in caplog.text
    with pytest.raises(ValueError, match="缺少 manifest"):
        validate_manifest(str(output), "c13", require=True)


def test_manifest_path_does_not_replace_json_extension():
    assert manifest_path("x.json") == "x.json.manifest.json"
