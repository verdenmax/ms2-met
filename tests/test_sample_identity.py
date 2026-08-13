import hashlib

import pandas as pd

from tools.spec_trainer.src.sample_identity import (
    combined_sample_id, identity_candidates, identity_text, local_sample_ids,
    namespace_sample_ids,
)


IDENTITY_COLUMNS = [
    "sequence", "charge", "precursor_mz", "rt", "raw_title1",
    "label_type",
]


def test_public_identity_contract_pins_existing_serialization_and_hashing():
    frame = pd.DataFrame([{
        "sequence": "PEPTIDE", "charge": "2", "precursor_mz": "500.0",
        "rt": "10.0", "raw_title1": "raw1", "label_type": "positive",
    }])
    serialized = "7:PEPTIDE|1:2|5:500.0|4:10.0|4:raw1|8:positive|"
    assert identity_text(frame, IDENTITY_COLUMNS).iloc[0] == serialized
    local = local_sample_ids(frame, IDENTITY_COLUMNS).iloc[0]
    assert local == hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    assert combined_sample_id("2da", local) == hashlib.sha256(
        f"2da|{local}".encode("utf-8")).hexdigest()


def test_namespace_preserves_local_id_and_does_not_mutate_input():
    original = pd.DataFrame({"sample_id": ["local-a"]})
    namespaced = namespace_sample_ids(original, "normal")
    assert original.columns.tolist() == ["sample_id"]
    assert namespaced.loc[0, "source_sample_id"] == "local-a"
    assert namespaced.loc[0, "sample_id"] == combined_sample_id(
        "normal", "local-a")
    assert namespaced.loc[0, "dataset"] == "normal"


def test_identity_candidate_preference_remains_frozen():
    columns = {*IDENTITY_COLUMNS, "query_id", "protein_names", "q_value"}
    candidates = identity_candidates(columns)
    assert candidates[0] == ["query_id"]
    assert candidates[1] == IDENTITY_COLUMNS
    assert candidates[-1] == IDENTITY_COLUMNS + ["protein_names", "q_value"]
