from constant.keys import ConfigKeys


def test_speclib_config_keys_exist():
    assert ConfigKeys.SPECLIB == "speclib"
    assert ConfigKeys.SPECLIB_DIR == "speclib_dir"
    assert ConfigKeys.SPECLIB_FASTA == "speclib_fasta"
    assert ConfigKeys.SPECLIB_MOD == "speclib_mod"
    assert ConfigKeys.PRED_TOP_K == "pred_top_k"
