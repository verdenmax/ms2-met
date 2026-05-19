"""Light tests for tools/eval_baseline internals (refactoring guards)."""
import inspect


def test_cv_evaluate_does_not_pass_sample_weight_to_fit():
    """class_weight='balanced' already balances; explicit sample_weight
    is redundant and double-counts."""
    from tools import eval_baseline
    src = inspect.getsource(eval_baseline.cv_evaluate)
    assert "sample_weight=" not in src, (
        "sample_weight passed to fit duplicates class_weight='balanced'")


def test_compute_feature_importance_does_not_pass_sample_weight_to_fit():
    from tools import eval_baseline
    if not hasattr(eval_baseline, "compute_feature_importance"):
        return
    src = inspect.getsource(eval_baseline.compute_feature_importance)
    assert "sample_weight=" not in src


def test_ablation_cv_one_does_not_pass_sample_weight_to_fit():
    from tools import eval_feature_ablation
    if not hasattr(eval_feature_ablation, "cv_one"):
        return
    src = inspect.getsource(eval_feature_ablation.cv_one)
    assert "sample_weight=" not in src


def test_load_features_does_not_fillna_zero():
    """fillna(0.0) conflates 'no data' with 'value=0'.
    HistGBT handles NaN natively; preserve it."""
    import inspect
    from tools import eval_baseline
    src = inspect.getsource(eval_baseline.load_features)
    assert ".fillna(0" not in src and ".fillna(0.0)" not in src, (
        "fillna(0) destroys 'no data' signal — let HistGBT handle NaN natively")
    assert "replace([np.inf, -np.inf]" in src or "inf" in src.lower()


def test_ablation_main_does_not_fillna_zero():
    """Same for tools/eval_feature_ablation.py."""
    import inspect
    from tools import eval_feature_ablation
    src = inspect.getsource(eval_feature_ablation)
    assert ".fillna(0" not in src and ".fillna(0.0)" not in src

