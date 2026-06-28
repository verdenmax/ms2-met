import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "tools", "spec_trainer", "src"))


def test_make_cv_splits_grouped_no_leak_full_cover():
    from cv_core import make_cv_splits
    # 10 个肽段(group), 每个 2 行; group 与 label 一致(一条肽段同一类)
    groups = np.repeat(np.arange(10), 2)            # 0,0,1,1,...,9,9
    y = np.where(groups < 6, 1, 0)                  # 前 6 组正, 后 4 组负
    splits = make_cv_splits(y, groups, n_folds=5, seed=42)
    assert len(splits) == 5
    covered = np.concatenate([te for _, te in splits])
    assert sorted(covered.tolist()) == list(range(len(y)))   # 每行恰好测一次
    for tr, te in splits:
        assert set(groups[tr]).isdisjoint(set(groups[te]))   # 同肽段不跨 train/test


def test_make_cv_splits_no_groups_fallback():
    from cv_core import make_cv_splits
    y = np.array([1] * 16 + [0] * 4)
    splits = make_cv_splits(y, groups=None, n_folds=5, seed=42)
    assert len(splits) == 5
    covered = np.concatenate([te for _, te in splits])
    assert sorted(covered.tolist()) == list(range(len(y)))


def test_working_points_and_fnr_clean_separation():
    from cv_core import working_points, fnr_at_fpr5
    neg = np.linspace(0.0, 0.99, 100)      # 负例分数 0..0.99
    pos = np.ones(100)                     # 正例分数全 1.0(完全高于负例)
    y = np.r_[np.ones(100), np.zeros(100)]
    s = np.r_[pos, neg]
    wp = working_points(y, s)
    # 阈值 = 负例 95 分位 (<1.0); 正例全部 >= 阈值 -> pos_recall=1
    assert wp["neg_recall_95"]["pos_recall"] == 1.0
    assert 0.93 <= wp["neg_recall_95"]["neg_recall"] <= 0.96
    assert fnr_at_fpr5(y, s) == 0.0
    assert set(wp) == {"neg_recall_95", "neg_recall_90", "neg_recall_80"}


def test_working_points_matches_eval_baseline():
    import numpy as np, pytest
    compute_working_points = pytest.importorskip("tools.eval_baseline").compute_working_points
    from cv_core import working_points
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.random(200)
    assert working_points(y, s) == compute_working_points(y, s)


def test_evaluate_oof_perfect_and_reversed():
    from cv_core import evaluate_oof
    y = np.r_[np.ones(50), np.zeros(50)]
    perfect = np.r_[np.full(50, 0.9), np.full(50, 0.1)]   # 正高负低
    m = evaluate_oof(y, perfect)
    assert m["auc"] == 1.0
    assert m["fnr_at_fpr5"] == 0.0
    assert "neg_recall_95" in m["working_points"]
    m2 = evaluate_oof(y, 1.0 - perfect)                   # 完全反向
    assert m2["auc"] == 0.0
