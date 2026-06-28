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
