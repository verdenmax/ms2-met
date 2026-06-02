# spec_trainer Runnable Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `make train-exp1` / `make train-exp2` 在 ms2-met 流水线上能跑通，按 `docs/specs/2026-06-02-spec-trainer-runnable-design.md` 实施 3 个小补丁：(A) 3 个 baseline config.ini 补 centroid 字段；(B) main.py 加 feature_cols 自动检测；(C) exp1/exp2.yaml feature_cols 留空。

**Architecture:** A 是纯 ini 文件追加，不动其他行；B 在 main.py 加 ~15 行自动检测逻辑（提取为可独立测试的 helper function `_resolve_feature_cols`），新增 3 个 pytest 单元测试；C 简单删除 yaml 中显式 feature_cols 列表，留空 `[]`。

**Tech Stack:** Python 3.13, configparser, pyyaml, pytest, conda env `jianyan`.

---

## File Structure

**Modified files:**
- `runs/baseline_2da_clean/config.ini` — append centroid fields at end of [general] section
- `runs/baseline_5da_clean/config.ini` — same
- `runs/baseline_normal_clean/config.ini` — same
- `tools/spec_trainer/src/main.py` — add `_resolve_feature_cols` helper + call site change
- `tools/spec_trainer/config/exp1.yaml` — feature_cols 改为空
- `tools/spec_trainer/config/exp2.yaml` — feature_cols 改为空

**New test file:**
- `tests/test_spec_trainer_main.py` — 3 unit tests for `_resolve_feature_cols`

(plan body continues in subsequent edits)

---

## Task 1: 3 个 baseline config.ini 补 centroid 字段

**Files:**
- Modify: `runs/baseline_2da_clean/config.ini`
- Modify: `runs/baseline_5da_clean/config.ini`
- Modify: `runs/baseline_normal_clean/config.ini`

- [ ] **Step 1: 验证当前缺 centroid 字段**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
for f in runs/baseline_2da_clean/config.ini runs/baseline_5da_clean/config.ini runs/baseline_normal_clean/config.ini; do
    echo "=== $f ==="
    grep -E "centroid_enabled|centroid_rel_threshold" "$f" || echo "(no centroid fields)"
done
```

Expected: 3 个 `(no centroid fields)` 行。

- [ ] **Step 2: 用 edit 工具在每个文件末尾追加 centroid 字段**

对每个 baseline config.ini，找到文件末尾的最后一行（比如 `result_file = ./runs/baseline_2da_clean/features.csv`），用 edit 工具替换 — old_str 是该行，new_str 是该行 + 空行 + centroid 字段块。

例如对 baseline_2da_clean/config.ini：

**old_str:**
```
result_file = ./runs/baseline_2da_clean/features.csv
```

**new_str:**
```
result_file = ./runs/baseline_2da_clean/features.csv

# 加载 mzML 时是否对 profile 谱图做 centroiding。
# 设为 false 退回旧行为（保留 profile，所有点）。
centroid_enabled = true

# centroid 阈值：单张谱图内 intensity < max * 该比值 的局部极大值丢弃。
# 典型范围 1e-4 ~ 1e-2；推荐 1e-3。
centroid_rel_threshold = 0.001
```

3 个文件各自的 result_file 路径不同（2da/5da/normal），按各自实际路径替换。

- [ ] **Step 3: 验证字段加入且值正确**

把以下 Python 写入临时脚本 `/tmp/verify_centroid.py`，然后运行：

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
python3 /tmp/verify_centroid.py
```

`/tmp/verify_centroid.py` 内容：

```python
import configparser
for f in [
    "runs/baseline_2da_clean/config.ini",
    "runs/baseline_5da_clean/config.ini",
    "runs/baseline_normal_clean/config.ini",
]:
    c = configparser.ConfigParser()
    c.read(f)
    e = c["general"].getboolean("centroid_enabled")
    t = c["general"].getfloat("centroid_rel_threshold")
    print(f"{f}: enabled={e}, threshold={t}")
```

Expected: 3 行都显示 `enabled=True, threshold=0.001`。

- [ ] **Step 4: 验证其他字段保持原值**

写脚本 `/tmp/verify_other_fields.py`：

```python
import configparser
for f in [
    "runs/baseline_2da_clean/config.ini",
    "runs/baseline_5da_clean/config.ini",
    "runs/baseline_normal_clean/config.ini",
]:
    c = configparser.ConfigParser()
    c.read(f)
    print(f"{f}:")
    print(f"  feature_type={c['general']['feature_type']}")
    print(f"  mass_tol_ppm={c['general']['mass_tol_ppm']}")
    print(f"  xic_cycle_window={c['general']['xic_cycle_window']}")
    print(f"  result_file={c['general']['result_file']}")
```

运行：`python3 /tmp/verify_other_fields.py`

Expected: 各字段值正常显示，无 KeyError。3 个文件的 result_file 路径分别指向 2da/5da/normal。

- [ ] **Step 5: 主项目测试不受影响**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
conda run -n jianyan pytest tests/ 2>&1 | tail -3
```

Expected: 266 passed.

- [ ] **Step 6: Commit**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
git add runs/baseline_2da_clean/config.ini runs/baseline_5da_clean/config.ini runs/baseline_normal_clean/config.ini
git commit -m "config: baseline config.ini 补 centroid_enabled / centroid_rel_threshold

3 个 baseline config.ini 是早期版本，缺最近 mzML centroiding 功能加的
两个字段。追加默认值与 root config.ini 一致：

  centroid_enabled = true
  centroid_rel_threshold = 0.001

这样下次重跑 make 2th/5th/normal 时 centroid 功能生效。

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 2: main.py 加 _resolve_feature_cols helper + 自动检测

**Files:**
- Modify: `tools/spec_trainer/src/main.py` — add helper + wire into main()
- Create: `tests/test_spec_trainer_main.py` — 3 unit tests

- [ ] **Step 1: 创建测试文件 tests/test_spec_trainer_main.py**

用 create 工具创建文件，内容如下：

```python
"""Test spec_trainer/src/main.py auto feature-column detection."""
import os
import sys
import importlib.util
import pytest


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC_TRAINER_SRC = os.path.join(_PROJECT_ROOT, "tools", "spec_trainer", "src")


def _load_main_module():
    """Load main.py without crashing on missing lightgbm/etc.

    Returns the module if loadable; pytest.skip otherwise.
    """
    if _SPEC_TRAINER_SRC not in sys.path:
        sys.path.insert(0, _SPEC_TRAINER_SRC)
    spec = importlib.util.spec_from_file_location(
        "main_for_test",
        os.path.join(_SPEC_TRAINER_SRC, "main.py"),
    )
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except ImportError as e:
        pytest.skip(f"main.py imports unavailable: {e}")
    return m


def test_resolve_feature_cols_explicit_list_passthrough():
    """When yaml provides explicit feature_cols list, return it unchanged."""
    m = _load_main_module()
    result = m._resolve_feature_cols(
        explicit=["a", "b", "c"],
        sample_csv_path="/nonexistent.csv",
        target_col="label",
    )
    assert result == ["a", "b", "c"]


def test_resolve_feature_cols_empty_triggers_auto_detect(tmp_path):
    """Empty feature_cols triggers auto-detection from CSV header."""
    m = _load_main_module()
    csv = tmp_path / "fake.csv"
    csv.write_text(
        "sequence,charge,protein_names,label,precursor_mz,sequence_len,"
        "raw_title1,raw_title2,label_type,modification_count,"
        "precursor_pearson,b_mean,y_p50\n"
    )
    result = m._resolve_feature_cols(
        explicit=[],
        sample_csv_path=str(csv),
        target_col="label",
    )
    assert result == ["precursor_pearson", "b_mean", "y_p50"]
    assert "label" not in result
    assert "modification_count" not in result
    assert "precursor_mz" not in result
    assert "sequence_len" not in result
    assert "raw_title1" not in result
    assert "protein_names" not in result


def test_resolve_feature_cols_none_triggers_auto_detect(tmp_path):
    """None feature_cols (yaml missing key) also triggers auto-detection."""
    m = _load_main_module()
    csv = tmp_path / "fake.csv"
    csv.write_text(
        "label,precursor_pearson,b_mean,modification_count\n"
    )
    result = m._resolve_feature_cols(
        explicit=None,
        sample_csv_path=str(csv),
        target_col="label",
    )
    assert result == ["precursor_pearson", "b_mean"]
```

- [ ] **Step 2: Run failing tests to verify**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
conda run -n jianyan pytest tests/test_spec_trainer_main.py -v
```

Expected: 3 tests FAIL with AttributeError (_resolve_feature_cols 未定义), 或全部 SKIP（lightgbm 缺失，pre-existing 问题）。两种情况都可接受 — 在 Step 4 之后再次 run 应该 PASS 或同样 SKIP。

- [ ] **Step 3: 在 main.py 中实现 _resolve_feature_cols**

打开 `tools/spec_trainer/src/main.py`。找到合适插入点：推荐在最后一个 `import` 语句之后、`def load_data(...)` 之前。

用 edit 工具插入这段代码（新 paragraph）：

```python


# META columns that are not features themselves (PSM identification + label).
# 与 tools/eval_baseline.py:37-41 保持一致。
META_COLUMNS = {
    "sequence", "charge", "raw_title1", "raw_title2",
    "protein_names", "label", "label_type",
    "precursor_mz", "sequence_len",
}

# 额外排除的特征列：modification_count 在训练时倾向于过拟合非物理信号
# （负样本 entrapment 大多带修饰），见 PLAN.md 三-2 分析。
EXCLUDED_EXTRA = {"modification_count"}


def _resolve_feature_cols(explicit, sample_csv_path, target_col):
    """Resolve final feature column list.

    If explicit is a non-empty list, return it unchanged (yaml took
    care of selection). Otherwise auto-detect from the CSV column
    header, excluding META_COLUMNS + EXCLUDED_EXTRA + target_col.

    The CSV column order determines the feature order (pandas read_csv
    is deterministic for a given file). Cross-runs with the same
    features.csv produce the same feature_cols list.
    """
    if explicit:
        return list(explicit)
    sample_df = pd.read_csv(sample_csv_path, nrows=0)
    all_cols = list(sample_df.columns)
    return [
        c for c in all_cols
        if c not in META_COLUMNS
        and c not in EXCLUDED_EXTRA
        and c != target_col
    ]
```

- [ ] **Step 4: 在 main() 中调用 _resolve_feature_cols**

打开 `tools/spec_trainer/src/main.py`。找到现有的 `feature_cols = cfg['data']['feature_cols']` 这一行。

用 edit 工具替换为：

```python
    target_col = cfg['data']['target_col']
    feature_cols = _resolve_feature_cols(
        explicit=cfg['data'].get('feature_cols'),
        sample_csv_path=cfg['data']['train_files'][0],
        target_col=target_col,
    )
    logging.info(f"using {len(feature_cols)} feature columns")
```

注意：原 main.py 可能已经在后面有 `target_col = cfg['data']['target_col']` 行；如果已存在，删除重复定义，确保 `target_col` 定义在 `_resolve_feature_cols` 调用之前（同一行 / 之上）。

- [ ] **Step 5: 运行测试验证通过**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
conda run -n jianyan pytest tests/test_spec_trainer_main.py -v
```

Expected: 3 tests PASS。如果 SKIP 表示 lightgbm 缺失，验证 helper 已定义：

```bash
python3 -c "src = open('tools/spec_trainer/src/main.py').read(); print('helper defined:', 'def _resolve_feature_cols' in src); print('META defined:', 'META_COLUMNS = {' in src)"
```

Expected: 两个 True。

- [ ] **Step 6: 主项目测试不受影响**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
conda run -n jianyan pytest tests/ 2>&1 | tail -3
```

Expected: 总通过数 = 旧数 + 3 (新增 3 个)，或同等数（如果 SKIP 而非 PASS）。无 FAIL。

- [ ] **Step 7: Commit**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
git add tools/spec_trainer/src/main.py tests/test_spec_trainer_main.py
git commit -m "feat(spec_trainer): main.py auto-detect feature_cols when yaml empty

加 _resolve_feature_cols(explicit, sample_csv_path, target_col) helper：
- explicit 非空列表 -> 返回原列表（yaml 显式指定）
- 否则从 sample_csv_path CSV header 自动推导，排除：
  - META_COLUMNS (与 tools/eval_baseline.py 一致)
  - EXCLUDED_EXTRA = {modification_count} (PLAN.md 三-2 物理意义弱)
  - target_col

main() 中改 'feature_cols = cfg[data][feature_cols]' 为调用 helper。

新增 3 个 pytest 单元测试覆盖 explicit / empty / None 三种 yaml 状态。
lightgbm 缺失时优雅 skip。

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 3: exp1.yaml + exp2.yaml feature_cols 改为空

**Files:**
- Modify: `tools/spec_trainer/config/exp1.yaml`
- Modify: `tools/spec_trainer/config/exp2.yaml`

- [ ] **Step 1: 查看当前 yaml 的 data 块**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
for f in tools/spec_trainer/config/exp1.yaml tools/spec_trainer/config/exp2.yaml; do
    echo "=== $f ==="
    sed -n '/^data:/,/^model:/p' "$f" | head -25
done
```

Expected: 显示 `data:` 段含 train_files / test_files / feature_cols 列表（多行 `- xxx` 特征名）/ target_col。

- [ ] **Step 2: 用 edit 工具改 exp1.yaml 的 feature_cols**

打开 `tools/spec_trainer/config/exp1.yaml`。

old_str（从 `  feature_cols:` 开始一直到最后一个特征条目的行）— 例如：
```
  feature_cols:
    - precursor_mz
    - sequence_len
    - precursor_pearson
    - b_count
    ...
    - matched_intensity_percent
```

new_str:
```
  feature_cols: []
```

注意保留 `feature_cols:` 之后的 `target_col: label` 不动。

- [ ] **Step 3: 用 edit 工具改 exp2.yaml 的 feature_cols**

对 exp2.yaml 做同样操作：删除 feature_cols 下的所有 `- xxx` 列表条目，改为 `feature_cols: []`。

- [ ] **Step 4: 验证两个 yaml 合法 + feature_cols 为空**

写脚本 `/tmp/verify_yaml.py`：

```python
import yaml
for f in [
    "tools/spec_trainer/config/exp1.yaml",
    "tools/spec_trainer/config/exp2.yaml",
]:
    cfg = yaml.safe_load(open(f))
    print(f"{f}:")
    print(f"  train_files: {cfg['data']['train_files']}")
    print(f"  test_files: {cfg['data']['test_files']}")
    print(f"  feature_cols: {cfg['data'].get('feature_cols')}")
    print(f"  target_col: {cfg['data']['target_col']}")
    print(f"  model.type: {cfg['model']['type']}")
```

运行：`python3 /tmp/verify_yaml.py`

Expected:
- `feature_cols: []` 两次
- 其他字段保持原值

- [ ] **Step 5: 主项目测试不受影响**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
conda run -n jianyan pytest tests/ 2>&1 | tail -3
```

Expected: 测试通过总数不变（yaml 改动不应影响测试）。

- [ ] **Step 6: Commit**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
git add tools/spec_trainer/config/exp1.yaml tools/spec_trainer/config/exp2.yaml
git commit -m "feat(spec_trainer): exp1/exp2.yaml feature_cols 留空 — 自动检测全部特征

把 yaml 中硬编码的 18 个旧 pearson 类特征列表删除，改为 feature_cols: []。
main.py 的 _resolve_feature_cols 自动检测会读 train_files[0] 的 CSV
header 推导出所有特征列（排除 META + modification_count + target_col）。

这样 yaml 自动跟随 features.csv schema 演进，加新特征列不用改 yaml。

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Summary

After all 3 tasks:

- 1 spec doc (committed in `cc0dacf` during brainstorming).
- 3 implementation commits (Tasks 1-3):
  - Task 1: 3 baseline config.ini 补 centroid 字段
  - Task 2: main.py 加 _resolve_feature_cols + 3 单元测试
  - Task 3: exp1/exp2.yaml feature_cols 改为空
- 5 files modified: 3 baseline config.ini + main.py + exp1.yaml + exp2.yaml
- 1 new test file: tests/test_spec_trainer_main.py
- 3 new tests added (or SKIP if lightgbm missing)
- 主项目 266 tests 不受影响
- 流水线 `make 2th` -> `make train-exp1` 现在能跑通（pending 用户实际重跑 features.csv）

