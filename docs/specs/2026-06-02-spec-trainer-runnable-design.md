# 设计文档：spec_trainer 跑通 — 补 centroid 字段 + 自动特征列检测

> 编写日期：2026-06-02 | 目标：让 make train-exp1/exp2 真正能在 ms2-met 流水线上跑通

---

## 一、动机

spec_trainer 已经通过 git subtree 合并进 ms2-met（见 2026-06-02-spec-trainer-integration-design.md）。但实际跑通还差三个小坑：

1. **3 个 baseline config.ini 缺 centroid 字段**：项目最近加了 mzML centroiding 功能（centroid_enabled / centroid_rel_threshold），root config.ini 已经有这两个字段，但 baseline_*da_clean/config.ini 是早期版本，缺这两个字段 → 重跑 make 2th/5th/normal 会用旧 profile 模式（如果代码读这两个字段会 fallback 默认）。

2. **spec_trainer yaml feature_cols 只列了 18 个旧特征**：原 exp1.yaml feature_cols 是手工列出的 18 个 b_*/y_*/all_* pearson 类基础特征。现在 ms2-met 代码能产出 100+ 列（含 R3 26 列 + R4 20 列），手工列举不可行也容易漏。

3. **modification_count 应排除**：根据 PLAN.md 三-2 的分析，modification_count 在训练时会"压性能高分但物理意义弱"（负样本带修饰太多导致 ML 学到非物理特征）。yaml 不应列入。

## 二、设计哲学

- **YAGNI**：3 个改动都是小补丁，不重构。
- **不破坏向后兼容**：yaml 仍可显式列 feature_cols；自动检测只是缺失时的 fallback。
- **跨 CSV 鲁棒**：自动检测从第一个 train_file 推导列名，单一来源；2da/5da/normal 的列差异通过 pandas concat 处理（缺列自动 NaN）。

---

## 三、改动 A — baseline config.ini 加 centroid 字段

### 3.1 现状

`config.ini`（项目根）有：
```ini
[general]
...
xic_cycle_window = 6
result_file = result.csv

# 加载 mzML 时是否对 profile 谱图做 centroiding。
centroid_enabled = true

# centroid 阈值
centroid_rel_threshold = 0.001
```

`runs/baseline_2da_clean/config.ini` / `5da_clean` / `normal_clean` 都缺最后两个字段（注释 + 字段）。

### 3.2 改动

在 3 个 baseline config.ini 的 `[general]` 段末尾追加同样的 centroid 块（含注释）。**仅 append，不动其他行**，避免改动 raw_path / mass_tol_ppm 等已经定制的字段。

### 3.3 影响

下次 `make 2th` 触发 ms2-met 重跑特征提取时，centroid 功能生效（profile 谱图被 centroid 化）。如果 ms2-met 代码对这两个字段有默认值兜底，旧 config 仍能跑（只是不开 centroid）；明确写出更安全。

## 四、改动 B — main.py 加自动特征列检测

### 4.1 现状

`tools/spec_trainer/src/main.py` 用法：

```python
feature_cols = cfg['data']['feature_cols']  # 必须存在的硬性键
X, y = load_data(file_paths, feature_cols, target_col)
```

`load_data` 调用 `df[feature_cols]`，缺列会报错。

### 4.2 改动

在调用 `load_data` 之前加自动检测逻辑：

```python
META_COLUMNS = {
    "sequence", "charge", "raw_title1", "raw_title2",
    "protein_names", "label", "label_type",
    "precursor_mz", "sequence_len",
}
EXCLUDED_EXTRA = {"modification_count"}

target_col = cfg['data']['target_col']
feature_cols_cfg = cfg['data'].get('feature_cols')

if not feature_cols_cfg:
    # 自动从第一个 train_file 推导
    sample_df = pd.read_csv(cfg['data']['train_files'][0], nrows=0)
    all_cols = list(sample_df.columns)
    feature_cols = [
        c for c in all_cols
        if c not in META_COLUMNS
        and c not in EXCLUDED_EXTRA
        and c != target_col
    ]
    logging.info(
        f"auto-detected {len(feature_cols)} feature columns from "
        f"{cfg[\'data\'][\'train_files\'][0]}")
else:
    feature_cols = feature_cols_cfg
    logging.info(f"using {len(feature_cols)} explicit feature columns from yaml")
```

### 4.3 META_COLUMNS 来源

完全复用 `tools/eval_baseline.py:37-41` 已有常量：

```python
META_COLUMNS = {
    "sequence", "charge", "raw_title1", "raw_title2",
    "protein_names", "label", "label_type",
    "precursor_mz", "sequence_len",
}
```

注意：`precursor_mz` 和 `sequence_len` 在 ms2-met 中是 PSM 元信息，不是物理 SILAC 验证特征 → 与 eval_baseline 保持一致排除。

### 4.4 列顺序

`pd.read_csv(..., nrows=0)` 返回的 DataFrame 列顺序 = features.csv 物理列顺序，跨调用 deterministic。所以自动检测的 feature_cols 列表顺序是稳定的，不会跑两次结果不同。

### 4.5 边界

| 输入 | 行为 |
|------|------|
| `feature_cols:` 缺失（yaml 没有这个键）| `cfg['data'].get(...)` 返回 None → 自动检测 |
| `feature_cols: []`（空列表）| `not []` 为 True → 自动检测 |
| `feature_cols: null`（yaml 显式 null）| `not None` 为 True → 自动检测 |
| `feature_cols: [a, b, c]`（显式列表）| 走原路径，用显式 |
| 第一个 train_file 不存在 | `pd.read_csv` 抛 FileNotFoundError → 与原 load_data 行为一致 |
| 第一个 train_file 是 0 行（只有 header）| nrows=0 仍能读 header → OK |

## 五、改动 C — exp1.yaml / exp2.yaml 改 feature_cols 为空

### 5.1 改动 exp1.yaml

把整段 feature_cols 显式列表删除，留空字段：

```yaml
data:
  train_files:
    - runs/baseline_2da_clean/features.csv
  test_files:
    - runs/baseline_2da_clean/features.csv
  feature_cols: []
  target_col: label
```

### 5.2 改动 exp2.yaml

同样把显式 feature_cols 删除：

```yaml
data:
  train_files:
    - runs/baseline_2da_clean/features.csv
    - runs/baseline_5da_clean/features.csv
  test_files:
    - runs/baseline_2da_clean/features.csv
    - runs/baseline_5da_clean/features.csv
  feature_cols: []
  target_col: label
```

### 5.3 优势

- yaml 自动跟随 features.csv schema 演进（加新特征列不用改 yaml）
- 跨数据集（2da 66 列 vs 5da 77 列）combined 时，pandas concat 自动取列并集（缺列填 NaN，LightGBM 原生支持 NaN）
- 显式 `feature_cols: []` 比省略字段更明确——"这里是个有意为之的空，不是漏写"

---

## 六、验证

### 6.1 改动 A 验证

```bash
python -c "
import configparser
for f in ['runs/baseline_2da_clean/config.ini',
          'runs/baseline_5da_clean/config.ini',
          'runs/baseline_normal_clean/config.ini']:
    c = configparser.ConfigParser()
    c.read(f)
    e = c['general'].getboolean('centroid_enabled')
    t = c['general'].getfloat('centroid_rel_threshold')
    print(f'{f}: enabled={e}, threshold={t}')
"
```

Expected: 3 行都显示 `enabled=True, threshold=0.001`。

### 6.2 改动 B 单元测试

写一个 pytest 测试，构造 mock CSV 含 META 列 + 一些特征列 + modification_count，验证自动检测返回正确的特征列表：

```python
def test_main_auto_detects_feature_cols(tmp_path):
    """When yaml feature_cols is empty, main.py auto-detects from CSV header."""
    csv = tmp_path / "fake.csv"
    csv.write_text(
        "sequence,charge,protein_names,label,precursor_mz,sequence_len,"
        "raw_title1,raw_title2,label_type,modification_count,"
        "precursor_pearson,b_mean,y_p50\n"
    )
    # 实际从 main.py 抽取的自动检测逻辑（或 import main 后调用）
    ...
    assert feature_cols == ["precursor_pearson", "b_mean", "y_p50"]
    assert "label" not in feature_cols  # target excluded
    assert "modification_count" not in feature_cols  # extra excluded
    assert "precursor_mz" not in feature_cols  # META excluded
```

测试放在 `tests/test_spec_trainer_main.py`（新建）。

### 6.3 改动 C 验证

```bash
python -c "
import yaml
for f in ['tools/spec_trainer/config/exp1.yaml',
          'tools/spec_trainer/config/exp2.yaml']:
    cfg = yaml.safe_load(open(f))
    print(f'{f}: feature_cols = {cfg[\'data\'].get(\'feature_cols\')}')
"
```

Expected: 两行都显示 `feature_cols = []` 或 `None`。

### 6.4 端到端 import sanity

```bash
python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('main', 'tools/spec_trainer/src/main.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print('OK')
"
```

Expected: OK 或 IMPORT-OK-BUT-MISSING-DEP: No module named 'lightgbm'（pre-existing）.

---

## 七、不在本次范围

- 实际重跑 `make 2th/5th/normal`（用户说"之后会重跑"）
- 修改 Makefile（user 选 A：现有 train-exp1/exp2/all 已经够）
- 修改其他 yaml（base_*.yaml/loo_test_*.yaml — 它们根本没合并进来）
- 装 lightgbm
- 把 EXCLUDED_EXTRA 也提到 yaml 配置（YAGNI；如果未来要排除别的特征，再加配置）

---

## 八、风险

| 风险 | 评估 | 应对 |
|------|------|------|
| centroid_enabled / centroid_rel_threshold 字段名拼写错 | 中 | 改动 A 验证步骤捕获 |
| 自动检测把意外列（如 future "score"）当成特征 | 中 | META_COLUMNS 设计为"白名单排除"，需后续扩展时维护 |
| pandas concat 2da+5da 列不齐导致 LightGBM 抛错 | 低 | LightGBM 原生支持 NaN |
| 自动检测在 train_files[0] 不存在时报错（不是 FileNotFoundError 而是 OSError） | 低 | 与原 pd.read_csv 行为一致 |
| modification_count 排除策略硬编码在 main.py，未来增删难 | 低 | YAGNI；将来需要时加 yaml exclude_cols 字段 |
| 单元测试可能与 conda env 缺 lightgbm 冲突 | 中 | 测试不 import main 全模块，只测试自动检测函数（提取为 helper）|
