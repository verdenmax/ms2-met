# spec_trainer Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把独立的 spec_trainer 训练框架通过 git subtree 合并进 ms2-met，作为 tools/spec_trainer/ 子项目；调整 exp1/exp2.yaml 路径；修复 main.py 导入；在主 Makefile 加 train-* target；流水线 `make 2th` -> `make train-exp1` 一键贯通。

**Architecture:** 使用 `git subtree add --prefix=tools/spec_trainer` 保留 5 个历史 commit 合并进来。子项目数据/模型/产出目录在原仓库就已 gitignored，自然不跟过来；主项目把训练输出统一落到 `runs/spec_trainer/{models,results,figures}/`。yaml 配置路径相对 ms2-met 根目录；main.py 顶部加 `sys.path` 注入修复隐式相对 import（仿照 tools/eval_baseline.py:25-28 pattern）。

**Tech Stack:** git subtree, GNU Make, Python 3.13, conda env `jianyan` at `/home/verden/.conda/envs/jianyan`.

---

## File Structure

**New (from subtree):**
- `tools/spec_trainer/` — 整个 spec_trainer 目录（src/, config/, .gitignore, Makefile, readme.md, split_by_rep.py）

**Modified after subtree:**
- `tools/spec_trainer/src/main.py` — 加 sys.path 修复 + figures 路径硬编码到 runs/spec_trainer/
- `tools/spec_trainer/config/exp1.yaml` — 路径改为 runs/baseline_2da_clean/features.csv 等
- `tools/spec_trainer/config/exp2.yaml` — 路径改为 runs/baseline_5da_clean/features.csv 等
- `Makefile` (ms2-met 根) — 加 train-exp1/exp2/all/clean-train target

**Not touched:**
- 其他 39 个 yaml (base_*.yaml / loo_test_*.yaml)
- `tools/spec_trainer/Makefile`（子项目自己的 Makefile，保留供独立调用）
- `tools/spec_trainer/.gitignore`（保留作子项目局部 ignore）
- 原 `../spec_trainer/` 仓库（保留兑底）

---

## Task 1: subtree merge — 把 spec_trainer 合并进 tools/spec_trainer/

**Files:**
- Create (via subtree): `tools/spec_trainer/` (整个子目录)
- Test: verify subtree merge correctness

- [ ] **Step 1: Pre-check 工作树干净**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
git status --short
```

Expected: 只有 `cross_domain_analysis/` 和 `docs/PF2_FORMAT.md` 这两个 baseline untracked（pre-existing），其他无 modified/staged。

如果工作树不干净（有未提交修改），先 commit 或 stash 再继续。

- [ ] **Step 2: 添加 spec_trainer 为临时 git remote**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
git remote add spec_trainer_local /home/verden/pfind/2025-fall/code/spec_trainer
git fetch spec_trainer_local
```

Expected: fetch 成功，显示拉取了 spec_trainer 的 5 个 commit。

- [ ] **Step 3: 用 subtree add 合并进 tools/spec_trainer/**

```bash
git subtree add --prefix=tools/spec_trainer spec_trainer_local master
```

Expected:
- 创建一个 "Add 'tools/spec_trainer/' from commit '...'" 类型的 merge commit
- 工作树多出 `tools/spec_trainer/` 目录，含 src/, config/, .gitignore, Makefile, readme.md, split_by_rep.py

- [ ] **Step 4: 验证文件已合并 + 历史保留**

```bash
ls tools/spec_trainer/
ls tools/spec_trainer/src/
ls tools/spec_trainer/config/ | head -10
git --no-pager log --oneline -10
```

Expected:
- ls 显示 src/ config/ readme.md Makefile split_by_rep.py .gitignore
- src/ 显示 main.py train.py train2.py models/
- config/ 显示 41 个 yaml（exp1.yaml exp2.yaml base_*.yaml loo_test_*.yaml）
- git log 显示新的 subtree merge commit 在顶端

- [ ] **Step 5: 验证数据/模型/产出目录没跟过来**

```bash
ls tools/spec_trainer/ | grep -E "^(data|models|results|figures)$" && echo "WARN: should be empty" || echo "OK: data/models/results/figures absent as expected"
```

Expected: `OK: data/models/results/figures absent as expected`

- [ ] **Step 6: 清除临时 remote**

```bash
git remote remove spec_trainer_local
git remote -v
```

Expected: 只剩 origin 和 gitlab，无 spec_trainer_local。

- [ ] **Step 7: 主项目测试不受影响**

```bash
conda run -n jianyan pytest tests/ 2>&1 | tail -3
```

Expected: 266 passed (与基线一致).

- [ ] **Step 8: Commit**

`subtree add` 已经自动生成 commit。验证 git log 顶部已经是 subtree merge commit（消息类似 "Add 'tools/spec_trainer/' from commit '...'"）。无需额外手动 commit；此步骤直接通过。

如果 git log 不显示 subtree commit（异常情况），按报错处理。

---

## Task 2: 修复 main.py 的隐式相对导入

**Files:**
- Modify: `tools/spec_trainer/src/main.py` (顶部加 sys.path 注入)

- [ ] **Step 1: 验证当前 import 从根目录跑会失败**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
conda run -n jianyan python tools/spec_trainer/src/main.py --config tools/spec_trainer/config/exp1.yaml --name exp1 2>&1 | head -5
```

Expected: `ModuleNotFoundError: No module named 'models'` 或 `from models.model_manager import ModelManager` 失败.

如果不报错（异常），说明 spec_trainer 已经有 sys.path 自修复 — 跳过本 task。

- [ ] **Step 2: 修改 main.py 顶部添加 sys.path 注入**

打开 `tools/spec_trainer/src/main.py`，找到第 1 行（当前是 `from models.model_manager import ModelManager`），在它之前插入：

```python
import os
import sys

# 让 src/ 进入 sys.path，使得 main.py 可以从任何工作目录被调用
# 仿照 tools/eval_baseline.py 的 pattern
_SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

```

完整修改后的前 12 行应该是：

```python
import os
import sys

# 让 src/ 进入 sys.path，使得 main.py 可以从任何工作目录被调用
# 仿照 tools/eval_baseline.py 的 pattern
_SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from models.model_manager import ModelManager
import pandas as pd
import yaml
```

- [ ] **Step 3: 验证修复后 import 不再失败**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
conda run -n jianyan python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('main', 'tools/spec_trainer/src/main.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print('OK: import succeeded')
"
```

Expected: `OK: import succeeded` (no ImportError).

注意：这个测试方式实际触发了 main.py 的 import 语句但不调用 main()，所以不会真启动训练。

- [ ] **Step 4: Commit**

```bash
git add tools/spec_trainer/src/main.py
git commit -m "fix(spec_trainer): main.py 顶部加 sys.path 注入修复隐式相对导入

合并 spec_trainer 后从 ms2-met 根目录跑 tools/spec_trainer/src/main.py 会
失败，因为 from models.model_manager 是相对 src/ 的隐式相对导入。

修复 pattern 与 tools/eval_baseline.py:25-28 一致。这样既能从 ms2-met
根目录跑（主 Makefile 入口），也能 cd 进子目录跑（子项目原 Makefile）。

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 3: 更新 exp1.yaml 路径

**Files:**
- Modify: `tools/spec_trainer/config/exp1.yaml`

- [ ] **Step 1: 查看当前 exp1.yaml**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
cat tools/spec_trainer/config/exp1.yaml
```

记下当前 `data.train_files`、`data.test_files`、`output.model_path`、`output.result_path` 的值。

- [ ] **Step 2: 编辑 exp1.yaml — data 块**

打开 `tools/spec_trainer/config/exp1.yaml`，找到 `data:` 块。原内容形如：

```yaml
data:
  train_files:
    - data/hela_2da_20_mix_2.csv
  test_files:
    - data/hela_2da_mix_new_base_feature.csv
```

替换为：

```yaml
data:
  train_files:
    - runs/baseline_2da_clean/features.csv
  test_files:
    - runs/baseline_2da_clean/features.csv
```

`feature_cols` 块完全保留不动。

- [ ] **Step 3: 编辑 exp1.yaml — output 块**

找到 `output:` 块。原内容形如：

```yaml
output:
  model_path: models/exp1.txt
  result_path: results/exp1.json
```

替换为：

```yaml
output:
  model_path: runs/spec_trainer/models/exp1.txt
  result_path: runs/spec_trainer/results/exp1.json
```

`model` 和 `training` 块完全保留不动。

- [ ] **Step 4: 验证 yaml 合法**

```bash
conda run -n jianyan python -c "
import yaml
with open('tools/spec_trainer/config/exp1.yaml') as f:
    cfg = yaml.safe_load(f)
print('OK: yaml valid')
print('  train_files:', cfg['data']['train_files'])
print('  test_files:', cfg['data']['test_files'])
print('  model_path:', cfg['output']['model_path'])
print('  result_path:', cfg['output']['result_path'])
"
```

Expected output:
```
OK: yaml valid
  train_files: ['runs/baseline_2da_clean/features.csv']
  test_files: ['runs/baseline_2da_clean/features.csv']
  model_path: runs/spec_trainer/models/exp1.txt
  result_path: runs/spec_trainer/results/exp1.json
```

- [ ] **Step 5: Commit**

```bash
git add tools/spec_trainer/config/exp1.yaml
git commit -m "feat(spec_trainer): exp1.yaml 路径相对 ms2-met 根目录

迁移完成后 exp1.yaml 的数据 / 输出路径调整：
  data/hela_2da_*.csv  -> runs/baseline_2da_clean/features.csv
  models/exp1.txt      -> runs/spec_trainer/models/exp1.txt
  results/exp1.json    -> runs/spec_trainer/results/exp1.json

这样从 ms2-met 根目录跑能找到所有文件。其他 40 个 yaml
（base_*.yaml / loo_test_*.yaml）保持原样，用户用到再改。

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 4: 更新 exp2.yaml 路径

**Files:**
- Modify: `tools/spec_trainer/config/exp2.yaml`

- [ ] **Step 1: 查看当前 exp2.yaml**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
cat tools/spec_trainer/config/exp2.yaml
```

记下当前的 `data` 和 `output` 路径。注意 exp2 在原 spec_trainer 中可能指向 5da 或 normal 数据集，看实际语义。

- [ ] **Step 2: 编辑 exp2.yaml — data 块**

打开 `tools/spec_trainer/config/exp2.yaml`，找到 `data:` 块。如果原 train_files / test_files 路径含 `5da` 字样：

替换为：

```yaml
data:
  train_files:
    - runs/baseline_5da_clean/features.csv
  test_files:
    - runs/baseline_5da_clean/features.csv
```

如果原路径含 `normal` 字样（看 exp2.yaml 实际内容判断），则替换为：

```yaml
data:
  train_files:
    - runs/baseline_normal_clean/features.csv
  test_files:
    - runs/baseline_normal_clean/features.csv
```

如果含 2da 字样（与 exp1 重复），保持 2da 路径不变。

- [ ] **Step 3: 编辑 exp2.yaml — output 块**

找到 `output:` 块。原内容形如：

```yaml
output:
  model_path: models/exp2.txt
  result_path: results/exp2.json
```

替换为：

```yaml
output:
  model_path: runs/spec_trainer/models/exp2.txt
  result_path: runs/spec_trainer/results/exp2.json
```

- [ ] **Step 4: 验证 yaml 合法**

```bash
conda run -n jianyan python -c "
import yaml
with open('tools/spec_trainer/config/exp2.yaml') as f:
    cfg = yaml.safe_load(f)
print('OK: yaml valid')
print('  train_files:', cfg['data']['train_files'])
print('  test_files:', cfg['data']['test_files'])
print('  model_path:', cfg['output']['model_path'])
print('  result_path:', cfg['output']['result_path'])
"
```

Expected: `OK: yaml valid` + paths all rewritten to `runs/...`.

- [ ] **Step 5: Commit**

```bash
git add tools/spec_trainer/config/exp2.yaml
git commit -m "feat(spec_trainer): exp2.yaml 路径相对 ms2-met 根目录

迁移后 exp2.yaml 的数据 / 输出路径调整为 runs/... 根相对路径。
具体数据集（2da / 5da / normal）按 exp2.yaml 原本指向的语义。

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```


---

## Task 5: main.py figures 路径硬编码到 runs/spec_trainer/figures/

**Files:**
- Modify: `tools/spec_trainer/src/main.py` (figures 路径硬编码)

- [ ] **Step 1: 找出 figures 输出路径在 main.py 里的位置**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
grep -n "figures" tools/spec_trainer/src/main.py
```

Expected: 显示 1-3 行类似 `fig_path = f"figures/{name}_..."` 或 `roc_path = f"figures/{name}_roc.png"` 的硬编码路径。

- [ ] **Step 2: 替换硬编码 `figures/` 前缀为 `runs/spec_trainer/figures/`**

对每个 grep 找到的行，将 `figures/` 替换为 `runs/spec_trainer/figures/`。例如：

```python
# 原（如果有这样的行）：
fig_path = f"figures/{name}_feature_importance.png"
roc_path = f"figures/{name}_roc.png"

# 改为：
fig_path = f"runs/spec_trainer/figures/{name}_feature_importance.png"
roc_path = f"runs/spec_trainer/figures/{name}_roc.png"
```

注意：如果 main.py 用了 `os.makedirs(os.path.dirname(fig_path), exist_ok=True)` 这种模式，目录会自动创建，无需额外处理。

如果 figures 路径来自 yaml `output.fig_dir`（YAGNI 决策不选这条路），则不要在此 task 中做；改为 Task 7 加 yaml 字段（本计划不在范围）。

- [ ] **Step 3: 验证修改**

```bash
grep -n "figures" tools/spec_trainer/src/main.py
```

Expected: 所有 figures 路径都以 `runs/spec_trainer/figures/` 开头。

- [ ] **Step 4: Commit**

```bash
git add tools/spec_trainer/src/main.py
git commit -m "feat(spec_trainer): main.py figures 输出落到 runs/spec_trainer/figures/

将 ROC / feature_importance 图的输出路径从硬编码的 figures/<name>_*.png
改为 runs/spec_trainer/figures/<name>_*.png，与其他训练产出 (models/
results/) 同位于 runs/spec_trainer/。

YAGNI 决策：未引入 yaml output.fig_dir 字段，因为只 exp1/exp2 需要这
条路径；如果未来需要按实验切换图目录，再扩展配置项。

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 6: 主 Makefile 新增 train-* target

**Files:**
- Modify: `Makefile` (ms2-met 根目录的 Makefile)

- [ ] **Step 1: 查看当前 Makefile 末尾结构**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
tail -30 Makefile
```

确认 Makefile 末尾是 `clean:` block（旧式清理）。新增 train-* target 应该插入在 `clean-all` block 之后，`# ---------- 兼容旧用法 ----------` 行之前。

- [ ] **Step 2: 编辑 Makefile，在 `clean-all` 之后插入新 section**

找到 Makefile 中这一行：

```make
clean-all: clean-2th clean-5th clean-normal
```

在它之后，`# ---------- 兼容旧用法 ----------` 之前，插入：

```make

# ---------- spec_trainer 训练 target ----------
#
# train-exp1 / train-exp2: 调用 tools/spec_trainer/src/main.py 训练对应实验
# 自动依赖 features.csv 存在；缺失时级联触发 make 2th / 5th。
# 输出落到 runs/spec_trainer/{models,results,figures}/

.PHONY: train-exp1 train-exp2 train-all clean-train

# features.csv 不存在时自动跑对应特征提取
runs/baseline_2da_clean/features.csv:
$(MAKE) 2th

runs/baseline_5da_clean/features.csv:
$(MAKE) 5th

runs/baseline_normal_clean/features.csv:
$(MAKE) normal

train-exp1: runs/baseline_2da_clean/features.csv tools/spec_trainer/config/exp1.yaml
$(call BANNER,train-exp1)
@mkdir -p runs/spec_trainer/models runs/spec_trainer/results runs/spec_trainer/figures
$(PY) tools/spec_trainer/src/main.py --config tools/spec_trainer/config/exp1.yaml --name exp1
@echo "[done] train-exp1 finished"

train-exp2: runs/baseline_5da_clean/features.csv tools/spec_trainer/config/exp2.yaml
$(call BANNER,train-exp2)
@mkdir -p runs/spec_trainer/models runs/spec_trainer/results runs/spec_trainer/figures
$(PY) tools/spec_trainer/src/main.py --config tools/spec_trainer/config/exp2.yaml --name exp2
@echo "[done] train-exp2 finished"

train-all: train-exp1 train-exp2

clean-train:
@if [ -d runs/spec_trainer ]; then \
find runs/spec_trainer -mindepth 1 -delete 2>/dev/null || true; \
echo "[cleaned] runs/spec_trainer/"; \
else \
echo "[skip] runs/spec_trainer/ does not exist"; \
fi
```

**注意**：如果 Task 4 决定 exp2.yaml 指向 normal 而非 5da，对应改 train-exp2 的 features.csv 依赖路径。

- [ ] **Step 3: 同时更新 help target 文本**

找到 `help:` block 中列出 clean-all 的行（约在文件中部），在其后插入：

```make
@echo ""
@echo "  make train-exp1      训练 exp1（依赖 runs/baseline_2da_clean/features.csv）"
@echo "  make train-exp2      训练 exp2（依赖对应数据集 features.csv）"
@echo "  make train-all       顺序跑 train-exp1 + train-exp2"
@echo "  make clean-train     清理 runs/spec_trainer/ 训练产出"
```

具体插入位置：在 `@echo "  make clean           旧式清理（checkpoint.pkl 等）"` 这一行之前。

- [ ] **Step 4: 验证 Makefile 语法 + 新 target 可见**

```bash
make help 2>&1
```

Expected output: 帮助信息中显示新的 train-exp1 / train-exp2 / train-all / clean-train 4 个 target。

- [ ] **Step 5: dry-run 验证依赖链**

```bash
make -n train-exp1 2>&1
```

Expected: 显示命令链。如果 `runs/baseline_2da_clean/features.csv` 已存在，命令应该直接是：
```
... mkdir -p ...
python3 tools/spec_trainer/src/main.py --config tools/spec_trainer/config/exp1.yaml --name exp1
[done] train-exp1 finished
```

如果 features.csv 不存在，应该看到先跑 make 2th 的命令（嵌套 make 输出）。

- [ ] **Step 6: dry-run train-all**

```bash
make -n train-all 2>&1 | head -20
```

Expected: 包含 train-exp1 和 train-exp2 的命令链。

- [ ] **Step 7: dry-run clean-train**

```bash
make -n clean-train 2>&1
```

Expected: 显示 if/else 删除 runs/spec_trainer/ 的命令链。

- [ ] **Step 8: 主项目测试不受影响**

```bash
conda run -n jianyan pytest tests/ 2>&1 | tail -3
```

Expected: 266 passed.

- [ ] **Step 9: Commit**

```bash
git add Makefile
git commit -m "build(make): add train-exp1/exp2/all + clean-train targets

新增训练流水线 target：
  make train-exp1     -> tools/spec_trainer/src/main.py --config exp1.yaml
  make train-exp2     -> tools/spec_trainer/src/main.py --config exp2.yaml
  make train-all      -> train-exp1 + train-exp2
  make clean-train    -> 清理 runs/spec_trainer/

关键设计：
- features.csv 不存在时级联触发 make 2th/5th/normal
- 训练产出统一落到 runs/spec_trainer/{models,results,figures}/
- 自动 mkdir -p runs/spec_trainer/* 防 main.py 找不到输出目录
- help target 同步列出新 4 个

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Summary

After all 6 tasks:

- 1 spec doc (committed in `16142de` during brainstorming).
- 6 implementation commits (Tasks 1-6):
  - Task 1: 1 subtree merge commit (auto-generated, no manual commit needed)
  - Tasks 2-6: 5 manual commits (fix import + 2 yaml + figures + Makefile)
- 1 new directory: `tools/spec_trainer/` (整个子项目 from subtree)
- 4 files modified within `tools/spec_trainer/`: src/main.py + config/exp1.yaml + config/exp2.yaml
- 1 file modified at root: `Makefile` (+train-* targets)
- 0 changes to tests (主项目 266 tests 不受影响)
- 流水线一键贯通：`make 2th` -> `make train-exp1` 自动级联
- 保留 spec_trainer 5 个历史 commit + 子项目独立调用能力（cd tools/spec_trainer && make）
