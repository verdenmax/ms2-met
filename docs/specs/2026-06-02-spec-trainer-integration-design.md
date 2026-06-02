# 设计文档：spec_trainer 集成为 ms2-met 子项目

> 编写日期：2026-06-02 | 目标：把独立的 spec_trainer 训练框架合并进 ms2-met，单仓库维护

---

## 一、动机

spec_trainer 是一个独立训练框架（202 行 main.py + ModelManager 抽象 + 4 种模型 + 41 个 yaml + 自己的 Makefile），消费 ms2-met 输出的 features.csv 来训练 / 评估模型。两者在 pipeline 上连续：

```
ms2-met (make 2th)  ->  runs/baseline_2da_clean/features.csv  ->  spec_trainer  ->  训练模型 + 评估
```

合并后：
1. 单仓库维护：训练相关 commit 与特征代码 commit 在同一 git history
2. 流水线一键贯通：make train-exp1 自动级联 make 2th 生成 features.csv 再训练
3. 路径统一：训练输出落到 runs/spec_trainer/ 与已有 runs/baseline_*/ 对齐
4. 保留独立调用能力：子项目自己的 Makefile 仍可用

---

## 二、迁移工具：git subtree（非 squash）

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
git remote add spec_trainer_local /home/verden/pfind/2025-fall/code/spec_trainer
git fetch spec_trainer_local
git subtree add --prefix=tools/spec_trainer spec_trainer_local master
git remote remove spec_trainer_local
```

效果：
- spec_trainer 的 5 个历史 commit（61bc7a4 -> fdef201）完整保留
- 路径在新 commit 中变成 tools/spec_trainer/...
- subtree 自动加一个 merge commit

注意：subtree 只带 git tracked 的文件。spec_trainer 的 .gitignore 已经排除 data/ / models/ / results/ / figures/，所以这些目录的内容（合计 1.2GB+）不会跟过来。

---

## 三、目标目录结构

```
ms2-met/
├── workflows/, spectrum/, manager/
├── tools/
│   ├── extract_common.py            (已有)
│   ├── eval_baseline.py             (已有)
│   ├── eval_feature_ablation.py     (已有)
│   ├── entrapment_classify.py       (已有)
│   └── spec_trainer/                (新)
│       ├── .gitignore               (保留)
│       ├── readme.md
│       ├── Makefile                 (子项目原 Makefile)
│       ├── split_by_rep.py
│       ├── src/
│       │   ├── main.py
│       │   ├── train.py / train2.py
│       │   └── models/
│       └── config/
│           ├── exp1.yaml, exp2.yaml          (本迁移调整路径)
│           ├── base_*.yaml (15 个)           (保持原样)
│           └── loo_test_*.yaml (24 个)       (保持原样)
├── runs/
│   ├── baseline_2da_clean/          (已有)
│   ├── baseline_5da_clean/          (已有)
│   ├── baseline_normal_clean/       (已有)
│   └── spec_trainer/                (新：首次跑 train-* 自动创建)
│       ├── models/
│       ├── results/
│       └── figures/
└── Makefile                         (新增 train-* target)
```

---

## 四、yaml 路径策略

### 4.1 路径相对哪个目录

所有路径相对 ms2-met 根目录。命令从根目录跑，不进入子目录。理由：主 Makefile 一直在根跑；runs/ 在根；与已有 python tools/eval_baseline.py 风格一致。

### 4.2 本迁移仅改 exp1.yaml 和 exp2.yaml

其他 41 个 yaml（base_*.yaml / loo_test_*.yaml）保持原样，用户用到时再手工改。

### 4.3 exp1.yaml 路径改动示例

原：
```yaml
data:
  train_files:
    - data/hela_2da_20_mix_2.csv
  test_files:
    - data/hela_2da_mix_new_base_feature.csv
output:
  model_path: models/exp1.txt
  result_path: results/exp1.json
```

改后：
```yaml
data:
  train_files:
    - runs/baseline_2da_clean/features.csv
  test_files:
    - runs/baseline_2da_clean/features.csv
output:
  model_path: runs/spec_trainer/models/exp1.txt
  result_path: runs/spec_trainer/results/exp1.json
```

exp2.yaml 同理改 5da 数据集路径。

### 4.4 figures/ 路径

观察 main.py 第 100 行附近，figures 输出路径目前从 --name 推导（如 figures/<name>_roc.png）。本迁移采用方案 (a)：硬编码 figures/ 改为 runs/spec_trainer/figures/。

YAGNI 决策：方案 (b)（yaml 加 output.fig_dir）更灵活但要改更多代码，暂不做。

---

## 五、main.py 的导入路径修复

spec_trainer/src/main.py 第 1 行：

```python
from models.model_manager import ModelManager
```

这是相对 src/ 工作目录的隐式相对导入。从 ms2-met 根目录跑会失败。

修复：仿照 tools/eval_baseline.py:25-28 已有的 pattern，在 main.py 顶部添加：

```python
import os
import sys
_SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)
```

无需改 import 语句本身。

train.py / train2.py / models/*.py 内部互相导入也需检查；预期它们的相对导入都在 src/ 下，加 _SRC_ROOT 到 sys.path 后可解析。

---

## 六、主 Makefile 新增 target

```make
.PHONY: train-exp1 train-exp2 train-all clean-train

# features.csv 不存在时级联触发对应 make 2th/5th/normal
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
	@if [ -d runs/spec_trainer ]; then \\
		find runs/spec_trainer -mindepth 1 -delete 2>/dev/null || true; \\
		echo "[cleaned] runs/spec_trainer/"; \\
	else \\
		echo "[skip] runs/spec_trainer/ does not exist"; \\
	fi
```

help target 同步更新，列出新 target。

---

## 七、子项目自身 Makefile 的去留

tools/spec_trainer/Makefile 保留不动，但只能在子目录下用：

```bash
cd tools/spec_trainer && make exp1   # 子项目内部独立调用
```

此时工作目录是 tools/spec_trainer/，yaml 里 runs/baseline_2da_clean/... 的路径会指向 tools/spec_trainer/runs/... 不存在 → 报错。

这是已知不一致。本设计的优先级：主 Makefile 是首选入口，子项目 Makefile 保留是"备份"。文档明确建议：优先用 ms2-met 根目录的 make train-*。

---

## 八、不在本次范围

- 重写 41 个 yaml：只改 exp1/exp2 两个示范
- 训练代码重构 / 改进：仅做必要的导入路径修复
- 实际跑训练：数据 1.1GB，时间长，按用户需要再触发
- 删除原 ../spec_trainer/：用户保留兑底
- 模型管理子系统融合
- 子项目独立 Makefile 改路径

---

## 九、验证步骤（迁移后立即跑）

1. 主项目测试不受影响：conda run -n jianyan pytest tests/ → 266 passed
2. 主 Makefile help 列出新 target：make help → 列出 train-exp1
3. yaml 路径已更新：grep -E result_file tools/spec_trainer/config/exp1.yaml → 显示 runs/...
4. dry-run train-exp1：make -n train-exp1 → 显示级联 make 2th 然后跑 main.py
5. 导入路径修复验证：从根目录跑 python -c "import sys; sys.path.insert(0, \'tools/spec_trainer/src\'); import main"  → 无 ImportError
6. 子项目原 Makefile 仍可用：cd tools/spec_trainer && make help → 显示子项目 help

---

## 十、风险

| 风险 | 评估 | 应对 |
|------|------|------|
| from models.model_manager 等相对导入失败 | 高 | main.py 顶部加 _SRC_ROOT 到 sys.path |
| 41 个 yaml 中其他配置仍指向 data/ | 中 | 文档警告，用户用到再改 |
| spec_trainer 的 spec.log 默认在根目录 | 低 | --logpath 已支持 |
| subtree 合并冲突 | 低 | subtree add 是干净添加 |
| 训练数据巨大导致 train-* 在 CI 超时 | 高 | train-* 不进 CI |
| 子项目 Makefile 与主 Makefile 路径不一致 | 中 | 文档明确建议优先用主 Makefile |
