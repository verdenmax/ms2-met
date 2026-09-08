# ms2-met Makefile
# ---------------------------------------------------------------------------
# 三种数据集的特征提取流水线。每个 target 自动确保：
#   1. extract_common 生成的 JSON 已存在（否则先跑 extract_common 生成）
#   2. main.py 用对应 baseline_*/config.ini 跑特征提取
#
# 用法：
#   make 2th      # 2Da 窗口
#   make 5th      # 5Da 窗口
#   make normal   # Normal（变窗）
#   make all      # 三个全跑（顺序）
#   make help     # 显示所有 target
#
# JSON 路径从 extract_*.ini 的 result_file= 行动态抽取，所以您改 ini 后
# Makefile 会自动跟踪新位置。

.DEFAULT_GOAL := help

# --------------------------- 配置（可在命令行覆盖） ---------------------------

# Python 解释器（如果用 conda 环境，可改成 conda run -n jianyan python3）
PY ?= python3

# Formal training input/output roots. FEATURE_ROOT must directly contain the
# baseline_{2da,5da,normal}_{clean,neg05,...,neg20}/ directories. Keeping the
# feature snapshot separate makes later extraction updates reproducible.
FEATURE_ROOT ?= runs
CV_OUTPUT_ROOT ?= runs/spec_trainer
CV_CONFIG_DIR ?= $(CV_OUTPUT_ROOT)/configs
CV_FEATURE_ARM ?= evidence_all
FIXED_NEGPOOL_FEATURE_ARM ?= evidence_all
CV_OVERWRITE ?= 0
CV_OVERWRITE_FLAG = $(if $(filter 1 true yes,$(CV_OVERWRITE)),--overwrite,)
FIXED_NEGPOOL_OUTPUT_ROOT ?= $(CV_OUTPUT_ROOT)/fixed-negpool
FIXED_NEGPOOL_BOOTSTRAPS ?= 1000
FIXED_NEGPOOL_TEST_FRACTION ?= 0.20
DEEP_OUTPUT_ROOT ?= runs/deep_trainer
DEEP_CONFIG ?= tools/deep_trainer/config/tabular_mlp.yaml
DEEP_PROTOCOL_ROOT ?= $(FIXED_NEGPOOL_OUTPUT_ROOT)/combined
PHASE2_BUILD_CONFIG ?= tools/deep_trainer/phase2/config/raw_xic_pilot.yaml
PHASE2_XIC_OUTPUT_ROOT ?= $(DEEP_OUTPUT_ROOT)/phase2-xic/pilot
PHASE2_FULL_BUILD_CONFIG ?= tools/deep_trainer/phase2/config/raw_xic_full.yaml
PHASE2_FULL_XIC_OUTPUT_ROOT ?= $(DEEP_OUTPUT_ROOT)/phase2-xic/full
PHASE2_CACHE_ROOT ?= workspace
PHASE2_TRAIN_CONFIG ?= tools/deep_trainer/phase2/config/xic_fusion.yaml
PHASE2_TRAIN_OUTPUT_ROOT ?= $(DEEP_OUTPUT_ROOT)/phase2-xic-model/combined
PHASE2_STRONG_TRAIN_CONFIG ?= tools/deep_trainer/phase2/config/xic_pair_interaction.yaml
PHASE2_STRONG_TRAIN_OUTPUT_ROOT ?= $(DEEP_OUTPUT_ROOT)/phase2-xic-model/strong-combined

# Audited 2Da counterfactual pilot. Parent/negative adapters use the paths in
# their configs; mass-spectrum inputs live in the ordinary run-directory
# config.ini, matching every other feature-extraction target in this file.
COUNTERFACTUAL_2DA_PARENT_CONFIG ?= config/counterfactual/2da_label_dev_train.parents.ini
COUNTERFACTUAL_2DA_NEGATIVE_CONFIG ?= config/counterfactual/2da_label_dev_train.negatives.ini
COUNTERFACTUAL_2DA_DIR ?= runs/counterfactual_2da_label_dev_train
COUNTERFACTUAL_2DA_FEATURE_CONFIG ?= $(COUNTERFACTUAL_2DA_DIR)/config.ini
COUNTERFACTUAL_2DA_TRAIN_CONFIG ?= config/counterfactual/2da_label_dev_train.cv.yaml
COUNTERFACTUAL_2DA_GROUP_HOLDOUT_CONFIG ?= config/counterfactual/2da_group_holdout_experiment.yaml
COUNTERFACTUAL_2DA_FEATURES ?= $(COUNTERFACTUAL_2DA_DIR)/features.csv
COUNTERFACTUAL_2DA_ENTRAPMENT_FEATURES ?= $(FEATURE_ROOT)/baseline_2da_clean/features.csv
COUNTERFACTUAL_2DA_GROUP_HOLDOUT_ROOT ?= $(CV_OUTPUT_ROOT)/counterfactual-2da-group-holdout

# 一键过滤现有 features.csv 的目标范围（可命令行覆盖，如 runs_new/...）
# 例：make filter FILTER_GLOB='runs_new/baseline_*/features.csv'
FILTER_GLOB ?= runs/baseline_*/features.csv

# 三个 extract_common 配置文件
INI_2TH    ?= extract_2da_pfind_diann.ini
INI_5TH    ?= extract_5da_pfind_diann.ini
INI_NORMAL ?= extract_normal_pfind_diann.ini

# 三个 baseline 目录（含 config.ini + 输出位置）
DIR_2TH    ?= runs/baseline_2da_clean
DIR_5TH    ?= runs/baseline_5da_clean
DIR_NORMAL ?= runs/baseline_normal_clean

# Neg-FDR 变体 ini（dual-FDR：仅放宽负例池）
# 见 docs/specs/2026-06-03-neg-fdr-variants-design.md
INI_2TH_NEG05    ?= extract_2da_neg05.ini
INI_2TH_NEG10    ?= extract_2da_neg10.ini
INI_5TH_NEG05    ?= extract_5da_neg05.ini
INI_5TH_NEG10    ?= extract_5da_neg10.ini
INI_NORMAL_NEG05 ?= extract_normal_neg05.ini
INI_NORMAL_NEG10 ?= extract_normal_neg10.ini

# Neg-FDR 15%/20% 变体 ini
INI_2TH_NEG15    ?= extract_2da_neg15.ini
INI_2TH_NEG20    ?= extract_2da_neg20.ini
INI_5TH_NEG15    ?= extract_5da_neg15.ini
INI_5TH_NEG20    ?= extract_5da_neg20.ini
INI_NORMAL_NEG15 ?= extract_normal_neg15.ini
INI_NORMAL_NEG20 ?= extract_normal_neg20.ini

# Neg-FDR 变体 baseline 目录
DIR_2TH_NEG05    ?= runs/baseline_2da_neg05
DIR_2TH_NEG10    ?= runs/baseline_2da_neg10
DIR_5TH_NEG05    ?= runs/baseline_5da_neg05
DIR_5TH_NEG10    ?= runs/baseline_5da_neg10
DIR_NORMAL_NEG05 ?= runs/baseline_normal_neg05
DIR_NORMAL_NEG10 ?= runs/baseline_normal_neg10

# Neg-FDR 15%/20% 变体 baseline 目录
DIR_2TH_NEG15    ?= runs/baseline_2da_neg15
DIR_2TH_NEG20    ?= runs/baseline_2da_neg20
DIR_5TH_NEG15    ?= runs/baseline_5da_neg15
DIR_5TH_NEG20    ?= runs/baseline_5da_neg20
DIR_NORMAL_NEG15 ?= runs/baseline_normal_neg15
DIR_NORMAL_NEG20 ?= runs/baseline_normal_neg20

# pos50 变体：正例放宽至 FDR<=50%（q∈0..0.50），负例不变；用于观察召回随 FDR 衰减
INI_2TH_POS50    ?= extract_2da_pos50.ini
DIR_2TH_POS50    ?= runs/baseline_2da_pos50

# 从 extract_*.ini 中动态抽取 result_file 路径。
# 行格式：result_file = ./path/to/output.json    # optional comment
# 处理：
#   1) grep 抓首行匹配 ^result_file=
#   2) sed 砍掉 = 之前 + 行尾 #comment + 首尾空白 + 包围引号
#   3) 若 ini 不存在 -> 结果为空字符串（target 会用 ifeq 显式检查）
# 注意：保留路径中部的空格（用 sed 替代旧的 tr -d ' '）。
# 已知限制：路径中不能含 '#'（会被当作注释截断）；引号必须配对。
define EXTRACT_RESULT_FILE
$(strip $(shell test -f $(1) && \
    grep -E '^[[:space:]]*result_file[[:space:]]*=' $(1) | head -1 | \
    sed -E -e 's/^[^=]*=[[:space:]]*//' -e 's/[[:space:]]*#.*$$//' -e 's/^["'"'"']//' -e 's/["'"'"']$$//' -e 's/[[:space:]]+$$//'))
endef

JSON_2TH    := $(call EXTRACT_RESULT_FILE,$(INI_2TH))
JSON_5TH    := $(call EXTRACT_RESULT_FILE,$(INI_5TH))
JSON_NORMAL := $(call EXTRACT_RESULT_FILE,$(INI_NORMAL))

# Neg-FDR 变体 JSON 路径（从对应 ini 抽取）
JSON_2TH_NEG05    := $(call EXTRACT_RESULT_FILE,$(INI_2TH_NEG05))
JSON_2TH_NEG10    := $(call EXTRACT_RESULT_FILE,$(INI_2TH_NEG10))
JSON_5TH_NEG05    := $(call EXTRACT_RESULT_FILE,$(INI_5TH_NEG05))
JSON_5TH_NEG10    := $(call EXTRACT_RESULT_FILE,$(INI_5TH_NEG10))
JSON_NORMAL_NEG05 := $(call EXTRACT_RESULT_FILE,$(INI_NORMAL_NEG05))
JSON_NORMAL_NEG10 := $(call EXTRACT_RESULT_FILE,$(INI_NORMAL_NEG10))

# Neg-FDR 15%/20% 变体 JSON 路径
JSON_2TH_NEG15    := $(call EXTRACT_RESULT_FILE,$(INI_2TH_NEG15))
JSON_2TH_NEG20    := $(call EXTRACT_RESULT_FILE,$(INI_2TH_NEG20))
JSON_5TH_NEG15    := $(call EXTRACT_RESULT_FILE,$(INI_5TH_NEG15))
JSON_5TH_NEG20    := $(call EXTRACT_RESULT_FILE,$(INI_5TH_NEG20))
JSON_NORMAL_NEG15 := $(call EXTRACT_RESULT_FILE,$(INI_NORMAL_NEG15))
JSON_NORMAL_NEG20 := $(call EXTRACT_RESULT_FILE,$(INI_NORMAL_NEG20))

# pos50 变体 JSON 路径
JSON_2TH_POS50    := $(call EXTRACT_RESULT_FILE,$(INI_2TH_POS50))

# --------------------------- 工具 / banner ---------------------------

# 优先用 toilet | lolcat（漂亮），缺失则退回 echo
define BANNER
@command -v toilet >/dev/null 2>&1 && command -v lolcat >/dev/null 2>&1 \
&& toilet -f mono12 "$(1)" | lolcat \
|| echo "==================== $(1) ===================="
endef

# --------------------------- 主 target ---------------------------

.PHONY: help all run clean build-deep-xic-pilot build-deep-xic-full train-deep-xic-combined smoke-deep-xic-cuda train-deep-xic-strong-combined
.PHONY: 2th 5th normal
.PHONY: filter filter-dry
.PHONY: extract-2th extract-5th extract-normal
.PHONY: clean-2th clean-5th clean-normal clean-all
.PHONY: 2th-neg05 2th-neg10 5th-neg05 5th-neg10 normal-neg05 normal-neg10
.PHONY: extract-2th-neg05 extract-2th-neg10 extract-5th-neg05 extract-5th-neg10 extract-normal-neg05 extract-normal-neg10
.PHONY: clean-2th-neg05 clean-2th-neg10 clean-5th-neg05 clean-5th-neg10 clean-normal-neg05 clean-normal-neg10
.PHONY: all-clean all-neg05 all-neg10

.PHONY: 2th-neg15 2th-neg20 5th-neg15 5th-neg20 normal-neg15 normal-neg20
.PHONY: extract-2th-neg15 extract-2th-neg20 extract-5th-neg15 extract-5th-neg20 extract-normal-neg15 extract-normal-neg20
.PHONY: clean-2th-neg15 clean-2th-neg20 clean-5th-neg15 clean-5th-neg20 clean-normal-neg15 clean-normal-neg20
.PHONY: all-neg15 all-neg20

.PHONY: 2th-pos50 extract-2th-pos50 pos50-2da
.PHONY: counterfactual-2da counterfactual-2da-parents
.PHONY: counterfactual-2da-negatives counterfactual-2da-features
.PHONY: counterfactual-2da-group-holdout
.PHONY: counterfactual-2da-group-holdout-build counterfactual-2da-group-holdout-train

help:
	@echo "ms2-met Makefile — 三种数据集的特征提取流水线"
	@echo ""
	@echo "  make 2th             跑 2Da 数据集（runs/baseline_2da_clean/）"
	@echo "  make 5th             跑 5Da 数据集（runs/baseline_5da_clean/）"
	@echo "  make normal          跑 Normal 数据集（runs/baseline_normal_clean/）"
	@echo "  make all             顺序跑 2th / 5th / normal"
	@echo ""
	@echo "  make extract-2th     仅生成 2da 的 input JSON"
	@echo "  make extract-5th     仅生成 5da 的 input JSON"
	@echo "  make extract-normal  仅生成 normal 的 input JSON"
	@echo ""
	@echo "  Counterfactual 2Da pilot（沿用 2Da config.ini 的质谱配置）："
	@echo "  make counterfactual-2da-parents    从已过滤 JSON positive 准备 parent"
	@echo "  make counterfactual-2da-negatives  8 进程生成三类候选（默认 5000 parent pilot）"
	@echo "  make counterfactual-2da-features   提取普通 light/heavy 特征"
	@echo "  make counterfactual-2da            顺序执行以上三步"
	@echo "  make counterfactual-2da-group-holdout       构建统一分组留出集并训练 M-C/M-K/M-L/M-All"
	@echo "  make counterfactual-2da-group-holdout-build 仅冻结四套训练集与同一真实 entrapment 测试集"
	@echo "  make counterfactual-2da-group-holdout-train 训练已经冻结的四套数据"
	@echo ""
	@echo "  注：extract-* 仅在对应 extract_*.ini 存在时可用。"
	@echo "      5th / normal 的 ini 默认未提供，features.csv 须外部生成。"
	@echo ""
	@echo ""
	@echo "  Neg-FDR 变体（dual-FDR，仅放宽负例 FDR；正例保持 1%）："
	@echo "  make 2th-neg05       2Da × negative FDR 5%"
	@echo "  make 2th-neg10       2Da × negative FDR 10%"
	@echo "  make 5th-neg05       5Da × negative FDR 5%"
	@echo "  make 5th-neg10       5Da × negative FDR 10%"
	@echo "  make normal-neg05    Normal × negative FDR 5%"
	@echo "  make normal-neg10    Normal × negative FDR 10%"
	@echo "  make all-clean       别名：make all（FDR 1%）"
	@echo "  make all-neg05       三个数据集 × negative FDR 5%"
	@echo "  make all-neg10       三个数据集 × negative FDR 10%"
	@echo "  make all-neg15       三个数据集 × negative FDR 15%"
	@echo "  make all-neg20       三个数据集 × negative FDR 20%"
	@echo ""
	@echo "  make extract-2th-neg05  仅生成 2da neg05 JSON（其他类同）"
	@echo "  make clean-2th-neg05    删除 2da neg05 features.csv（其他类同）"
	@echo ""
	@echo "  make clean-2th       删除 2da features.csv / log，强制下次重跑"
	@echo "  make clean-5th       删除 5da features.csv / log"
	@echo "  make clean-normal    删除 normal features.csv / log"
	@echo "  make clean-all       上述三者全清"
	@echo ""
	@echo "  make filter-dry      预览：列出各 features.csv 会删多少 heavy-out-of-range 行（不改文件）"
	@echo "  make filter          就地过滤现有 features.csv（删 heavy_out_of_range==1，备份 *.prefilter.bak）"
	@echo "                       范围由 FILTER_GLOB 控制，默认 runs/baseline_*/features.csv"
	@echo "                       例：make filter FILTER_GLOB='runs_new/baseline_*/features.csv'"
	@echo "                       注：新提取已在 main.py 内自动过滤，本目标用于旧文件补过滤"
	@echo ""
	@echo "  make train-exp1         旧实验：训练 exp1（2da 单独）"
	@echo "  make train-exp2         旧实验：训练 exp2（2da + 5da combined）"
	@echo "  make train-legacy-all   旧组合：train-exp1 + train-exp2"
	@echo ""
	@echo "  Systematic training matrix（18 实验：3 FDR × 2 schemes × 3 datasets）："
	@echo "  make train-clean-all    6 个 clean（1% FDR）实验"
	@echo "  make train-neg05-all    6 个 neg05 实验"
	@echo "  make train-neg10-all    6 个 neg10 实验"
	@echo "  make train-neg15-all    6 个 neg15 实验"
	@echo "  make train-neg20-all    6 个 neg20 实验"
	@echo "  make train-all          所有 30 个实验（clean → neg05 → neg10 → neg15 → neg20）"
	@echo ""
	@echo "  CV(5 折分组 CV + 折间 ensemble + 标签审计；in-sample + cross_test)："
	@echo "  make train-cv-2da       单个 2da clean CV 实验"
	@echo "  make train-cv-clean-all 6 个 clean CV 实验（3 in-sample + 3 cross_test）"
	@echo "  make train-cv-neg05-all 6 个 neg05 CV 实验"
	@echo "  make train-cv-neg10-all 6 个 neg10 CV 实验"
	@echo "  make train-cv-neg15-all 6 个 neg15 CV 实验"
	@echo "  make train-cv-neg20-all 6 个 neg20 CV 实验"
	@echo "  make train-cv-all FEATURE_ROOT=/path/to/results  30 个 evidence_all CV 实验"
	@echo "  make train-cv-core-all FEATURE_ROOT=/path/to/results  30 个 evidence_core CV 实验"
	@echo "  make train-fixed-test-negpool-2da FEATURE_ROOT=/path/to/results  固定 E20 测试集比较 M5/M10/M20"
	@echo "  make train-fixed-test-negpool-all FEATURE_ROOT=/path/to/results  三数据集固定测试实验"
	@echo "  make train-fixed-test-negpool-combined FEATURE_ROOT=/path/to/results  三数据集合并后全局固定测试"
	@echo "  make train-deep-mlp-combined FEATURE_ROOT=/path/to/results  同一 combined E20 协议训练表格 MLP"
	@echo "  make build-deep-xic-pilot FEATURE_ROOT=/path/to/results  构建 1200 条 Phase 2 原始 XIC 完整性 pilot"
	@echo "  make build-deep-xic-full FEATURE_ROOT=/path/to/results   可恢复地构建全量 Phase 2 XIC"
	@echo "  make train-deep-xic-combined FEATURE_ROOT=/path/to/results  冻结 E20 上训练 15 成员 XIC 模型"
	@echo "  make smoke-deep-xic-cuda                         严格确定性 CUDA 前后向冒烟测试"
	@echo "  make train-deep-xic-strong-combined FEATURE_ROOT=/path/to/results  训练增强的 pair-interaction XIC 模型"
	@echo ""
	@echo "  Formal MS1/MS2 ablation（共同队列 + 注册表特征组）："
	@echo "  make train-ablation-neg20-2da FEATURE_ROOT=/path/to/results  2da 预跑（8 组）"
	@echo "  make train-ablation-neg20 FEATURE_ROOT=/path/to/results      三数据集（24 组）"
	@echo ""
	@echo "  make clean-train        清理 runs/spec_trainer/ 训练产出"
	@echo "  make clean           旧式清理（checkpoint.pkl 等）"
	@echo ""
	@echo "  make run             旧 target：跑 main.py 使用根目录 config.ini"
	@echo ""
	@echo "当前抽取的 JSON 路径："
	@echo "  2th         -> $(JSON_2TH)"
	@echo "  5th         -> $(JSON_5TH)"
	@echo "  normal      -> $(JSON_NORMAL)"
	@echo "  2th-neg05   -> $(JSON_2TH_NEG05)"
	@echo "  2th-neg10   -> $(JSON_2TH_NEG10)"
	@echo "  5th-neg05   -> $(JSON_5TH_NEG05)"
	@echo "  5th-neg10   -> $(JSON_5TH_NEG10)"
	@echo "  normal-neg05-> $(JSON_NORMAL_NEG05)"
	@echo "  normal-neg10-> $(JSON_NORMAL_NEG10)"
	@echo "  2th-neg15  -> $(JSON_2TH_NEG15)"
	@echo "  2th-neg20  -> $(JSON_2TH_NEG20)"
	@echo "  5th-neg15  -> $(JSON_5TH_NEG15)"
	@echo "  5th-neg20  -> $(JSON_5TH_NEG20)"
	@echo "  normal-neg15-> $(JSON_NORMAL_NEG15)"
	@echo "  normal-neg20-> $(JSON_NORMAL_NEG20)"

# ---------------- Counterfactual 2Da pilot ----------------

counterfactual-2da-parents: $(COUNTERFACTUAL_2DA_PARENT_CONFIG)
	$(call BANNER,counterfactual parents)
	$(PY) -m tools.counterfactual_parents --config $(COUNTERFACTUAL_2DA_PARENT_CONFIG)

counterfactual-2da-negatives: \
	counterfactual-2da-parents \
	$(COUNTERFACTUAL_2DA_NEGATIVE_CONFIG)
	$(call BANNER,counterfactual negatives)
	$(PY) -m tools.counterfactual_negatives --config $(COUNTERFACTUAL_2DA_NEGATIVE_CONFIG)

counterfactual-2da-features: \
	counterfactual-2da-negatives \
	$(COUNTERFACTUAL_2DA_FEATURE_CONFIG)
	$(call BANNER,counterfactual features)
	$(PY) main.py --configpath $(COUNTERFACTUAL_2DA_FEATURE_CONFIG) \
		--logpath $(COUNTERFACTUAL_2DA_DIR)/extract.log

counterfactual-2da: counterfactual-2da-features

.PHONY: counterfactual-2da-train
# Train an existing feature snapshot; do not regenerate a dataset implicitly.
counterfactual-2da-train: $(COUNTERFACTUAL_2DA_TRAIN_CONFIG)
	$(PY) tools/spec_trainer/src/cv_train.py --config $(COUNTERFACTUAL_2DA_TRAIN_CONFIG) \
		--name counterfactual_2da_label_dev_train \
		--logpath $(COUNTERFACTUAL_2DA_DIR)/train.log

# Freeze one peptide/family-held-out split before training. Entrapment errors
# are always test-only; replicate/raw names never determine the split.
counterfactual-2da-group-holdout-build: $(COUNTERFACTUAL_2DA_GROUP_HOLDOUT_CONFIG)
	$(call BANNER,counterfactual grouped holdout build)
	$(PY) -m tools.counterfactual_group_holdout \
		--config $(COUNTERFACTUAL_2DA_GROUP_HOLDOUT_CONFIG) \
		--counterfactual-features $(COUNTERFACTUAL_2DA_FEATURES) \
		--entrapment-features $(COUNTERFACTUAL_2DA_ENTRAPMENT_FEATURES) \
		--output-root $(COUNTERFACTUAL_2DA_GROUP_HOLDOUT_ROOT)

counterfactual-2da-group-holdout-train:
	$(call BANNER,counterfactual grouped holdout train)
	@test -f $(COUNTERFACTUAL_2DA_GROUP_HOLDOUT_ROOT)/bundle_status.json || \
		{ echo "missing complete bundle: $(COUNTERFACTUAL_2DA_GROUP_HOLDOUT_ROOT)"; exit 1; }
	@set -e; for variant in m_c m_k m_l m_all; do \
		$(PY) tools/spec_trainer/src/cv_train.py \
			--config $(COUNTERFACTUAL_2DA_GROUP_HOLDOUT_ROOT)/configs/$$variant.yaml \
			--name counterfactual_2da_group_holdout_$$variant \
			--logpath $(COUNTERFACTUAL_2DA_GROUP_HOLDOUT_ROOT)/training/$$variant/train.log \
			$(CV_OVERWRITE_FLAG); \
	done

counterfactual-2da-group-holdout:
	$(MAKE) counterfactual-2da-group-holdout-build \
		COUNTERFACTUAL_2DA_FEATURES=$(COUNTERFACTUAL_2DA_FEATURES) \
		COUNTERFACTUAL_2DA_ENTRAPMENT_FEATURES=$(COUNTERFACTUAL_2DA_ENTRAPMENT_FEATURES) \
		COUNTERFACTUAL_2DA_GROUP_HOLDOUT_ROOT=$(COUNTERFACTUAL_2DA_GROUP_HOLDOUT_ROOT)
	$(MAKE) counterfactual-2da-group-holdout-train \
		COUNTERFACTUAL_2DA_GROUP_HOLDOUT_ROOT=$(COUNTERFACTUAL_2DA_GROUP_HOLDOUT_ROOT) \
		CV_OVERWRITE=$(CV_OVERWRITE)


# ---------- 2th ----------
#
# If extract_2da_pfind_diann.ini exists, declare full pipeline dependency
# (JSON regenerated when ini changes). Otherwise the JSON / features.csv
# is treated as externally-provided; we still run main.py if the user
# explicitly invokes `make 2th`, but only after verifying the necessary
# inputs already exist on disk.
ifneq ($(wildcard $(INI_2TH)),)

# JSON 缺失时自动跑 extract_common
$(JSON_2TH): $(INI_2TH)
	$(call BANNER,extract 2th)
	$(PY) tools/extract_common.py --configpath $(INI_2TH)

extract-2th: $(JSON_2TH)

2th: $(INI_2TH) $(JSON_2TH) $(DIR_2TH)/config.ini
	$(call BANNER,2th)
	$(PY) main.py --configpath $(DIR_2TH)/config.ini --logpath $(DIR_2TH)/extract.log
	@echo "[done] features written under $(DIR_2TH)/"

else  # $(INI_2TH) absent — features.csv must be externally provided

extract-2th:
	@echo "[error] $(INI_2TH) not found — cannot extract; provide ini or use a pre-built JSON" >&2
	@false

2th: $(DIR_2TH)/config.ini
	$(call BANNER,2th)
	@if [ ! -f "$(DIR_2TH)/features.csv" ]; then \
		echo "[note] $(INI_2TH) absent — $(DIR_2TH)/features.csv must be present" >&2; \
		echo "       (extract step skipped; main.py will fail if light_result_file in config.ini is invalid)" >&2; \
	fi
	$(PY) main.py --configpath $(DIR_2TH)/config.ini --logpath $(DIR_2TH)/extract.log
	@echo "[done] features written under $(DIR_2TH)/"

endif

# ---------- 5th ----------
#
# If extract_5da_pfind_diann.ini exists, declare full pipeline dependency
# (JSON regenerated when ini changes). Otherwise the JSON / features.csv
# is treated as externally-provided; we still run main.py if the user
# explicitly invokes `make 5th`, but only after verifying the necessary
# inputs already exist on disk.
ifneq ($(wildcard $(INI_5TH)),)

$(JSON_5TH): $(INI_5TH)
	$(call BANNER,extract 5th)
	$(PY) tools/extract_common.py --configpath $(INI_5TH)

extract-5th: $(JSON_5TH)

5th: $(INI_5TH) $(JSON_5TH) $(DIR_5TH)/config.ini
	$(call BANNER,5th)
	$(PY) main.py --configpath $(DIR_5TH)/config.ini --logpath $(DIR_5TH)/extract.log
	@echo "[done] features written under $(DIR_5TH)/"

else  # $(INI_5TH) absent — features.csv must be externally provided

extract-5th:
	@echo "[error] $(INI_5TH) not found — cannot extract; provide ini or use a pre-built JSON" >&2
	@false

5th: $(DIR_5TH)/config.ini
	$(call BANNER,5th)
	@if [ ! -f "$(DIR_5TH)/features.csv" ]; then \
		echo "[note] $(INI_5TH) absent — $(DIR_5TH)/features.csv must be present" >&2; \
		echo "       (extract step skipped; main.py will fail if light_result_file in config.ini is invalid)" >&2; \
	fi
	$(PY) main.py --configpath $(DIR_5TH)/config.ini --logpath $(DIR_5TH)/extract.log
	@echo "[done] features written under $(DIR_5TH)/"

endif

# ---------- normal ----------
#
# If extract_normal_pfind_diann.ini exists, declare full pipeline dependency
# (JSON regenerated when ini changes). Otherwise the JSON / features.csv
# is treated as externally-provided; we still run main.py if the user
# explicitly invokes `make normal`, but only after verifying the necessary
# inputs already exist on disk.
ifneq ($(wildcard $(INI_NORMAL)),)

$(JSON_NORMAL): $(INI_NORMAL)
	$(call BANNER,extract normal)
	$(PY) tools/extract_common.py --configpath $(INI_NORMAL)

extract-normal: $(JSON_NORMAL)

normal: $(INI_NORMAL) $(JSON_NORMAL) $(DIR_NORMAL)/config.ini
	$(call BANNER,normal)
	$(PY) main.py --configpath $(DIR_NORMAL)/config.ini --logpath $(DIR_NORMAL)/extract.log
	@echo "[done] features written under $(DIR_NORMAL)/"

else  # $(INI_NORMAL) absent — features.csv must be externally provided

extract-normal:
	@echo "[error] $(INI_NORMAL) not found — cannot extract; provide ini or use a pre-built JSON" >&2
	@false

normal: $(DIR_NORMAL)/config.ini
	$(call BANNER,normal)
	@if [ ! -f "$(DIR_NORMAL)/features.csv" ]; then \
		echo "[note] $(INI_NORMAL) absent — $(DIR_NORMAL)/features.csv must be present" >&2; \
		echo "       (extract step skipped; main.py will fail if light_result_file in config.ini is invalid)" >&2; \
	fi
	$(PY) main.py --configpath $(DIR_NORMAL)/config.ini --logpath $(DIR_NORMAL)/extract.log
	@echo "[done] features written under $(DIR_NORMAL)/"

endif

# ---------- 2th-neg05 ----------
ifneq ($(wildcard $(INI_2TH_NEG05)),)

$(JSON_2TH_NEG05): $(INI_2TH_NEG05)
	$(call BANNER,extract 2th-neg05)
	$(PY) tools/extract_common.py --configpath $(INI_2TH_NEG05)

extract-2th-neg05: $(JSON_2TH_NEG05)

2th-neg05: $(INI_2TH_NEG05) $(JSON_2TH_NEG05) $(DIR_2TH_NEG05)/config.ini
	$(call BANNER,2th-neg05)
	$(PY) main.py --configpath $(DIR_2TH_NEG05)/config.ini --logpath $(DIR_2TH_NEG05)/extract.log
	@echo "[done] features written under $(DIR_2TH_NEG05)/"

else  # $(INI_2TH_NEG05) absent — features.csv must be externally provided

extract-2th-neg05:
	@echo "[error] $(INI_2TH_NEG05) not found — cannot extract; provide ini or use a pre-built JSON" >&2
	@false

2th-neg05: $(DIR_2TH_NEG05)/config.ini
	$(call BANNER,2th-neg05)
	@if [ ! -f "$(DIR_2TH_NEG05)/features.csv" ]; then \
		echo "[note] $(INI_2TH_NEG05) absent — $(DIR_2TH_NEG05)/features.csv must be present" >&2; \
		echo "       (extract step skipped; main.py will fail if light_result_file in config.ini is invalid)" >&2; \
	fi
	$(PY) main.py --configpath $(DIR_2TH_NEG05)/config.ini --logpath $(DIR_2TH_NEG05)/extract.log
	@echo "[done] features written under $(DIR_2TH_NEG05)/"

endif

# ---------- 2th-neg10 ----------
ifneq ($(wildcard $(INI_2TH_NEG10)),)

$(JSON_2TH_NEG10): $(INI_2TH_NEG10)
	$(call BANNER,extract 2th-neg10)
	$(PY) tools/extract_common.py --configpath $(INI_2TH_NEG10)

extract-2th-neg10: $(JSON_2TH_NEG10)

2th-neg10: $(INI_2TH_NEG10) $(JSON_2TH_NEG10) $(DIR_2TH_NEG10)/config.ini
	$(call BANNER,2th-neg10)
	$(PY) main.py --configpath $(DIR_2TH_NEG10)/config.ini --logpath $(DIR_2TH_NEG10)/extract.log
	@echo "[done] features written under $(DIR_2TH_NEG10)/"

else  # $(INI_2TH_NEG10) absent — features.csv must be externally provided

extract-2th-neg10:
	@echo "[error] $(INI_2TH_NEG10) not found — cannot extract; provide ini or use a pre-built JSON" >&2
	@false

2th-neg10: $(DIR_2TH_NEG10)/config.ini
	$(call BANNER,2th-neg10)
	@if [ ! -f "$(DIR_2TH_NEG10)/features.csv" ]; then \
		echo "[note] $(INI_2TH_NEG10) absent — $(DIR_2TH_NEG10)/features.csv must be present" >&2; \
		echo "       (extract step skipped; main.py will fail if light_result_file in config.ini is invalid)" >&2; \
	fi
	$(PY) main.py --configpath $(DIR_2TH_NEG10)/config.ini --logpath $(DIR_2TH_NEG10)/extract.log
	@echo "[done] features written under $(DIR_2TH_NEG10)/"

endif

# ---------- 5th-neg05 ----------
ifneq ($(wildcard $(INI_5TH_NEG05)),)

$(JSON_5TH_NEG05): $(INI_5TH_NEG05)
	$(call BANNER,extract 5th-neg05)
	$(PY) tools/extract_common.py --configpath $(INI_5TH_NEG05)

extract-5th-neg05: $(JSON_5TH_NEG05)

5th-neg05: $(INI_5TH_NEG05) $(JSON_5TH_NEG05) $(DIR_5TH_NEG05)/config.ini
	$(call BANNER,5th-neg05)
	$(PY) main.py --configpath $(DIR_5TH_NEG05)/config.ini --logpath $(DIR_5TH_NEG05)/extract.log
	@echo "[done] features written under $(DIR_5TH_NEG05)/"

else  # $(INI_5TH_NEG05) absent — features.csv must be externally provided

extract-5th-neg05:
	@echo "[error] $(INI_5TH_NEG05) not found — cannot extract; provide ini or use a pre-built JSON" >&2
	@false

5th-neg05: $(DIR_5TH_NEG05)/config.ini
	$(call BANNER,5th-neg05)
	@if [ ! -f "$(DIR_5TH_NEG05)/features.csv" ]; then \
		echo "[note] $(INI_5TH_NEG05) absent — $(DIR_5TH_NEG05)/features.csv must be present" >&2; \
		echo "       (extract step skipped; main.py will fail if light_result_file in config.ini is invalid)" >&2; \
	fi
	$(PY) main.py --configpath $(DIR_5TH_NEG05)/config.ini --logpath $(DIR_5TH_NEG05)/extract.log
	@echo "[done] features written under $(DIR_5TH_NEG05)/"

endif

# ---------- 5th-neg10 ----------
ifneq ($(wildcard $(INI_5TH_NEG10)),)

$(JSON_5TH_NEG10): $(INI_5TH_NEG10)
	$(call BANNER,extract 5th-neg10)
	$(PY) tools/extract_common.py --configpath $(INI_5TH_NEG10)

extract-5th-neg10: $(JSON_5TH_NEG10)

5th-neg10: $(INI_5TH_NEG10) $(JSON_5TH_NEG10) $(DIR_5TH_NEG10)/config.ini
	$(call BANNER,5th-neg10)
	$(PY) main.py --configpath $(DIR_5TH_NEG10)/config.ini --logpath $(DIR_5TH_NEG10)/extract.log
	@echo "[done] features written under $(DIR_5TH_NEG10)/"

else  # $(INI_5TH_NEG10) absent — features.csv must be externally provided

extract-5th-neg10:
	@echo "[error] $(INI_5TH_NEG10) not found — cannot extract; provide ini or use a pre-built JSON" >&2
	@false

5th-neg10: $(DIR_5TH_NEG10)/config.ini
	$(call BANNER,5th-neg10)
	@if [ ! -f "$(DIR_5TH_NEG10)/features.csv" ]; then \
		echo "[note] $(INI_5TH_NEG10) absent — $(DIR_5TH_NEG10)/features.csv must be present" >&2; \
		echo "       (extract step skipped; main.py will fail if light_result_file in config.ini is invalid)" >&2; \
	fi
	$(PY) main.py --configpath $(DIR_5TH_NEG10)/config.ini --logpath $(DIR_5TH_NEG10)/extract.log
	@echo "[done] features written under $(DIR_5TH_NEG10)/"

endif

# ---------- normal-neg05 ----------
ifneq ($(wildcard $(INI_NORMAL_NEG05)),)

$(JSON_NORMAL_NEG05): $(INI_NORMAL_NEG05)
	$(call BANNER,extract normal-neg05)
	$(PY) tools/extract_common.py --configpath $(INI_NORMAL_NEG05)

extract-normal-neg05: $(JSON_NORMAL_NEG05)

normal-neg05: $(INI_NORMAL_NEG05) $(JSON_NORMAL_NEG05) $(DIR_NORMAL_NEG05)/config.ini
	$(call BANNER,normal-neg05)
	$(PY) main.py --configpath $(DIR_NORMAL_NEG05)/config.ini --logpath $(DIR_NORMAL_NEG05)/extract.log
	@echo "[done] features written under $(DIR_NORMAL_NEG05)/"

else  # $(INI_NORMAL_NEG05) absent — features.csv must be externally provided

extract-normal-neg05:
	@echo "[error] $(INI_NORMAL_NEG05) not found — cannot extract; provide ini or use a pre-built JSON" >&2
	@false

normal-neg05: $(DIR_NORMAL_NEG05)/config.ini
	$(call BANNER,normal-neg05)
	@if [ ! -f "$(DIR_NORMAL_NEG05)/features.csv" ]; then \
		echo "[note] $(INI_NORMAL_NEG05) absent — $(DIR_NORMAL_NEG05)/features.csv must be present" >&2; \
		echo "       (extract step skipped; main.py will fail if light_result_file in config.ini is invalid)" >&2; \
	fi
	$(PY) main.py --configpath $(DIR_NORMAL_NEG05)/config.ini --logpath $(DIR_NORMAL_NEG05)/extract.log
	@echo "[done] features written under $(DIR_NORMAL_NEG05)/"

endif

# ---------- normal-neg10 ----------
ifneq ($(wildcard $(INI_NORMAL_NEG10)),)

$(JSON_NORMAL_NEG10): $(INI_NORMAL_NEG10)
	$(call BANNER,extract normal-neg10)
	$(PY) tools/extract_common.py --configpath $(INI_NORMAL_NEG10)

extract-normal-neg10: $(JSON_NORMAL_NEG10)

normal-neg10: $(INI_NORMAL_NEG10) $(JSON_NORMAL_NEG10) $(DIR_NORMAL_NEG10)/config.ini
	$(call BANNER,normal-neg10)
	$(PY) main.py --configpath $(DIR_NORMAL_NEG10)/config.ini --logpath $(DIR_NORMAL_NEG10)/extract.log
	@echo "[done] features written under $(DIR_NORMAL_NEG10)/"

else  # $(INI_NORMAL_NEG10) absent — features.csv must be externally provided

extract-normal-neg10:
	@echo "[error] $(INI_NORMAL_NEG10) not found — cannot extract; provide ini or use a pre-built JSON" >&2
	@false

normal-neg10: $(DIR_NORMAL_NEG10)/config.ini
	$(call BANNER,normal-neg10)
	@if [ ! -f "$(DIR_NORMAL_NEG10)/features.csv" ]; then \
		echo "[note] $(INI_NORMAL_NEG10) absent — $(DIR_NORMAL_NEG10)/features.csv must be present" >&2; \
		echo "       (extract step skipped; main.py will fail if light_result_file in config.ini is invalid)" >&2; \
	fi
	$(PY) main.py --configpath $(DIR_NORMAL_NEG10)/config.ini --logpath $(DIR_NORMAL_NEG10)/extract.log
	@echo "[done] features written under $(DIR_NORMAL_NEG10)/"

endif

# ---------- 2th-neg15 ----------
ifneq ($(wildcard $(INI_2TH_NEG15)),)

$(JSON_2TH_NEG15): $(INI_2TH_NEG15)
	$(call BANNER,extract 2th-neg15)
	$(PY) tools/extract_common.py --configpath $(INI_2TH_NEG15)

extract-2th-neg15: $(JSON_2TH_NEG15)

2th-neg15: $(INI_2TH_NEG15) $(JSON_2TH_NEG15) $(DIR_2TH_NEG15)/config.ini
	$(call BANNER,2th-neg15)
	$(PY) main.py --configpath $(DIR_2TH_NEG15)/config.ini --logpath $(DIR_2TH_NEG15)/extract.log
	@echo "[done] features written under $(DIR_2TH_NEG15)/"

else  # $(INI_2TH_NEG15) absent — features.csv must be externally provided

extract-2th-neg15:
	@echo "[error] $(INI_2TH_NEG15) not found — cannot extract; provide ini or use a pre-built JSON" >&2
	@false

2th-neg15: $(DIR_2TH_NEG15)/config.ini
	$(call BANNER,2th-neg15)
	@if [ ! -f "$(DIR_2TH_NEG15)/features.csv" ]; then \
		echo "[note] $(INI_2TH_NEG15) absent — $(DIR_2TH_NEG15)/features.csv must be present" >&2; \
		echo "       (extract step skipped; main.py will fail if light_result_file in config.ini is invalid)" >&2; \
	fi
	$(PY) main.py --configpath $(DIR_2TH_NEG15)/config.ini --logpath $(DIR_2TH_NEG15)/extract.log
	@echo "[done] features written under $(DIR_2TH_NEG15)/"

endif

# ---------- 2th-neg20 ----------
ifneq ($(wildcard $(INI_2TH_NEG20)),)

$(JSON_2TH_NEG20): $(INI_2TH_NEG20)
	$(call BANNER,extract 2th-neg20)
	$(PY) tools/extract_common.py --configpath $(INI_2TH_NEG20)

extract-2th-neg20: $(JSON_2TH_NEG20)

2th-neg20: $(INI_2TH_NEG20) $(JSON_2TH_NEG20) $(DIR_2TH_NEG20)/config.ini
	$(call BANNER,2th-neg20)
	$(PY) main.py --configpath $(DIR_2TH_NEG20)/config.ini --logpath $(DIR_2TH_NEG20)/extract.log
	@echo "[done] features written under $(DIR_2TH_NEG20)/"

else  # $(INI_2TH_NEG20) absent — features.csv must be externally provided

extract-2th-neg20:
	@echo "[error] $(INI_2TH_NEG20) not found — cannot extract; provide ini or use a pre-built JSON" >&2
	@false

2th-neg20: $(DIR_2TH_NEG20)/config.ini
	$(call BANNER,2th-neg20)
	@if [ ! -f "$(DIR_2TH_NEG20)/features.csv" ]; then \
		echo "[note] $(INI_2TH_NEG20) absent — $(DIR_2TH_NEG20)/features.csv must be present" >&2; \
		echo "       (extract step skipped; main.py will fail if light_result_file in config.ini is invalid)" >&2; \
	fi
	$(PY) main.py --configpath $(DIR_2TH_NEG20)/config.ini --logpath $(DIR_2TH_NEG20)/extract.log
	@echo "[done] features written under $(DIR_2TH_NEG20)/"

endif

# ---------- 2th-pos50 ----------
ifneq ($(wildcard $(INI_2TH_POS50)),)

$(JSON_2TH_POS50): $(INI_2TH_POS50)
	$(call BANNER,extract 2th-pos50)
	$(PY) tools/extract_common.py --configpath $(INI_2TH_POS50)

extract-2th-pos50: $(JSON_2TH_POS50)

2th-pos50: $(INI_2TH_POS50) $(JSON_2TH_POS50) $(DIR_2TH_POS50)/config.ini
	$(call BANNER,2th-pos50)
	$(PY) main.py --configpath $(DIR_2TH_POS50)/config.ini --logpath $(DIR_2TH_POS50)/extract.log
	@echo "[done] features written under $(DIR_2TH_POS50)/"

else  # $(INI_2TH_POS50) absent — features.csv must be externally provided

extract-2th-pos50:
	@echo "[error] $(INI_2TH_POS50) not found — cannot extract; provide ini or use a pre-built JSON" >&2
	@false

2th-pos50: $(DIR_2TH_POS50)/config.ini
	$(call BANNER,2th-pos50)
	@if [ ! -f "$(DIR_2TH_POS50)/features.csv" ]; then \
		echo "[note] $(INI_2TH_POS50) absent — $(DIR_2TH_POS50)/features.csv must be present" >&2; \
		echo "       (extract step skipped; main.py will fail if light_result_file in config.ini is invalid)" >&2; \
	fi
	$(PY) main.py --configpath $(DIR_2TH_POS50)/config.ini --logpath $(DIR_2TH_POS50)/extract.log
	@echo "[done] features written under $(DIR_2TH_POS50)/"

endif

# pos50-2da：2th-pos50 别名
pos50-2da: 2th-pos50

# ---------- 5th-neg15 ----------
ifneq ($(wildcard $(INI_5TH_NEG15)),)

$(JSON_5TH_NEG15): $(INI_5TH_NEG15)
	$(call BANNER,extract 5th-neg15)
	$(PY) tools/extract_common.py --configpath $(INI_5TH_NEG15)

extract-5th-neg15: $(JSON_5TH_NEG15)

5th-neg15: $(INI_5TH_NEG15) $(JSON_5TH_NEG15) $(DIR_5TH_NEG15)/config.ini
	$(call BANNER,5th-neg15)
	$(PY) main.py --configpath $(DIR_5TH_NEG15)/config.ini --logpath $(DIR_5TH_NEG15)/extract.log
	@echo "[done] features written under $(DIR_5TH_NEG15)/"

else  # $(INI_5TH_NEG15) absent — features.csv must be externally provided

extract-5th-neg15:
	@echo "[error] $(INI_5TH_NEG15) not found — cannot extract; provide ini or use a pre-built JSON" >&2
	@false

5th-neg15: $(DIR_5TH_NEG15)/config.ini
	$(call BANNER,5th-neg15)
	@if [ ! -f "$(DIR_5TH_NEG15)/features.csv" ]; then \
		echo "[note] $(INI_5TH_NEG15) absent — $(DIR_5TH_NEG15)/features.csv must be present" >&2; \
		echo "       (extract step skipped; main.py will fail if light_result_file in config.ini is invalid)" >&2; \
	fi
	$(PY) main.py --configpath $(DIR_5TH_NEG15)/config.ini --logpath $(DIR_5TH_NEG15)/extract.log
	@echo "[done] features written under $(DIR_5TH_NEG15)/"

endif

# ---------- 5th-neg20 ----------
ifneq ($(wildcard $(INI_5TH_NEG20)),)

$(JSON_5TH_NEG20): $(INI_5TH_NEG20)
	$(call BANNER,extract 5th-neg20)
	$(PY) tools/extract_common.py --configpath $(INI_5TH_NEG20)

extract-5th-neg20: $(JSON_5TH_NEG20)

5th-neg20: $(INI_5TH_NEG20) $(JSON_5TH_NEG20) $(DIR_5TH_NEG20)/config.ini
	$(call BANNER,5th-neg20)
	$(PY) main.py --configpath $(DIR_5TH_NEG20)/config.ini --logpath $(DIR_5TH_NEG20)/extract.log
	@echo "[done] features written under $(DIR_5TH_NEG20)/"

else  # $(INI_5TH_NEG20) absent — features.csv must be externally provided

extract-5th-neg20:
	@echo "[error] $(INI_5TH_NEG20) not found — cannot extract; provide ini or use a pre-built JSON" >&2
	@false

5th-neg20: $(DIR_5TH_NEG20)/config.ini
	$(call BANNER,5th-neg20)
	@if [ ! -f "$(DIR_5TH_NEG20)/features.csv" ]; then \
		echo "[note] $(INI_5TH_NEG20) absent — $(DIR_5TH_NEG20)/features.csv must be present" >&2; \
		echo "       (extract step skipped; main.py will fail if light_result_file in config.ini is invalid)" >&2; \
	fi
	$(PY) main.py --configpath $(DIR_5TH_NEG20)/config.ini --logpath $(DIR_5TH_NEG20)/extract.log
	@echo "[done] features written under $(DIR_5TH_NEG20)/"

endif

# ---------- normal-neg15 ----------
ifneq ($(wildcard $(INI_NORMAL_NEG15)),)

$(JSON_NORMAL_NEG15): $(INI_NORMAL_NEG15)
	$(call BANNER,extract normal-neg15)
	$(PY) tools/extract_common.py --configpath $(INI_NORMAL_NEG15)

extract-normal-neg15: $(JSON_NORMAL_NEG15)

normal-neg15: $(INI_NORMAL_NEG15) $(JSON_NORMAL_NEG15) $(DIR_NORMAL_NEG15)/config.ini
	$(call BANNER,normal-neg15)
	$(PY) main.py --configpath $(DIR_NORMAL_NEG15)/config.ini --logpath $(DIR_NORMAL_NEG15)/extract.log
	@echo "[done] features written under $(DIR_NORMAL_NEG15)/"

else  # $(INI_NORMAL_NEG15) absent — features.csv must be externally provided

extract-normal-neg15:
	@echo "[error] $(INI_NORMAL_NEG15) not found — cannot extract; provide ini or use a pre-built JSON" >&2
	@false

normal-neg15: $(DIR_NORMAL_NEG15)/config.ini
	$(call BANNER,normal-neg15)
	@if [ ! -f "$(DIR_NORMAL_NEG15)/features.csv" ]; then \
		echo "[note] $(INI_NORMAL_NEG15) absent — $(DIR_NORMAL_NEG15)/features.csv must be present" >&2; \
		echo "       (extract step skipped; main.py will fail if light_result_file in config.ini is invalid)" >&2; \
	fi
	$(PY) main.py --configpath $(DIR_NORMAL_NEG15)/config.ini --logpath $(DIR_NORMAL_NEG15)/extract.log
	@echo "[done] features written under $(DIR_NORMAL_NEG15)/"

endif

# ---------- normal-neg20 ----------
ifneq ($(wildcard $(INI_NORMAL_NEG20)),)

$(JSON_NORMAL_NEG20): $(INI_NORMAL_NEG20)
	$(call BANNER,extract normal-neg20)
	$(PY) tools/extract_common.py --configpath $(INI_NORMAL_NEG20)

extract-normal-neg20: $(JSON_NORMAL_NEG20)

normal-neg20: $(INI_NORMAL_NEG20) $(JSON_NORMAL_NEG20) $(DIR_NORMAL_NEG20)/config.ini
	$(call BANNER,normal-neg20)
	$(PY) main.py --configpath $(DIR_NORMAL_NEG20)/config.ini --logpath $(DIR_NORMAL_NEG20)/extract.log
	@echo "[done] features written under $(DIR_NORMAL_NEG20)/"

else  # $(INI_NORMAL_NEG20) absent — features.csv must be externally provided

extract-normal-neg20:
	@echo "[error] $(INI_NORMAL_NEG20) not found — cannot extract; provide ini or use a pre-built JSON" >&2
	@false

normal-neg20: $(DIR_NORMAL_NEG20)/config.ini
	$(call BANNER,normal-neg20)
	@if [ ! -f "$(DIR_NORMAL_NEG20)/features.csv" ]; then \
		echo "[note] $(INI_NORMAL_NEG20) absent — $(DIR_NORMAL_NEG20)/features.csv must be present" >&2; \
		echo "       (extract step skipped; main.py will fail if light_result_file in config.ini is invalid)" >&2; \
	fi
	$(PY) main.py --configpath $(DIR_NORMAL_NEG20)/config.ini --logpath $(DIR_NORMAL_NEG20)/extract.log
	@echo "[done] features written under $(DIR_NORMAL_NEG20)/"

endif

# ---------- 组合 ----------

all: 2th 5th normal

# ---------- 清理 ----------
#
# clean-2th / clean-5th / clean-normal: 仅删除对应 baseline 目录下的
# features.csv（含 .PARTIAL_INCOMPLETE）和 *.log，强制下次重跑特征提取。
# 保留 config.ini、eval/（手工维护的指标）和 workspace/（缓存的中间数据，
# I-MK2 之后 per-baseline workspace 就在这里）。

clean-2th:
	@if [ -d $(DIR_2TH) ]; then \
		rm -f $(DIR_2TH)/features.csv $(DIR_2TH)/features.csv.PARTIAL_INCOMPLETE $(DIR_2TH)/*.log; \
		echo "[cleaned] $(DIR_2TH)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_2TH)/ does not exist"; \
	fi

clean-5th:
	@if [ -d $(DIR_5TH) ]; then \
		rm -f $(DIR_5TH)/features.csv $(DIR_5TH)/features.csv.PARTIAL_INCOMPLETE $(DIR_5TH)/*.log; \
		echo "[cleaned] $(DIR_5TH)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_5TH)/ does not exist"; \
	fi

clean-normal:
	@if [ -d $(DIR_NORMAL) ]; then \
		rm -f $(DIR_NORMAL)/features.csv $(DIR_NORMAL)/features.csv.PARTIAL_INCOMPLETE $(DIR_NORMAL)/*.log; \
		echo "[cleaned] $(DIR_NORMAL)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_NORMAL)/ does not exist"; \
	fi

clean-all: clean-2th clean-5th clean-normal

# --------------------------- heavy-out-of-range 一键过滤 ---------------------------
# 对已有 features.csv 一次性删除 heavy_out_of_range==1（正负例都删）。
# 新提取已在 main.py 内自动过滤；本目标用于过滤"过滤功能上线前"产出的旧文件。
filter-dry:
	$(call BANNER,filter dry-run)
	$(PY) -m workflows.feature_postfilter $(wildcard $(FILTER_GLOB))

filter:
	$(call BANNER,filter in-place)
	$(PY) -m workflows.feature_postfilter --in-place $(wildcard $(FILTER_GLOB))
	@echo "[done] 已就地过滤 $(FILTER_GLOB)（原文件备份为 *.prefilter.bak）"

# Neg-FDR variant clean targets (same conservative pattern as clean-2th/5th/normal)
clean-2th-neg05:
	@if [ -d $(DIR_2TH_NEG05) ]; then \
		rm -f $(DIR_2TH_NEG05)/features.csv $(DIR_2TH_NEG05)/features.csv.PARTIAL_INCOMPLETE $(DIR_2TH_NEG05)/*.log; \
		echo "[cleaned] $(DIR_2TH_NEG05)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_2TH_NEG05)/ does not exist"; \
	fi

clean-2th-neg10:
	@if [ -d $(DIR_2TH_NEG10) ]; then \
		rm -f $(DIR_2TH_NEG10)/features.csv $(DIR_2TH_NEG10)/features.csv.PARTIAL_INCOMPLETE $(DIR_2TH_NEG10)/*.log; \
		echo "[cleaned] $(DIR_2TH_NEG10)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_2TH_NEG10)/ does not exist"; \
	fi

clean-5th-neg05:
	@if [ -d $(DIR_5TH_NEG05) ]; then \
		rm -f $(DIR_5TH_NEG05)/features.csv $(DIR_5TH_NEG05)/features.csv.PARTIAL_INCOMPLETE $(DIR_5TH_NEG05)/*.log; \
		echo "[cleaned] $(DIR_5TH_NEG05)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_5TH_NEG05)/ does not exist"; \
	fi

clean-5th-neg10:
	@if [ -d $(DIR_5TH_NEG10) ]; then \
		rm -f $(DIR_5TH_NEG10)/features.csv $(DIR_5TH_NEG10)/features.csv.PARTIAL_INCOMPLETE $(DIR_5TH_NEG10)/*.log; \
		echo "[cleaned] $(DIR_5TH_NEG10)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_5TH_NEG10)/ does not exist"; \
	fi

clean-normal-neg05:
	@if [ -d $(DIR_NORMAL_NEG05) ]; then \
		rm -f $(DIR_NORMAL_NEG05)/features.csv $(DIR_NORMAL_NEG05)/features.csv.PARTIAL_INCOMPLETE $(DIR_NORMAL_NEG05)/*.log; \
		echo "[cleaned] $(DIR_NORMAL_NEG05)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_NORMAL_NEG05)/ does not exist"; \
	fi

clean-normal-neg10:
	@if [ -d $(DIR_NORMAL_NEG10) ]; then \
		rm -f $(DIR_NORMAL_NEG10)/features.csv $(DIR_NORMAL_NEG10)/features.csv.PARTIAL_INCOMPLETE $(DIR_NORMAL_NEG10)/*.log; \
		echo "[cleaned] $(DIR_NORMAL_NEG10)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_NORMAL_NEG10)/ does not exist"; \
	fi

clean-2th-neg15:
	@if [ -d $(DIR_2TH_NEG15) ]; then \
		rm -f $(DIR_2TH_NEG15)/features.csv $(DIR_2TH_NEG15)/features.csv.PARTIAL_INCOMPLETE $(DIR_2TH_NEG15)/*.log; \
		echo "[cleaned] $(DIR_2TH_NEG15)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_2TH_NEG15)/ does not exist"; \
	fi

clean-2th-neg20:
	@if [ -d $(DIR_2TH_NEG20) ]; then \
		rm -f $(DIR_2TH_NEG20)/features.csv $(DIR_2TH_NEG20)/features.csv.PARTIAL_INCOMPLETE $(DIR_2TH_NEG20)/*.log; \
		echo "[cleaned] $(DIR_2TH_NEG20)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_2TH_NEG20)/ does not exist"; \
	fi

clean-5th-neg15:
	@if [ -d $(DIR_5TH_NEG15) ]; then \
		rm -f $(DIR_5TH_NEG15)/features.csv $(DIR_5TH_NEG15)/features.csv.PARTIAL_INCOMPLETE $(DIR_5TH_NEG15)/*.log; \
		echo "[cleaned] $(DIR_5TH_NEG15)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_5TH_NEG15)/ does not exist"; \
	fi

clean-5th-neg20:
	@if [ -d $(DIR_5TH_NEG20) ]; then \
		rm -f $(DIR_5TH_NEG20)/features.csv $(DIR_5TH_NEG20)/features.csv.PARTIAL_INCOMPLETE $(DIR_5TH_NEG20)/*.log; \
		echo "[cleaned] $(DIR_5TH_NEG20)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_5TH_NEG20)/ does not exist"; \
	fi

clean-normal-neg15:
	@if [ -d $(DIR_NORMAL_NEG15) ]; then \
		rm -f $(DIR_NORMAL_NEG15)/features.csv $(DIR_NORMAL_NEG15)/features.csv.PARTIAL_INCOMPLETE $(DIR_NORMAL_NEG15)/*.log; \
		echo "[cleaned] $(DIR_NORMAL_NEG15)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_NORMAL_NEG15)/ does not exist"; \
	fi

clean-normal-neg20:
	@if [ -d $(DIR_NORMAL_NEG20) ]; then \
		rm -f $(DIR_NORMAL_NEG20)/features.csv $(DIR_NORMAL_NEG20)/features.csv.PARTIAL_INCOMPLETE $(DIR_NORMAL_NEG20)/*.log; \
		echo "[cleaned] $(DIR_NORMAL_NEG20)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_NORMAL_NEG20)/ does not exist"; \
	fi

# Group targets
all-clean: all
all-neg05: 2th-neg05 5th-neg05 normal-neg05
all-neg10: 2th-neg10 5th-neg10 normal-neg10
all-neg15: 2th-neg15 5th-neg15 normal-neg15
all-neg20: 2th-neg20 5th-neg20 normal-neg20

# ---------- spec_trainer 训练 target ----------
#
# train-exp1 / train-exp2: 调用 tools/spec_trainer/src/main.py 训练对应实验
# 自动依赖 features.csv 存在；缺失时级联触发 make 2th / 5th。
# 输出落到 runs/spec_trainer/{models,results,figures}/
#
# exp1: 2da only
# exp2: combined (2da + 5da)

.PHONY: train-exp1 train-exp2 clean-train
.PHONY: train-legacy-all train-clean-all train-neg05-all train-neg10-all train-neg15-all train-neg20-all train-all
.PHONY: train-cv-2da

# features.csv 规则：缺失时自动跑对应特征提取；并声明上游依赖
# （baseline config.ini + extract ini + dataset JSON），任一比 features.csv 新
# 时强制重跑——否则改了 [speclib] 等配置后 train-* 会静默用旧特征训练。
runs/baseline_2da_clean/features.csv: $(DIR_2TH)/config.ini $(INI_2TH) $(JSON_2TH)
	$(MAKE) 2th

runs/baseline_5da_clean/features.csv: $(DIR_5TH)/config.ini $(INI_5TH) $(JSON_5TH)
	$(MAKE) 5th

runs/baseline_normal_clean/features.csv: $(DIR_NORMAL)/config.ini $(INI_NORMAL) $(JSON_NORMAL)
	$(MAKE) normal

# neg-FDR variants (review fix: ensure train-{neg05,neg10}-all can auto-trigger
# the extraction chain if features.csv is missing OR stale, matching *_clean).
runs/baseline_2da_neg05/features.csv: $(DIR_2TH_NEG05)/config.ini $(INI_2TH_NEG05) $(JSON_2TH_NEG05)
	$(MAKE) 2th-neg05

runs/baseline_5da_neg05/features.csv: $(DIR_5TH_NEG05)/config.ini $(INI_5TH_NEG05) $(JSON_5TH_NEG05)
	$(MAKE) 5th-neg05

runs/baseline_normal_neg05/features.csv: $(DIR_NORMAL_NEG05)/config.ini $(INI_NORMAL_NEG05) $(JSON_NORMAL_NEG05)
	$(MAKE) normal-neg05

runs/baseline_2da_neg10/features.csv: $(DIR_2TH_NEG10)/config.ini $(INI_2TH_NEG10) $(JSON_2TH_NEG10)
	$(MAKE) 2th-neg10

runs/baseline_5da_neg10/features.csv: $(DIR_5TH_NEG10)/config.ini $(INI_5TH_NEG10) $(JSON_5TH_NEG10)
	$(MAKE) 5th-neg10

runs/baseline_normal_neg10/features.csv: $(DIR_NORMAL_NEG10)/config.ini $(INI_NORMAL_NEG10) $(JSON_NORMAL_NEG10)
	$(MAKE) normal-neg10

runs/baseline_2da_neg15/features.csv: $(DIR_2TH_NEG15)/config.ini $(INI_2TH_NEG15) $(JSON_2TH_NEG15)
	$(MAKE) 2th-neg15

runs/baseline_2da_neg20/features.csv: $(DIR_2TH_NEG20)/config.ini $(INI_2TH_NEG20) $(JSON_2TH_NEG20)
	$(MAKE) 2th-neg20

runs/baseline_2da_pos50/features.csv: $(DIR_2TH_POS50)/config.ini $(INI_2TH_POS50) $(JSON_2TH_POS50)
	$(MAKE) 2th-pos50

runs/baseline_5da_neg15/features.csv: $(DIR_5TH_NEG15)/config.ini $(INI_5TH_NEG15) $(JSON_5TH_NEG15)
	$(MAKE) 5th-neg15

runs/baseline_5da_neg20/features.csv: $(DIR_5TH_NEG20)/config.ini $(INI_5TH_NEG20) $(JSON_5TH_NEG20)
	$(MAKE) 5th-neg20

runs/baseline_normal_neg15/features.csv: $(DIR_NORMAL_NEG15)/config.ini $(INI_NORMAL_NEG15) $(JSON_NORMAL_NEG15)
	$(MAKE) normal-neg15

runs/baseline_normal_neg20/features.csv: $(DIR_NORMAL_NEG20)/config.ini $(INI_NORMAL_NEG20) $(JSON_NORMAL_NEG20)
	$(MAKE) normal-neg20

train-exp1: runs/baseline_2da_clean/features.csv tools/spec_trainer/config/exp1.yaml
	$(call BANNER,train-exp1)
	@mkdir -p runs/spec_trainer/models runs/spec_trainer/results runs/spec_trainer/figures
	$(PY) tools/spec_trainer/src/main.py --config tools/spec_trainer/config/exp1.yaml --name exp1
	@echo "[done] train-exp1 finished"

# train-exp2: exp2.yaml uses combined 2da + 5da, so depend on BOTH features.csv
train-exp2: runs/baseline_2da_clean/features.csv runs/baseline_5da_clean/features.csv tools/spec_trainer/config/exp2.yaml
	$(call BANNER,train-exp2)
	@mkdir -p runs/spec_trainer/models runs/spec_trainer/results runs/spec_trainer/figures
	$(PY) tools/spec_trainer/src/main.py --config tools/spec_trainer/config/exp2.yaml --name exp2
	@echo "[done] train-exp2 finished"

# Legacy train-all (now exposed as train-legacy-all to free 'train-all'
# for the 18-experiment matrix; see docs/specs/2026-06-03-systematic-
# training-matrix-design.md).
train-legacy-all: train-exp1 train-exp2

# Systematic training matrix (18 experiments)
# 3 FDR conditions × 2 schemes × 3 datasets.
# Each yaml is invoked via 'python tools/spec_trainer/src/main.py
# --config <yaml> --name <basename>'.

SPEC_CFG := tools/spec_trainer/config

# Features.csv lists per FDR — used as prerequisites so 'make train-*-all'
# auto-triggers extraction when CSVs are missing (review fix).
CLEAN_FEATURES := runs/baseline_2da_clean/features.csv \
                  runs/baseline_5da_clean/features.csv \
                  runs/baseline_normal_clean/features.csv

NEG05_FEATURES := runs/baseline_2da_neg05/features.csv \
                  runs/baseline_5da_neg05/features.csv \
                  runs/baseline_normal_neg05/features.csv

NEG10_FEATURES := runs/baseline_2da_neg10/features.csv \
                  runs/baseline_5da_neg10/features.csv \
                  runs/baseline_normal_neg10/features.csv

NEG15_FEATURES := runs/baseline_2da_neg15/features.csv \
                  runs/baseline_5da_neg15/features.csv \
                  runs/baseline_normal_neg15/features.csv

NEG20_FEATURES := runs/baseline_2da_neg20/features.csv \
                  runs/baseline_5da_neg20/features.csv \
                  runs/baseline_normal_neg20/features.csv

# CV-only prerequisites follow FEATURE_ROOT. Legacy single-holdout targets
# above continue to use repository-local runs/ paths.
CV_CLEAN_FEATURES = $(FEATURE_ROOT)/baseline_2da_clean/features.csv \
                    $(FEATURE_ROOT)/baseline_5da_clean/features.csv \
                    $(FEATURE_ROOT)/baseline_normal_clean/features.csv
CV_NEG05_FEATURES = $(FEATURE_ROOT)/baseline_2da_neg05/features.csv \
                    $(FEATURE_ROOT)/baseline_5da_neg05/features.csv \
                    $(FEATURE_ROOT)/baseline_normal_neg05/features.csv
CV_NEG10_FEATURES = $(FEATURE_ROOT)/baseline_2da_neg10/features.csv \
                    $(FEATURE_ROOT)/baseline_5da_neg10/features.csv \
                    $(FEATURE_ROOT)/baseline_normal_neg10/features.csv
CV_NEG15_FEATURES = $(FEATURE_ROOT)/baseline_2da_neg15/features.csv \
                    $(FEATURE_ROOT)/baseline_5da_neg15/features.csv \
                    $(FEATURE_ROOT)/baseline_normal_neg15/features.csv
CV_NEG20_FEATURES = $(FEATURE_ROOT)/baseline_2da_neg20/features.csv \
                    $(FEATURE_ROOT)/baseline_5da_neg20/features.csv \
                    $(FEATURE_ROOT)/baseline_normal_neg20/features.csv
FIXED_NEGPOOL_2DA_FEATURES = $(FEATURE_ROOT)/baseline_2da_neg05/features.csv \
                            $(FEATURE_ROOT)/baseline_2da_neg10/features.csv \
                            $(FEATURE_ROOT)/baseline_2da_neg20/features.csv
FIXED_NEGPOOL_5DA_FEATURES = $(FEATURE_ROOT)/baseline_5da_neg05/features.csv \
                            $(FEATURE_ROOT)/baseline_5da_neg10/features.csv \
                            $(FEATURE_ROOT)/baseline_5da_neg20/features.csv
FIXED_NEGPOOL_NORMAL_FEATURES = $(FEATURE_ROOT)/baseline_normal_neg05/features.csv \
                               $(FEATURE_ROOT)/baseline_normal_neg10/features.csv \
                               $(FEATURE_ROOT)/baseline_normal_neg20/features.csv
FIXED_NEGPOOL_COMBINED_FEATURES = $(FIXED_NEGPOOL_2DA_FEATURES) \
                                 $(FIXED_NEGPOOL_5DA_FEATURES) \
                                 $(FIXED_NEGPOOL_NORMAL_FEATURES)

CLEAN_YAMLS := $(SPEC_CFG)/in_2da_clean.yaml \
               $(SPEC_CFG)/in_5da_clean.yaml \
               $(SPEC_CFG)/in_normal_clean.yaml \
               $(SPEC_CFG)/cross_test_2da_clean.yaml \
               $(SPEC_CFG)/cross_test_5da_clean.yaml \
               $(SPEC_CFG)/cross_test_normal_clean.yaml

NEG05_YAMLS := $(SPEC_CFG)/in_2da_neg05.yaml \
               $(SPEC_CFG)/in_5da_neg05.yaml \
               $(SPEC_CFG)/in_normal_neg05.yaml \
               $(SPEC_CFG)/cross_test_2da_neg05.yaml \
               $(SPEC_CFG)/cross_test_5da_neg05.yaml \
               $(SPEC_CFG)/cross_test_normal_neg05.yaml

NEG10_YAMLS := $(SPEC_CFG)/in_2da_neg10.yaml \
               $(SPEC_CFG)/in_5da_neg10.yaml \
               $(SPEC_CFG)/in_normal_neg10.yaml \
               $(SPEC_CFG)/cross_test_2da_neg10.yaml \
               $(SPEC_CFG)/cross_test_5da_neg10.yaml \
               $(SPEC_CFG)/cross_test_normal_neg10.yaml

NEG15_YAMLS := $(SPEC_CFG)/in_2da_neg15.yaml \
               $(SPEC_CFG)/in_5da_neg15.yaml \
               $(SPEC_CFG)/in_normal_neg15.yaml \
               $(SPEC_CFG)/cross_test_2da_neg15.yaml \
               $(SPEC_CFG)/cross_test_5da_neg15.yaml \
               $(SPEC_CFG)/cross_test_normal_neg15.yaml

NEG20_YAMLS := $(SPEC_CFG)/in_2da_neg20.yaml \
               $(SPEC_CFG)/in_5da_neg20.yaml \
               $(SPEC_CFG)/in_normal_neg20.yaml \
               $(SPEC_CFG)/cross_test_2da_neg20.yaml \
               $(SPEC_CFG)/cross_test_5da_neg20.yaml \
               $(SPEC_CFG)/cross_test_normal_neg20.yaml

train-clean-all: $(CLEAN_FEATURES)
	@mkdir -p runs/spec_trainer/models runs/spec_trainer/results runs/spec_trainer/figures
	@for yaml in $(CLEAN_YAMLS); do \
		name=$$(basename $$yaml .yaml); \
		echo "==================== train $$name ===================="; \
		$(PY) tools/spec_trainer/src/main.py --config $$yaml --name $$name || exit 1; \
	done
	@echo "[done] train-clean-all finished (6 experiments)"

# 5 折分组 CV + 折间 ensemble + 标签审计（生产 LightGBM；见
# docs/superpowers/specs/2026-06-28-cv-ensemble-label-audit-design.md）
train-cv-2da: $(FEATURE_ROOT)/baseline_2da_clean/features.csv
	@mkdir -p "$(CV_OUTPUT_ROOT)"
	$(PY) tools/spec_trainer/gen_cv_configs.py \
	    --feature-root "$(FEATURE_ROOT)" \
	    --output-root "$(CV_OUTPUT_ROOT)" \
	    --config-dir "$(CV_CONFIG_DIR)" \
	    --feature-arm "$(CV_FEATURE_ARM)"
	$(PY) tools/spec_trainer/src/cv_train.py \
	    --config "$(CV_CONFIG_DIR)/cv_in_2da_clean.yaml" \
	    --name cv_in_2da_clean \
	    --logpath "$(CV_OUTPUT_ROOT)/logs/cv_in_2da_clean.log" $(CV_OVERWRITE_FLAG)
	@echo "[done] CV → $(CV_OUTPUT_ROOT)/results/cv_in_2da_clean.cv.json (+ .suspects.csv)"

# ---------- CV 全矩阵(in-sample + cross_test ensemble)----------
.PHONY: train-cv-clean-all train-cv-neg05-all train-cv-neg10-all
.PHONY: train-cv-neg15-all train-cv-neg20-all train-cv-all train-cv-core-all
.PHONY: train-ablation-neg20-2da train-ablation-neg20
.PHONY: train-fixed-test-negpool-2da train-fixed-test-negpool-5da
.PHONY: train-fixed-test-negpool-normal train-fixed-test-negpool-all
.PHONY: train-fixed-test-negpool-combined
.PHONY: train-deep-mlp-combined

CV_CLEAN_YAMLS := cv_in_2da_clean cv_in_5da_clean cv_in_normal_clean \
                  cv_cross_test_2da_clean cv_cross_test_5da_clean cv_cross_test_normal_clean
CV_NEG05_YAMLS := $(subst _clean,_neg05,$(CV_CLEAN_YAMLS))
CV_NEG10_YAMLS := $(subst _clean,_neg10,$(CV_CLEAN_YAMLS))
CV_NEG15_YAMLS := $(subst _clean,_neg15,$(CV_CLEAN_YAMLS))
CV_NEG20_YAMLS := $(subst _clean,_neg20,$(CV_CLEAN_YAMLS))

# Formal paired ablation. FEATURE_ROOT points at the directory containing
# baseline_{2da,5da,normal}_neg20/features.csv. Generated configs and outputs
# stay under the repository so the external feature snapshot is never copied
# or modified.
ABLATION_OUTPUT_ROOT ?= runs/spec_trainer/ablation/neg20
ABLATION_CONFIG_DIR ?= $(ABLATION_OUTPUT_ROOT)/configs
ABLATION_ARMS := context_only ms1_only ms2_observed_only ms2_all \
		         ms1_ms2_no_prediction evidence_all evidence_core full

train-ablation-neg20-2da:
	@mkdir -p "$(ABLATION_OUTPUT_ROOT)"
	$(PY) tools/spec_trainer/gen_ablation_configs.py \
	    --feature-root "$(FEATURE_ROOT)" \
	    --output-root "$(ABLATION_OUTPUT_ROOT)" \
	    --config-dir "$(ABLATION_CONFIG_DIR)" \
	    --fdr neg20 --datasets 2da
	@for arm in $(ABLATION_ARMS); do \
		name="ablation_2da_neg20_$$arm"; \
		echo "==================== CV $$name ===================="; \
		$(PY) tools/spec_trainer/src/cv_train.py \
		    --config "$(ABLATION_CONFIG_DIR)/$$name.yaml" \
		    --name "$$name" \
		    --logpath "$(ABLATION_OUTPUT_ROOT)/logs/$$name.log" $(CV_OVERWRITE_FLAG) || exit 1; \
	done
	@echo "[done] train-ablation-neg20-2da (8 paired CV experiments)"

train-ablation-neg20:
	@mkdir -p "$(ABLATION_OUTPUT_ROOT)"
	$(PY) tools/spec_trainer/gen_ablation_configs.py \
	    --feature-root "$(FEATURE_ROOT)" \
	    --output-root "$(ABLATION_OUTPUT_ROOT)" \
	    --config-dir "$(ABLATION_CONFIG_DIR)" \
	    --fdr neg20 --datasets 2da 5da normal
	@for dataset in 2da 5da normal; do \
		for arm in $(ABLATION_ARMS); do \
			name="ablation_$${dataset}_neg20_$$arm"; \
			echo "==================== CV $$name ===================="; \
			$(PY) tools/spec_trainer/src/cv_train.py \
			    --config "$(ABLATION_CONFIG_DIR)/$$name.yaml" \
			    --name "$$name" \
			    --logpath "$(ABLATION_OUTPUT_ROOT)/logs/$$name.log" $(CV_OVERWRITE_FLAG) || exit 1; \
		done; \
	done
	@echo "[done] train-ablation-neg20 (24 paired CV experiments)"

train-cv-clean-all: $(CV_CLEAN_FEATURES)
	@mkdir -p "$(CV_OUTPUT_ROOT)"
	$(PY) tools/spec_trainer/gen_cv_configs.py \
	    --feature-root "$(FEATURE_ROOT)" \
	    --output-root "$(CV_OUTPUT_ROOT)" \
	    --config-dir "$(CV_CONFIG_DIR)" \
	    --feature-arm "$(CV_FEATURE_ARM)"
	@for y in $(CV_CLEAN_YAMLS); do \
		echo "==================== CV $$y ===================="; \
		$(PY) tools/spec_trainer/src/cv_train.py \
		    --config "$(CV_CONFIG_DIR)/$$y.yaml" --name $$y \
		    --logpath "$(CV_OUTPUT_ROOT)/logs/$$y.log" $(CV_OVERWRITE_FLAG) || exit 1; \
	done
	@echo "[done] train-cv-clean-all (6 CV experiments)"

train-cv-neg05-all: $(CV_NEG05_FEATURES)
	@mkdir -p "$(CV_OUTPUT_ROOT)"
	$(PY) tools/spec_trainer/gen_cv_configs.py \
	    --feature-root "$(FEATURE_ROOT)" \
	    --output-root "$(CV_OUTPUT_ROOT)" \
	    --config-dir "$(CV_CONFIG_DIR)" \
	    --feature-arm "$(CV_FEATURE_ARM)"
	@for y in $(CV_NEG05_YAMLS); do \
		echo "==================== CV $$y ===================="; \
		$(PY) tools/spec_trainer/src/cv_train.py \
		    --config "$(CV_CONFIG_DIR)/$$y.yaml" --name $$y \
		    --logpath "$(CV_OUTPUT_ROOT)/logs/$$y.log" $(CV_OVERWRITE_FLAG) || exit 1; \
	done
	@echo "[done] train-cv-neg05-all (6 CV experiments)"

train-cv-neg10-all: $(CV_NEG10_FEATURES)
	@mkdir -p "$(CV_OUTPUT_ROOT)"
	$(PY) tools/spec_trainer/gen_cv_configs.py \
	    --feature-root "$(FEATURE_ROOT)" \
	    --output-root "$(CV_OUTPUT_ROOT)" \
	    --config-dir "$(CV_CONFIG_DIR)" \
	    --feature-arm "$(CV_FEATURE_ARM)"
	@for y in $(CV_NEG10_YAMLS); do \
		echo "==================== CV $$y ===================="; \
		$(PY) tools/spec_trainer/src/cv_train.py \
		    --config "$(CV_CONFIG_DIR)/$$y.yaml" --name $$y \
		    --logpath "$(CV_OUTPUT_ROOT)/logs/$$y.log" $(CV_OVERWRITE_FLAG) || exit 1; \
	done
	@echo "[done] train-cv-neg10-all (6 CV experiments)"

train-cv-neg15-all: $(CV_NEG15_FEATURES)
	@mkdir -p "$(CV_OUTPUT_ROOT)"
	$(PY) tools/spec_trainer/gen_cv_configs.py \
	    --feature-root "$(FEATURE_ROOT)" \
	    --output-root "$(CV_OUTPUT_ROOT)" \
	    --config-dir "$(CV_CONFIG_DIR)" \
	    --feature-arm "$(CV_FEATURE_ARM)"
	@for y in $(CV_NEG15_YAMLS); do \
		echo "==================== CV $$y ===================="; \
		$(PY) tools/spec_trainer/src/cv_train.py \
		    --config "$(CV_CONFIG_DIR)/$$y.yaml" --name $$y \
		    --logpath "$(CV_OUTPUT_ROOT)/logs/$$y.log" $(CV_OVERWRITE_FLAG) || exit 1; \
	done
	@echo "[done] train-cv-neg15-all (6 CV experiments)"

train-cv-neg20-all: $(CV_NEG20_FEATURES)
	@mkdir -p "$(CV_OUTPUT_ROOT)"
	$(PY) tools/spec_trainer/gen_cv_configs.py \
	    --feature-root "$(FEATURE_ROOT)" \
	    --output-root "$(CV_OUTPUT_ROOT)" \
	    --config-dir "$(CV_CONFIG_DIR)" \
	    --feature-arm "$(CV_FEATURE_ARM)"
	@for y in $(CV_NEG20_YAMLS); do \
		echo "==================== CV $$y ===================="; \
		$(PY) tools/spec_trainer/src/cv_train.py \
		    --config "$(CV_CONFIG_DIR)/$$y.yaml" --name $$y \
		    --logpath "$(CV_OUTPUT_ROOT)/logs/$$y.log" $(CV_OVERWRITE_FLAG) || exit 1; \
	done
	@echo "[done] train-cv-neg20-all (6 CV experiments)"

train-cv-all:
	$(MAKE) train-cv-clean-all
	$(MAKE) train-cv-neg05-all
	$(MAKE) train-cv-neg10-all
	$(MAKE) train-cv-neg15-all
	$(MAKE) train-cv-neg20-all
	@echo "[done] train-cv-all finished (30 CV experiments)"

train-cv-core-all:
	$(MAKE) train-cv-all \
	    CV_FEATURE_ARM=evidence_core \
	    CV_OUTPUT_ROOT="$(CV_OUTPUT_ROOT)/evidence_core" \
	    CV_CONFIG_DIR="$(CV_OUTPUT_ROOT)/evidence_core/configs"

# Controlled nested-negative experiment. All three models use the neg20
# feature table, the same correct rows, one sequence-held-out E20 test, and
# the same predeclared outer/inner group assignments.
train-fixed-test-negpool-2da: $(FIXED_NEGPOOL_2DA_FEATURES)
	@mkdir -p "$(FIXED_NEGPOOL_OUTPUT_ROOT)/configs"
	$(PY) tools/spec_trainer/gen_cv_configs.py \
	    --feature-root "$(FEATURE_ROOT)" \
	    --output-root "$(FIXED_NEGPOOL_OUTPUT_ROOT)/reference-cv" \
	    --config-dir "$(FIXED_NEGPOOL_OUTPUT_ROOT)/configs" \
	    --feature-arm "$(FIXED_NEGPOOL_FEATURE_ARM)"
	$(PY) tools/spec_trainer/src/fixed_negpool.py \
	    --config "$(FIXED_NEGPOOL_OUTPUT_ROOT)/configs/cv_in_2da_neg20.yaml" \
	    --feature-root "$(FEATURE_ROOT)" --dataset 2da \
	    --output-root "$(FIXED_NEGPOOL_OUTPUT_ROOT)/2da" \
	    --test-fraction "$(FIXED_NEGPOOL_TEST_FRACTION)" \
	    --bootstrap-reps "$(FIXED_NEGPOOL_BOOTSTRAPS)" $(CV_OVERWRITE_FLAG)

train-fixed-test-negpool-5da: $(FIXED_NEGPOOL_5DA_FEATURES)
	@mkdir -p "$(FIXED_NEGPOOL_OUTPUT_ROOT)/configs"
	$(PY) tools/spec_trainer/gen_cv_configs.py \
	    --feature-root "$(FEATURE_ROOT)" \
	    --output-root "$(FIXED_NEGPOOL_OUTPUT_ROOT)/reference-cv" \
	    --config-dir "$(FIXED_NEGPOOL_OUTPUT_ROOT)/configs" \
	    --feature-arm "$(FIXED_NEGPOOL_FEATURE_ARM)"
	$(PY) tools/spec_trainer/src/fixed_negpool.py \
	    --config "$(FIXED_NEGPOOL_OUTPUT_ROOT)/configs/cv_in_5da_neg20.yaml" \
	    --feature-root "$(FEATURE_ROOT)" --dataset 5da \
	    --output-root "$(FIXED_NEGPOOL_OUTPUT_ROOT)/5da" \
	    --test-fraction "$(FIXED_NEGPOOL_TEST_FRACTION)" \
	    --bootstrap-reps "$(FIXED_NEGPOOL_BOOTSTRAPS)" $(CV_OVERWRITE_FLAG)

train-fixed-test-negpool-normal: $(FIXED_NEGPOOL_NORMAL_FEATURES)
	@mkdir -p "$(FIXED_NEGPOOL_OUTPUT_ROOT)/configs"
	$(PY) tools/spec_trainer/gen_cv_configs.py \
	    --feature-root "$(FEATURE_ROOT)" \
	    --output-root "$(FIXED_NEGPOOL_OUTPUT_ROOT)/reference-cv" \
	    --config-dir "$(FIXED_NEGPOOL_OUTPUT_ROOT)/configs" \
	    --feature-arm "$(FIXED_NEGPOOL_FEATURE_ARM)"
	$(PY) tools/spec_trainer/src/fixed_negpool.py \
	    --config "$(FIXED_NEGPOOL_OUTPUT_ROOT)/configs/cv_in_normal_neg20.yaml" \
	    --feature-root "$(FEATURE_ROOT)" --dataset normal \
	    --output-root "$(FIXED_NEGPOOL_OUTPUT_ROOT)/normal" \
	    --test-fraction "$(FIXED_NEGPOOL_TEST_FRACTION)" \
	    --bootstrap-reps "$(FIXED_NEGPOOL_BOOTSTRAPS)" $(CV_OVERWRITE_FLAG)

train-fixed-test-negpool-all:
	$(MAKE) train-fixed-test-negpool-2da
	$(MAKE) train-fixed-test-negpool-5da
	$(MAKE) train-fixed-test-negpool-normal
	@echo "[done] fixed E20 test comparison complete for 2da/5da/normal"

train-fixed-test-negpool-combined: $(FIXED_NEGPOOL_COMBINED_FEATURES)
	@mkdir -p "$(FIXED_NEGPOOL_OUTPUT_ROOT)/configs"
	$(PY) tools/spec_trainer/gen_cv_configs.py \
	    --feature-root "$(FEATURE_ROOT)" \
	    --output-root "$(FIXED_NEGPOOL_OUTPUT_ROOT)/reference-cv" \
	    --config-dir "$(FIXED_NEGPOOL_OUTPUT_ROOT)/configs" \
	    --feature-arm "$(FIXED_NEGPOOL_FEATURE_ARM)"
	$(PY) tools/spec_trainer/src/fixed_negpool.py \
	    --config "$(FIXED_NEGPOOL_OUTPUT_ROOT)/configs/cv_in_2da_neg20.yaml" \
	    --feature-root "$(FEATURE_ROOT)" --dataset combined \
	    --output-root "$(FIXED_NEGPOOL_OUTPUT_ROOT)/combined" \
	    --test-fraction "$(FIXED_NEGPOOL_TEST_FRACTION)" \
	    --bootstrap-reps "$(FIXED_NEGPOOL_BOOTSTRAPS)" $(CV_OVERWRITE_FLAG)

# Phase 1 deep-learning baseline. Reuses the exact feature arm, cohort,
# sequence-grouped fixed E20 test and reusable folds of fixed-negpool; only the
# LightGBM implementation is replaced by a fold-local-preprocessed PyTorch MLP.
train-deep-mlp-combined: $(FIXED_NEGPOOL_COMBINED_FEATURES) \
	$(DEEP_PROTOCOL_ROOT)/summary.json
	@mkdir -p "$(DEEP_OUTPUT_ROOT)/configs"
	$(PY) tools/spec_trainer/gen_cv_configs.py \
	    --feature-root "$(FEATURE_ROOT)" \
	    --output-root "$(DEEP_OUTPUT_ROOT)/reference-cv" \
	    --config-dir "$(DEEP_OUTPUT_ROOT)/configs" \
	    --feature-arm "$(FIXED_NEGPOOL_FEATURE_ARM)"
	$(PY) -m tools.deep_trainer.experiment \
	    --config "$(DEEP_CONFIG)" \
	    --split-config "$(DEEP_OUTPUT_ROOT)/configs/cv_in_2da_neg20.yaml" \
	    --feature-root "$(FEATURE_ROOT)" \
	    --protocol-root "$(DEEP_PROTOCOL_ROOT)" \
	    --dataset combined \
	    --output-root "$(DEEP_OUTPUT_ROOT)/tabular-mlp/combined" \
	    $(CV_OVERWRITE_FLAG)

# Phase 2 P0/P1: build a balanced 1200-row raw-XIC integrity pilot. It reuses
# the frozen combined membership/folds and refuses to publish unless every PSM
# identity is unique and selected legacy features can be reconstructed from
# the stored tensors within tolerance.
build-deep-xic-pilot: $(FIXED_NEGPOOL_COMBINED_FEATURES) \
	$(DEEP_PROTOCOL_ROOT)/summary.json
	@mkdir -p "$(DEEP_OUTPUT_ROOT)/configs"
	$(PY) tools/spec_trainer/gen_cv_configs.py \
	    --feature-root "$(FEATURE_ROOT)" \
	    --output-root "$(DEEP_OUTPUT_ROOT)/reference-cv" \
	    --config-dir "$(DEEP_OUTPUT_ROOT)/configs" \
	    --feature-arm "$(FIXED_NEGPOOL_FEATURE_ARM)"
	$(PY) -m tools.deep_trainer.phase2.builder \
	    --config "$(PHASE2_BUILD_CONFIG)" \
	    --split-config "$(DEEP_OUTPUT_ROOT)/configs/cv_in_2da_neg20.yaml" \
	    --feature-root "$(FEATURE_ROOT)" \
	    --protocol-root "$(DEEP_PROTOCOL_ROOT)" \
	    --output-root "$(PHASE2_XIC_OUTPUT_ROOT)" \
	    --cache-root "$(PHASE2_CACHE_ROOT)" \
	    $(CV_OVERWRITE_FLAG)

# Stream the complete frozen cohort. Fragment panels load each selected MS2
# scan once, and committed shards resume after interruption.
build-deep-xic-full: $(FIXED_NEGPOOL_COMBINED_FEATURES) \
	$(DEEP_PROTOCOL_ROOT)/summary.json
	@mkdir -p "$(DEEP_OUTPUT_ROOT)/configs"
	$(PY) tools/spec_trainer/gen_cv_configs.py \
	    --feature-root "$(FEATURE_ROOT)" \
	    --output-root "$(DEEP_OUTPUT_ROOT)/reference-cv" \
	    --config-dir "$(DEEP_OUTPUT_ROOT)/configs" \
	    --feature-arm "$(FIXED_NEGPOOL_FEATURE_ARM)"
	$(PY) -m tools.deep_trainer.phase2.builder \
	    --config "$(PHASE2_FULL_BUILD_CONFIG)" \
	    --split-config "$(DEEP_OUTPUT_ROOT)/configs/cv_in_2da_neg20.yaml" \
	    --feature-root "$(FEATURE_ROOT)" \
	    --protocol-root "$(DEEP_PROTOCOL_ROOT)" \
	    --output-root "$(PHASE2_FULL_XIC_OUTPUT_ROOT)" \
	    --cache-root "$(PHASE2_CACHE_ROOT)" \
	    --resume \
	    $(CV_OVERWRITE_FLAG)

# Phase 2 signal-native model. The immutable XIC dataset must contain the
# exact complete frozen cohort. Every assignment and protocol hash is checked
# again before the 3-seed x 5-fold ensemble is trained.
train-deep-xic-combined: $(FIXED_NEGPOOL_COMBINED_FEATURES) \
	$(DEEP_PROTOCOL_ROOT)/summary.json \
	$(PHASE2_FULL_XIC_OUTPUT_ROOT)/COMPLETE
	@mkdir -p "$(DEEP_OUTPUT_ROOT)/configs"
	$(PY) tools/spec_trainer/gen_cv_configs.py \
	    --feature-root "$(FEATURE_ROOT)" \
	    --output-root "$(DEEP_OUTPUT_ROOT)/reference-cv" \
	    --config-dir "$(DEEP_OUTPUT_ROOT)/configs" \
	    --feature-arm "$(FIXED_NEGPOOL_FEATURE_ARM)"
	$(PY) -m tools.deep_trainer.phase2.experiment \
	    --config "$(PHASE2_TRAIN_CONFIG)" \
	    --split-config "$(DEEP_OUTPUT_ROOT)/configs/cv_in_2da_neg20.yaml" \
	    --feature-root "$(FEATURE_ROOT)" \
	    --protocol-root "$(DEEP_PROTOCOL_ROOT)" \
	    --signal-root "$(PHASE2_FULL_XIC_OUTPUT_ROOT)" \
	    --output-root "$(PHASE2_TRAIN_OUTPUT_ROOT)" \
	    $(CV_OVERWRITE_FLAG)

# Strong signal-only arm. It reuses the exact v3 XIC dataset and frozen
# membership, but writes to a separate result root so the v2 baseline remains
# immutable and directly comparable.
smoke-deep-xic-cuda:
	CUBLAS_WORKSPACE_CONFIG="$${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" \
	$(PY) -m tools.deep_trainer.phase2.cuda_smoke \
	    --config "$(PHASE2_STRONG_TRAIN_CONFIG)"

train-deep-xic-strong-combined: $(FIXED_NEGPOOL_COMBINED_FEATURES) \
	$(DEEP_PROTOCOL_ROOT)/summary.json \
	$(PHASE2_FULL_XIC_OUTPUT_ROOT)/COMPLETE
	CUBLAS_WORKSPACE_CONFIG="$${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" \
	$(PY) -m tools.deep_trainer.phase2.cuda_smoke \
	    --config "$(PHASE2_STRONG_TRAIN_CONFIG)"
	@mkdir -p "$(DEEP_OUTPUT_ROOT)/configs"
	$(PY) tools/spec_trainer/gen_cv_configs.py \
	    --feature-root "$(FEATURE_ROOT)" \
	    --output-root "$(DEEP_OUTPUT_ROOT)/reference-cv" \
	    --config-dir "$(DEEP_OUTPUT_ROOT)/configs" \
	    --feature-arm "$(FIXED_NEGPOOL_FEATURE_ARM)"
	CUBLAS_WORKSPACE_CONFIG="$${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" \
	$(PY) -m tools.deep_trainer.phase2.experiment \
	    --config "$(PHASE2_STRONG_TRAIN_CONFIG)" \
	    --split-config "$(DEEP_OUTPUT_ROOT)/configs/cv_in_2da_neg20.yaml" \
	    --feature-root "$(FEATURE_ROOT)" \
	    --protocol-root "$(DEEP_PROTOCOL_ROOT)" \
	    --signal-root "$(PHASE2_FULL_XIC_OUTPUT_ROOT)" \
	    --output-root "$(PHASE2_STRONG_TRAIN_OUTPUT_ROOT)" \
	    $(CV_OVERWRITE_FLAG)

$(PHASE2_FULL_XIC_OUTPUT_ROOT)/COMPLETE:
	@echo "[error] missing complete full Phase 2 XIC dataset: $@" >&2
	@echo "Run make build-deep-xic-full FEATURE_ROOT=... first." >&2
	@false

# A completed LightGBM bundle is the frozen owner of membership/folds and its
# input SHA256 values. This rule deliberately does not regenerate/overwrite an
# existing protocol implicitly; use train-fixed-test-negpool-combined when a
# new feature snapshot needs a new frozen protocol.
$(DEEP_PROTOCOL_ROOT)/summary.json:
	@echo "[error] missing frozen fixed-negpool protocol: $@" >&2
	@echo "Run make train-fixed-test-negpool-combined FEATURE_ROOT=... first." >&2
	@false

train-neg05-all: $(NEG05_FEATURES)
	@mkdir -p runs/spec_trainer/models runs/spec_trainer/results runs/spec_trainer/figures
	@for yaml in $(NEG05_YAMLS); do \
		name=$$(basename $$yaml .yaml); \
		echo "==================== train $$name ===================="; \
		$(PY) tools/spec_trainer/src/main.py --config $$yaml --name $$name || exit 1; \
	done
	@echo "[done] train-neg05-all finished (6 experiments)"

train-neg10-all: $(NEG10_FEATURES)
	@mkdir -p runs/spec_trainer/models runs/spec_trainer/results runs/spec_trainer/figures
	@for yaml in $(NEG10_YAMLS); do \
		name=$$(basename $$yaml .yaml); \
		echo "==================== train $$name ===================="; \
		$(PY) tools/spec_trainer/src/main.py --config $$yaml --name $$name || exit 1; \
	done
	@echo "[done] train-neg10-all finished (6 experiments)"

train-neg15-all: $(NEG15_FEATURES)
	@mkdir -p runs/spec_trainer/models runs/spec_trainer/results runs/spec_trainer/figures
	@for yaml in $(NEG15_YAMLS); do \
		name=$$(basename $$yaml .yaml); \
		echo "==================== train $$name ===================="; \
		$(PY) tools/spec_trainer/src/main.py --config $$yaml --name $$name || exit 1; \
	done
	@echo "[done] train-neg15-all finished (6 experiments)"

train-neg20-all: $(NEG20_FEATURES)
	@mkdir -p runs/spec_trainer/models runs/spec_trainer/results runs/spec_trainer/figures
	@for yaml in $(NEG20_YAMLS); do \
		name=$$(basename $$yaml .yaml); \
		echo "==================== train $$name ===================="; \
		$(PY) tools/spec_trainer/src/main.py --config $$yaml --name $$name || exit 1; \
	done
	@echo "[done] train-neg20-all finished (6 experiments)"

# Recursive $(MAKE) invocations force strictly sequential execution even
# under 'make -j N' — phony-prereq chaining would otherwise let make
# parallelize the 3 groups, interleaving the per-experiment banners and
# breaking the documented "clean → neg05 → neg10" order (review fix).
train-all:
	$(MAKE) train-clean-all
	$(MAKE) train-neg05-all
	$(MAKE) train-neg10-all
	$(MAKE) train-neg15-all
	$(MAKE) train-neg20-all
	@echo "[done] train-all finished (30 experiments)"

clean-train:
	@if [ -d runs/spec_trainer ]; then \
		find runs/spec_trainer -mindepth 1 -delete 2>/dev/null || true; \
		echo "[cleaned] runs/spec_trainer/"; \
	else \
		echo "[skip] runs/spec_trainer/ does not exist"; \
	fi

# ---------- 兼容旧用法 ----------

run:
	@toilet -f mono12 "start" | lolcat 2>/dev/null || echo "===== start ====="
	@$(PY) main.py
	@toilet -f mono12 "end" | lolcat 2>/dev/null || echo "===== end ====="

clean:
	rm -f checkpoint.pkl
	rm -f c13_mix.csv
	rm -f *.cache
	rm -f *fake.mgf
