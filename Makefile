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

# 三个 extract_common 配置文件
INI_2TH    ?= extract_2da_pfind_diann.ini
INI_5TH    ?= extract_5da_pfind_diann.ini
INI_NORMAL ?= extract_normal_pfind_diann.ini

# 三个 baseline 目录（含 config.ini + 输出位置）
DIR_2TH    ?= runs/baseline_2da_clean
DIR_5TH    ?= runs/baseline_5da_clean
DIR_NORMAL ?= runs/baseline_normal_clean

# 从 extract_*.ini 中动态抽取 result_file 路径。
# grep 找 ^result_file= 行 -> 取 = 右侧 -> 去前后空格
# 如果 ini 不存在，结果为空，target 会在依赖检查时报缺失文件错。
JSON_2TH    := $(strip $(shell test -f $(INI_2TH)    && grep -E '^[[:space:]]*result_file[[:space:]]*=' $(INI_2TH)    | head -1 | cut -d= -f2- | tr -d ' '))
JSON_5TH    := $(strip $(shell test -f $(INI_5TH)    && grep -E '^[[:space:]]*result_file[[:space:]]*=' $(INI_5TH)    | head -1 | cut -d= -f2- | tr -d ' '))
JSON_NORMAL := $(strip $(shell test -f $(INI_NORMAL) && grep -E '^[[:space:]]*result_file[[:space:]]*=' $(INI_NORMAL) | head -1 | cut -d= -f2- | tr -d ' '))

# --------------------------- 工具 / banner ---------------------------

# 优先用 toilet | lolcat（漂亮），缺失则退回 echo
define BANNER
@command -v toilet >/dev/null 2>&1 && command -v lolcat >/dev/null 2>&1 \
&& toilet -f mono12 "$(1)" | lolcat \
|| echo "==================== $(1) ===================="
endef

# --------------------------- 主 target ---------------------------

.PHONY: help all run clean
.PHONY: 2th 5th normal
.PHONY: extract-2th extract-5th extract-normal
.PHONY: clean-2th clean-5th clean-normal clean-all

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
	@echo "  make clean-2th       删除 2da features.csv / log，强制下次重跑"
	@echo "  make clean-5th       删除 5da features.csv / log"
	@echo "  make clean-normal    删除 normal features.csv / log"
	@echo "  make clean-all       上述三者全清"
	@echo "  make clean           旧式清理（checkpoint.pkl 等）"
	@echo ""
	@echo "  make run             旧 target：跑 main.py 使用根目录 config.ini"
	@echo ""
	@echo "当前抽取的 JSON 路径："
	@echo "  2th    -> $(JSON_2TH)"
	@echo "  5th    -> $(JSON_5TH)"
	@echo "  normal -> $(JSON_NORMAL)"

# ---------- 2th ----------

# JSON 缺失时自动跑 extract_common
$(JSON_2TH): $(INI_2TH)
	$(call BANNER,extract 2th)
	$(PY) tools/extract_common.py --configpath $(INI_2TH)

extract-2th: $(JSON_2TH)

2th: $(INI_2TH) $(JSON_2TH) $(DIR_2TH)/config.ini
	$(call BANNER,2th)
	$(PY) main.py --configpath $(DIR_2TH)/config.ini --logpath $(DIR_2TH)/extract.log
	@echo "[done] features written under $(DIR_2TH)/"

# ---------- 5th ----------

$(JSON_5TH): $(INI_5TH)
	$(call BANNER,extract 5th)
	$(PY) tools/extract_common.py --configpath $(INI_5TH)

extract-5th: $(JSON_5TH)

5th: $(INI_5TH) $(JSON_5TH) $(DIR_5TH)/config.ini
	$(call BANNER,5th)
	$(PY) main.py --configpath $(DIR_5TH)/config.ini --logpath $(DIR_5TH)/extract.log
	@echo "[done] features written under $(DIR_5TH)/"

# ---------- normal ----------

$(JSON_NORMAL): $(INI_NORMAL)
	$(call BANNER,extract normal)
	$(PY) tools/extract_common.py --configpath $(INI_NORMAL)

extract-normal: $(JSON_NORMAL)

normal: $(INI_NORMAL) $(JSON_NORMAL) $(DIR_NORMAL)/config.ini
	$(call BANNER,normal)
	$(PY) main.py --configpath $(DIR_NORMAL)/config.ini --logpath $(DIR_NORMAL)/extract.log
	@echo "[done] features written under $(DIR_NORMAL)/"

# ---------- 组合 ----------

all: 2th 5th normal

# ---------- 清理 ----------
#
# clean-2th / clean-5th / clean-normal: 删除对应 baseline 目录下除根部
# config.ini 以外的所有内容（递归）。这保证未来添加新输出文件（如新 eval
# 子目录、新 metric json）也会被自动清理，不会因为漏写 rm 列表而残留旧产物。
#
# 实现：find -depth 深度优先删除（先删文件再删空目录），并显式排除根部的
# config.ini（用 -path 全路径匹配，子目录里同名文件不会被保护）。

clean-2th:
	@if [ -d $(DIR_2TH) ]; then \
		find $(DIR_2TH) -mindepth 1 -depth ! -path '$(DIR_2TH)/config.ini' -delete; \
		echo "[cleaned] $(DIR_2TH)/ (kept config.ini)"; \
	else \
		echo "[skip] $(DIR_2TH)/ does not exist"; \
	fi

clean-5th:
	@if [ -d $(DIR_5TH) ]; then \
		find $(DIR_5TH) -mindepth 1 -depth ! -path '$(DIR_5TH)/config.ini' -delete; \
		echo "[cleaned] $(DIR_5TH)/ (kept config.ini)"; \
	else \
		echo "[skip] $(DIR_5TH)/ does not exist"; \
	fi

clean-normal:
	@if [ -d $(DIR_NORMAL) ]; then \
		find $(DIR_NORMAL) -mindepth 1 -depth ! -path '$(DIR_NORMAL)/config.ini' -delete; \
		echo "[cleaned] $(DIR_NORMAL)/ (kept config.ini)"; \
	else \
		echo "[skip] $(DIR_NORMAL)/ does not exist"; \
	fi

clean-all: clean-2th clean-5th clean-normal

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
