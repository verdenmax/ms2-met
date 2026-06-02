# 默认目标
.PHONY: all clean

# 所有实验名称（你可以按需增减）
EXPS := exp1 exp2

# 默认目标：运行所有实验
all: $(EXPS)

# 为每个实验生成一个 target
$(EXPS):
	@echo "🚀 Running experiment: $@"
	@mkdir -p models results figures
	python src/main.py --config config/$@.yaml --name $@
	@touch $@.done  # 标记已完成，避免重复运行

# 强制重新运行某个实验
re-%:
	rm -f $*.done
	$(MAKE) $*

# 清理所有输出（保留原始数据和配置）
clean:
	rm -rf models/* results/* figures/*
	rm -f *.done

# 快速查看结果
results:
	@for f in results/*.json; do \
		echo "=== $$f ==="; \
		cat $$f | python -m json.tool; \
	done

# 帮助信息
help:
	@echo "Available targets:"
	@echo "  make all          # 运行所有实验"
	@echo "  make exp1         # 运行 exp1"
	@echo "  make re-exp1      # 重新运行 exp1（忽略缓存）"
	@echo "  make clean        # 清除模型、结果、图表"
	@echo "  make results      # 打印所有 JSON 结果"
