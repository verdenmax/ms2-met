# SILAC、¹³C 与 ¹⁵N 代谢标记对比图设计

**日期：** 2026-07-23

**目标：** 绘制两张内容一致、布局不同的英文 SVG 说明图，清楚比较 SILAC、全碳 ¹³C 与全氮 ¹⁵N 代谢标记对轻/重母离子和碎片离子的影响。图用于论文或 PPT，并沿用 `analysis/workflow_steps/silac_adaptation_experiment_workflow.svg` 的视觉风格。

## 1. 交付物

在 `analysis/workflow_steps/` 下新增：

1. `silac_c13_n15_comparison_matrix.svg`
   - A 版：三列对照矩阵。
2. `silac_c13_n15_comparison_radial.svg`
   - C 版：共享肽段、三向分支。
3. 一张由上述 SVG 渲染并排组成的 PNG 预览，便于快速比较两种布局。

两张 SVG 均使用透明画布、可编辑矢量元素和可编辑文本，不嵌入位图。

## 2. 共享科学内容

两版使用同一个无修饰示例肽段 `YLYEIAR`，从而把视觉差异限制在标记机制本身。

### 2.1 SILAC

- 示例肽段只有末端 Arg 标记位点。
- Heavy Arg 使用 Arg10：`¹³C₆¹⁵N₄`，完整肽段质量增加 `10.008 Da`。
- MS1 母离子整体产生确定的轻重质量间隔；图中以 `Δm/z = 10.008 / z` 表示。
- 不含 Arg 的 b fragments 轻重质量相同，峰保持对齐。
- 含末端 Arg 的 y fragments 产生 `10.008 Da` 的中性质量位移；图中以带配对指示的轻重峰表示。
- 核心概念：`Site-specific labeling`、`K/R residues only`。

### 2.2 全碳 ¹³C 标记

- 肽段中的所有碳原子由 ¹²C 替换为 ¹³C。
- MS1 中性质量位移为 `n(C) × 1.003355 Da`，m/z 位移再除以电荷 `z`。
- 每条 b/y fragment 都包含碳，因此均产生位移。
- 不同 fragment 的碳原子数不同，位移大小也不同；示意谱用不等长度的横向轻重间隔表达。
- 核心概念：`Composition-dependent labeling`、`All carbon atoms`。

### 2.3 全氮 ¹⁵N 标记

- 肽段中的所有氮原子由 ¹⁴N 替换为 ¹⁵N。
- MS1 中性质量位移为 `n(N) × 0.997035 Da`，m/z 位移再除以电荷 `z`。
- 每条 b/y fragment 都包含至少一个氮，因此均产生位移。
- 不同 fragment 的氮原子数不同，位移大小也不同；示意谱用不等长度的横向轻重间隔表达。
- 核心概念：`Composition-dependent labeling`、`All nitrogen atoms`。

## 3. A 版：三列对照矩阵

画布约为 `1600 × 1000`。主标题为：

`SILAC vs. ¹³C vs. ¹⁵N Metabolic Labeling`

三列从左到右分别为 SILAC、¹³C 和 ¹⁵N。每列严格使用相同的信息层级：

1. **Labeling rule**
   - 用圆形残基链展示 `YLYEIAR`。
   - SILAC 只突出末端 R；¹³C/¹⁵N 则以覆盖整条肽段的原子标记带表示全原子标记，避免误导为“每个残基增加相同质量”。
2. **MS1 precursor**
   - 使用简化同位素峰簇展示 Light 与 Heavy。
   - 紫色括号标注对应 `Δm/z` 规则。
3. **MS2 fragments**
   - 以共享 m/z 轴的镜像谱表示：Light 向上、Heavy 向下。
   - SILAC 中 b ions 保持同一 x 位置，y ions 成对位移。
   - ¹³C/¹⁵N 中所有选定 b/y ions 均位移，且不同离子对的位移宽度不同。
4. **Mass-shift rule**
   - 给出紧凑公式与一句机制总结。

底部共享总结条：

`SILAC: site-specific fragment shifts  |  ¹³C / ¹⁵N: composition-dependent shifts in every fragment`

## 4. C 版：共享肽段、三向分支

画布约为 `1600 × 1100`。中央放置 Light peptide `YLYEIAR`，从中央分出三条带箭头的路径：

- 左上：SILAC
- 右上：¹³C
- 下方：¹⁵N

每条路径进入一个浅色圆角卡片。三张卡片保持相同内部结构：

1. 标记名称与范围标签。
2. Heavy peptide 标记示意。
3. 紧凑型 MS1 轻重峰簇。
4. 紧凑型 MS2 镜像谱。
5. 质量偏移公式与一句结论。

该版强调“同一条轻标肽段经过不同标记策略后产生三种重标结果”，并通过放射结构增强讲解感。卡片内不重复长段解释，以保证缩放后仍可读。

## 5. 视觉规范

- 背景：透明。
- Light cyan：`#4DBBD5`。
- Heavy green：`#00A087`。
- 深蓝结构线：`#3C5488`。
- 正文深灰：`#2B2B2B`。
- 辅助灰蓝：`#59667A` / `#8491B4`。
- 质量差紫色：`#8B68AD`。
- 浅色卡片：参考 `#D2EEF4`、`#E8F5F1`，低透明度填充。
- 字体：`Arial, Helvetica, sans-serif`；仅在需要兼容特殊字符时保留通用 fallback。
- 线条：圆角端点，主轴约 2.5–3 px，谱峰约 4–6 px，配对引导线使用短虚线。
- 标题和标签均为英文；同位素使用 Unicode 上标 `¹³C`、`¹⁵N`。

## 6. 准确性边界

- 图中谱峰强度和横向位置是机制示意，不表示真实实验数据。
- 图中明确标注 `schematic` 或在说明文字中表达 `not to scale`。
- 使用中性质量增量公式时写 `ΔM`；使用谱图横轴间隔时写 `Δm/z = ΔM / z`，不混用单位。
- ¹³C/¹⁵N 的 fragment shift 按各 fragment 自身元素组成变化，不画成统一固定间隔。
- 不表现修饰基团中的 C/N 标记，因为示例肽段无修饰，且当前仓库对带修饰 CHEAVY/NHEAVY 的重标质量计算明确不支持。

## 7. 验证

1. 使用 XML 解析器验证两张 SVG 语法。
2. 使用 `rsvg-convert` 渲染 SVG，确认无缺字、裁切、重叠或 marker 冲突。
3. 检查图内公式、同位素上标、SILAC b/y 位移逻辑和 ¹³C/¹⁵N 可变 fragment shift。
4. 生成并排 PNG 预览，比较两版在论文/PPT 常见缩放下的可读性。
5. 检查 `git diff --check`，确保新增文件无格式错误。
