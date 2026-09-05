# pReLoc 负例生成与训练方式：对 ms2-met SILAC hard-negative 管线的可迁移性分析

日期：2026-09-01

## 0. 研究范围与来源

本笔记核对的主要一手来源是本地报告
[pReLoc.pdf](/home/verden/pfind/2026-fall/年中技术报告/pReLoc.pdf)，共 43 个 PDF 页面。
下文的“PDF p.N”均指 PDF 文件页码，不是正文中未显示的印刷页码。仓库侧证据来自当前工作区中的
`tools/counterfactual_negatives.py`、`tools/training_set_builder.py`、`tools/spec_trainer/src/`
和相关设计文档。

需要先说明两个证据边界：

1. 报告清楚描述了候选生成、特征、网络、FLR 公式和数据量，但没有给出完整训练超参数，例如优化器、
   batch size、epoch、温度系数和模型选择规则。因此只能总结报告明确写出的训练协议，不能把它视为
   可直接复现的完整实现说明。
2. 报告中“训练负例”“局部候选”和“FLR decoy”是三个不同对象。若把三者都概括成“负例生成”，
   很容易把统计校准用的诱饵错误地移植为分类训练标签。

## 1. 结论先行

当前项目**有明确可参考之处，但应该借鉴其竞争候选设计原则，而不是照搬它的 FLR decoy 或网络**。

最值得迁移的三点是：

1. **困难负例必须和正确候选共享观测上下文，并尽量只改变局部解释。** pReLoc 保留肽段骨架、
   局部移动修饰；当前 counterfactual 管线则让错误序列 `Q` 继承 parent 的 raw、观测前体、RT 和
   charge。这两者共同的核心是消除“有没有峰/是不是另一条 raw”之类捷径，让模型比较竞争解释。
2. **局部 de novo 应由实际碎片证据约束。** pReLoc 的局部图搜索使用处理后的实验谱峰、b/y 节点、
   氨基酸质量边和 4-residue 滑窗。当前 `synthetic_local_mass_gap` 只做理论等质量局部替换，明确没有
   observed fragment anchors。它是合理的 v1，但还不是 pReLoc 意义上的 spectrum-guided local de novo。
3. **正例和竞争负例应按 candidate family 组织和隔离。** pReLoc 用 InfoNCE 显式拉开正确位点和
   同一候选池内竞争位点；当前仓库已有 `parent_id/group_id/candidate_family_id`，但最终仍是普通
   binary LightGBM。近期先用 family 做严格切分、family 内难度度量和 OOF hard-negative mining；
   数据证明有效后，再考虑 pairwise/listwise 或 contrastive 辅助目标。

最不应直接迁移的三点是：

1. pReLoc 的“骨架保留修饰转移 decoy”用于**位点 FLR 估计**，不是错误肽段身份的训练真值；当前
   synthetic counterfactual 也明确只是 training-only Silver，不能据其数量估计 FDR/FLR。
2. pReLoc 把上游 peptide score 作为贝叶斯先验，因为它假设肽段骨架正确、只判断修饰位置。当前
   `Q` 改变肽段身份，继承 parent 的搜索分数会把正确 parent 的高置信度泄漏给错误 child。
3. pReLoc 的 ESM-2 sequence prior 在位点任务中有合理含义；在本项目中，正例来自蛋白组而合成
   `Q` 被强制排除于蛋白组，ESM 很可能学会“天然序列 vs generator 序列”，形成序列来源捷径。

## 2. pReLoc 中究竟有哪些“负例”

### 2.1 三类对象必须分开

| 对象 | 如何产生 | 用途 | 是否等同于当前 Silver negative |
|---|---|---|---|
| 监督竞争位点/残基 | 高置信参考确定正确位置；同一候选骨架内其他结构上可能的位置标为负 | 训练或验证位点判别 | 只在“同一 family 内竞争”这一抽象层面相似 |
| 局部 de novo 候选 | 在目标位点附近把谱图质量区间建图，搜索高分替代局部路径 | 构造真正难分的候选空间 | 可作为当前 local generator v2 的直接灵感 |
| 骨架保留 decoy | 不改原始肽段序列，把修饰转移到指定非目标残基 | target-decoy 竞争和 FLR 估计 | 不等同；不能直接写成当前分类标签 0 |

报告的 Results 首先明确了四个模块：竞争候选生成、多层次对比评分、肽段置信度整合和 decoy-based
FLR。局部序列搜索在位点附近的质量区间内寻找满足前体与碎片质量约束的替代序列/修饰配置；FLR
decoy 则是另一条路径，严格保留原肽段骨架，仅随机转移修饰（PDF p.3）。

### 2.2 监督正负样本的来源

对比学习语料 D-DS1 的构造流程是：

1. 在 LTQ Orbitrap Velos 的 HeLa 数据上，用 pFind、MSFragger 和 MS-GF+ 搜索同一人类 UniProt；
2. 搜索结果按 1% peptide-level FDR 过滤；
3. 三个引擎一致的 PSM 作为高置信参考；
4. 用 PEAKS 对这些参考谱图做 de novo sequencing，并把 de novo candidate 与高置信参考对齐，形成
   标注训练库；
5. 以**肽段为单位**按约 10:1 划分 train/test；报告给出的训练集为 189,094 PSM、2,537,910 site
   features（PDF p.9；补充表 3 在 PDF p.41）。

报告没有在这一段逐句定义“训练库中每个负 site feature 是怎样枚举的”。它明确给出了两个相邻的
标注协议，可以支持但不能完全替代这项缺失说明：

- 磷酸化 benchmark 中，MaxQuant 和 MSFragger 对 peptide sequence 与 site localization 一致的 PSM
  是正例；这些正例肽段内其他潜在修饰位点被整理为负例（PDF pp.9-10）。
- de novo 位点置信度验证中，和多引擎共识完全匹配的位置是正例，de novo 骨架内所有其他结构上
  可能的位置是负例（PDF p.10）。

因此可以确定 pReLoc 的基本监督单位是“一个已知可靠骨架/谱图下的正确位置与竞争位置”，而不是
从全空间随机抽无关肽段。至于 D-DS1 训练语料是否对所有局部图搜索候选采用完全相同的标注规则，
PDF 没有给出足够细节，不能擅自补全。

### 2.3 局部 de novo 如何生成强竞争候选

Methods 对算法给出了较具体的描述（PDF p.11）：

- 先删除 precursor 和 neutral-loss peaks，做 isotope deconvolution，把峰转成 +1；
- 仅保留强度最高的 150 个峰；
- 图节点包括理论 b/y ions 和起点、终点、`L-1` skeleton 等虚拟节点；
- 实峰节点权重为 `sqrt(intensity)`，虚拟节点权重为 0；
- 当两个节点质量差对应 20 种常见氨基酸或修饰质量时连边；
- 用长度 4 的滑动窗口扫描目标位点侧翼，深度优先搜索局部最优路径；
- 只保留会改变目标位点质量分布的 fragment，再拼回 full-length candidate。

这说明它的“hard”不是由随机 shuffle 的距离定义，而是由**同一真实谱图是否支持一个局部替代解释**
定义。它主动寻找可以解释强峰、同时仅在局部与原候选冲突的路径。

### 2.4 骨架保留 decoy 如何产生与使用

pReLoc 先枚举肽段骨架中所有生物学有效的目标修饰异构体，再按比例生成骨架保留 decoy；target 与
decoy 放进同一个候选池，由同一个模型同时打分，只保留每个 spectrum 的 top-1 配置（PDF pp.11-12）。

decoy 的构造是把整个修饰集合转移到指定的 non-target residue/combination，同时**不改变原始 peptide
sequence**。若 top-1 是 decoy，就记为不能自信定位的事件。某个分数阈值以上的累计 target/decoy
计数，还要用两种搜索空间的大小比进行归一化后才得到 estimated FLR（PDF p.12）。对于 multi-PTM，
整个多位点配置作为不可分割单元竞争，而不是把每个位点独立计数。

报告还直接比较了 sequence perturbation decoy 与 backbone-preserving modification transfer：前者在
高分尾部耗竭，比经验错误定位更弱，造成过于乐观的 FLR；后者的得分分布更接近真实错误定位
（PDF pp.4-5，补充图 5-7 的说明在 PDF pp.28-30）。

这里可迁移的是“保持错误机制之外的上下文不变，并实证检查 decoy 与真实错误分布”；不能迁移的是
“把移动修饰的 decoy 当作错误肽段身份”。两项任务的 null distribution 不同。

## 3. pReLoc 的特征、模型、最终打分与评估

### 3.1 三个特征模态

对每个候选位点，pReLoc 提取（PDF p.11）：

1. 3D MS signal：peptide-spectrum match score、matched-ion ratio、matched-intensity ratio；
2. 1280D sequence semantics：以候选 site 为中心、长度 7 的局部窗口，经
   `esm2_t33_650M_UR50D` 编码；
3. 3D position：absolute position、normalized relative position、peptide length。

需要注意：报告 Results 曾笼统提到“predicted-derived peptide property consistency”（PDF p.3），但
Methods 的明确枚举只有上述三类。若没有代码或补充方法，不能再推断额外的 RT/CCS 等预测特征。

### 3.2 对比学习与分类头

可以确定的高层结构是 dual-branch encoder：一支处理 3D signal，一支处理 ESM+position，融合表示经
归一化后用 InfoNCE 拉开 correct site 与 competitive site，再由 Softmax classification head 输出 site
probability（PDF p.11、补充图 1 在 PDF p.24）。

但报告内部的具体层宽和 dropout **不一致，不能当作可精确复现事实**：

- Methods（PDF p.11）写 signal branch 为两层 `128 -> 64`，semantic-structure branch 为
  `1283 -> 128 -> 64`，两支 dropout 都是 0.2，融合输出为 128D；
- 补充图 1（PDF p.24）画的是 signal `3 -> 128 -> 64`、semantic `1283 -> 256 -> 128`，两个 branch
  dropout 均为 0.1；拼接后为 192D，另有 `192 -> 256 -> 128` 的 fusion block（dropout 0.2），再接
  `128 -> 64 -> 2` classifier head。

因此能借鉴的是“双分支 + family contrastive + classification head”的设计，不是某一套层宽。若要复现
pReLoc，必须取得实际代码或模型配置来消解这一矛盾。

报告 Discussion 还称提供了 XGBoost lightweight variant，并在补充图 14 比较二者（PDF p.13、p.37），
但没有给出 XGBoost 特征子集、超参数或数值表。它不能支持“XGBoost 与对比模型完全等价”的结论。

### 3.3 最终候选级打分

对 database-search result，pReLoc 枚举指定质量容差内的所有目标修饰异构体和比例 decoy，统一评分。
一个多位点 configuration 内的最低 site score 作为该异构体的 baseline score；随后用 Bayesian
calibration 把 site probability 与 upstream peptide matching confidence 结合，最终取校准概率最高的
top-1 configuration（PDF p.11）。

这是一个“peptide identity 已可信、在其内部重定位”的条件推断流程。当前 ms2-met 的问题是判断
peptide identity `Q` 是否正确，不能给错误 `Q` 继承 parent `P` 的 upstream identification prior。

### 3.4 数据划分与评估

报告明确的训练内隔离是 peptide-level 约 10:1 split（PDF p.9），不是 raw-level split。补充表 3 列出：

- D-DS1：训练来源，同时在 PEAKS/Casanovo 上各有约 9.7k-9.9k PSM 的验证；
- D-DS2：HeLa/Q Exactive，约 157k-160k PSM；
- D-DS3：human/yeast mixture，约 34k-37k PSM（PDF p.41）。

报告采用的评估包括：

- localization accuracy；
- known-truth 数据上 empirical true FLR 与 target-decoy estimated FLR 的校准曲线；
- 正确/错误定位分数的 KS distance；
- de novo residue confidence 的 precision-recall curve 和 score density；
- 新位点的 AlphaPeptDeep spectrum similarity 与 DeepMVP sequence-context score；
- 分布汇总用 median/IQR（PDF pp.6-7、p.12、p.37）。

它没有报告 raw 是否跨 split、同一谱图的多个 candidate 如何跨 train/test 去重、训练集和 D-DS1
验证集的精确关系、重复训练随机种子、置信区间或完整训练超参数。特别是 Methods 只写“peptide-level
split”，因此不能据此宣称已经完成 raw/instrument/batch 级独立验证。

此外，pReLoc 的 FLR 是**位点错误率**，PR 图通常把正确 residue/site 当关注对象；本仓库的规范则
固定以错误鉴定为统计阳性类。二者数值和字段不能直接混用。

## 4. 当前仓库已做到什么

### 4.1 与 pReLoc 原则已经一致的部分

当前 counterfactual child `Q` 继承 parent `P` 的 observed raw、precursor m/z、RT 和 charge，随后用
`Q` 自己的 sequence 计算 light fragments 和 heavy coordinates
（[counterfactual_negatives.py](../../tools/counterfactual_negatives.py#L1)，
[child 构造](../../tools/counterfactual_negatives.py#L582)）。这已经实现了与 pReLoc “保留观测上下文”
相同的深层原则，而且比把 Q 重新搜索到别的 raw 更接近真正的 counterfactual test。

v1 提供三种 proposal source：composition shuffle、K/R-position shuffle 和 local mass-gap
（[来源常量](../../tools/counterfactual_negatives.py#L54)，
[配置](../../tools/counterfactual_negatives.py#L65)）。候选有效性要求：

- Q 与 P 在 L/I 归一化后有最小序列差异；
- Q 不属于当前 target/contaminant exact-or-L/I substring；
- Q theoretical precursor 与继承的 observed precursor 在质量容差内；
- 有足够多 distinguishing fragment positions；
- local proposal 还要求最低 theoretical fragment overlap
  （[有效性检查](../../tools/counterfactual_negatives.py#L360)）。

K/R 数量和位置不是统一有效性条件。manifest 只记录 parent/candidate 的 K/R count、position match 和
各自 label shift，实际 heavy shift 由 Q 重新计算
（[counterfactual_negatives.py](../../tools/counterfactual_negatives.py#L401)）。这符合“先广泛生成，
再按真实证据定义 hardness”的思路。

### 4.2 当前 local proposal 与 pReLoc 的差距

当前 `_local_mass_gap_proposal` 随机选择不包含末端酶切残基的 2-4 aa 区间，从预计算 residue-string
mass index 中找不同组成、但总质量在 precursor tolerance 内的 replacement
（[counterfactual_negatives.py](../../tools/counterfactual_negatives.py#L293)）。它没有读取谱峰，
manifest 也明确写入 `local_uses_observed_fragment_anchors=False`
（[counterfactual_negatives.py](../../tools/counterfactual_negatives.py#L615)）。audit 进一步声明 v1
只是 mass-gap proposal，hardness 必须在真实 light/heavy feature extraction 后再定义
（[counterfactual_negatives.py](../../tools/counterfactual_negatives.py#L633)；
[设计规范](../specs/2026-09-01-counterfactual-hard-negatives.md#local-mass-gap-proposal)）。

所以，“已经做了局部 de novo”只能作为接口/source 名称的宽泛说法；科学上应称
`local_mass_gap_v1`，不能称 observed-spectrum-guided de novo。

### 4.3 训练集组装与防捷径

builder 对 Silver 要求同时存在真实 light/heavy precursor intensity、light/heavy fragment count 和
acquisition-range evidence；还可要求 XIC non-empty、heavy in range
（[training_set_builder.py](../../tools/training_set_builder.py#L688)）。随后再次排除 target-like candidate，
并按 charge、precursor m/z、sequence length、label shift、window geometry、RT 等字段匹配正例分布
（[distribution matching](../../tools/training_set_builder.py#L803)，
[assembly flow](../../tools/training_set_builder.py#L1167)）。

parent 与 child 共用 `group_id`，query provenance 用稳定 `query_id` 连接，避免相同 candidate sequence
在不同 parent/raw 下被错误合并
（[query-aware join](../../tools/training_set_builder.py#L1112)）。组装前还检查 train raw 与 immutable
heldout raw 不重叠（[training_set_builder.py](../../tools/training_set_builder.py#L643)）。

生成器元数据被显式列为 metadata，不进入正式模型；search `q_value` 也被训练 registry 排除
（[feature_groups.py](../../tools/spec_trainer/src/feature_groups.py#L19)）。训练集组装还计算一个 metadata
generator-signature AUC，AUC 偏高提示 synthetic shortcut
（[training_set_builder.py](../../tools/training_set_builder.py#L939)）。

这些保护措施与 pReLoc “候选要共享竞争上下文”的经验方向一致，而且更适合本项目的 DIA/SILAC
观测模型。

### 4.4 当前最终训练与评价

正式训练入口使用 production LightGBM，按配置选择 group-aware CV，外折生成 leak-free OOF score，
每个外折内部另划 early-stopping validation
（[cv_train.py](../../tools/spec_trainer/src/cv_train.py#L1)，
[assemble_oof](../../tools/spec_trainer/src/cv_train.py#L453)）。外部测试由各折模型集成；正式 locked
working point 使用各 member 的 outer-OOF threshold 和 vote，而不是把 pooled OOF threshold 直接套到
ensemble average 上（[cv_train.py](../../tools/spec_trainer/src/cv_train.py#L598)）。

存储与模型输出仍为 `label=1`/high score = correct；评价边界显式变换为
`error_truth=1-stored_label`、`error_score=1-trust_score`
（[cv_core.py](../../tools/spec_trainer/src/cv_core.py#L1)）。因此未来即使增加 contrastive auxiliary loss，
最终 trust head 和所有机器可读评估仍必须保留
`metric_semantics=error_identification_positive_v1`，不能照搬 pReLoc 的 site-positive PR/FLR 字段。

## 5. 可以直接借鉴的开发点

### 5.1 第一优先：把 local mass-gap v1 深化为 observed-anchor v2

建议保留现有 `build_counterfactual_negatives(...)` 深模块接口，增加一个新的 proposal source，而不是
改写现有 v1。v2 可以借鉴 pReLoc 的窗口图搜索，但必须改成适合 DIA-SILAC 的证据：

1. 仅在预先划定的 train raw 中，从 parent `P` 选择 2-4 aa 局部窗口；
2. 用窗口边界附近可观测的 b/y fragment 作为 light-side anchors，而不是只用全肽 precursor mass；
3. 在质量图上枚举 precursor-compatible 的局部替代路径 `Q`，保留足够 shared fragments 和少量但
   明确的 distinguishing fragments；
4. 对每个 Q 用 Q 的 K/R 分布计算 fragment-specific heavy coordinates。若局部替换改变 K/R，图状态
   需要携带 prefix/suffix 的 label count，不能只使用一个全局 precursor shift；
5. proposal 阶段优先要求“light spectrum/XIC 可以较强解释 Q”，真实 light/heavy feature extraction
   后再分成 no-signal、partial-interference、high-interference 和 OOF-adversarial tiers；
6. 每个 parent 保留少量多样化 top-K，避免一个谱图产生大量高度相关 children。

pReLoc 的节点权重来自单张 DDA MS/MS 峰强度；本项目不能机械采用 top-150 peak 图。DIA 中更可靠的
节点/边分数应组合 fragment XIC 共洗脱、质量误差、peak shape，以及对应 heavy trace 的配对一致性。
报告自己也承认其方法主要针对 DDA，扩展到高度 multiplexed/chimeric DIA 是未解决方向（PDF p.13）。

### 5.2 第二优先：把 family 从“只防泄漏”提升为“难度和训练单位”

对每个 `P + {Q_i}` family，先增加不进入模型的诊断：

- `light_fit(Q) - light_fit(P)`；
- `paired_fit(Q) - paired_fit(P)`；
- shared/distinguishing fragment evidence；
- Q heavy evidence 是 absent、interference 还是 deceptively coherent；
- 当前基线模型对 Q 的 OOF trust score 与 family margin。

这样可回答“Q 是否真的是强负例”，而不是把 generator 名称当 hardness。第一轮仍可使用现有 binary
LightGBM，只在 sampling/mining 中使用 family。若真实 heldout 上稳定获益，再增加以下受控实验：

- pairwise margin：要求 `trust(P) > trust(Q_i)`；
- family softmax/listwise ranking：同一 family 的正确 P 胜出；
- contrastive embedding + 原有 calibrated binary trust head。

不能只报告 family ranking accuracy，因为生产任务仍要对单个 PSM 输出 `P(correct)` 并在正确 ID 上
控制 false-alarm rate。

### 5.3 第三优先：加强 parent 真值，而不是只扩 generator

pReLoc 用三搜索引擎共识构造高置信 reference。本仓库现已增加
[`tools.counterfactual_parents`](../../tools/counterfactual_parents.py)。本数据的 JSON 已经过前置过滤，
因此 `label_type=positive` 是权威 parent 真值；准备模块不再要求调用者用第二张 confirmation table
重复声明同一事实。模块结合 raw split manifest，并在 PSM、manifest 和 audit 中携带或验证：

- `filtered_input_label_type_positive_v1` 真值规则和 prepared-parent 标记；
- raw/split provenance；
- parent light/heavy evidence eligibility；
- 可选的多引擎/谱库共识，但这些 upstream scores 只用于 parent selection，不作为 child model feature。

高纯度 parent 比再增加一种随机 shuffle 更能降低错误监督传播。

### 5.4 第四优先：实证比较 synthetic 与真实错误分布

借鉴 pReLoc 对 decoy score distribution 与 empirical false target 的比较，应在开发集上比较各 synthetic
source/tier 与真实 entrapment：

- 完整允许模型特征上的 source-classifier grouped-OOF AUC；
- trust/error score density 与 KS distance；
- light-only、heavy-only、paired-evidence family 的覆盖差异；
- 加入某 source 后，immutable real entrapment 上的 `error_pr_auc`、`fnr_at_fpr5`、
  `error_recall_at_fpr10` 和 `fpr_1/fpr_5/fpr_10`。

这些统计只能在 dev 上用于选 source/tier；最终 heldout entrapment 不得用于 generator 参数选择、hardness
cutoff、feature shortlist 或反复试验。

## 6. 不适用之处与泄漏风险

### 6.1 任务层级不同

pReLoc 的主要任务是“peptide sequence 正确时，PTM 在哪个 residue”；当前项目是“这个 peptide ID
是否被独立 light/heavy 证据支持”。pReLoc 可以严格保留 sequence 并只移动 modification；当前若保留
全部 sequence 就没有错误 identity 可判。应迁移“最小局部反事实”原则，而非字面构造。

### 6.2 statistical decoy 与 supervised negative 不可混用

当前 synthetic Q 是 training-only Silver，并非 search-engine result 或 FDR decoy
（[设计规范](../specs/2026-09-01-counterfactual-hard-negatives.md#goal)）。在没有证明 target/decoy
exchangeability 和搜索空间比例之前：

- 不可用 Q 的数量估计 FDR；
- 不可用 Q 的 score tail 直接锁生产阈值；
- 不可把 synthetic-only test 的高 AUC 当真实错误识别性能。

最终统计评价继续复用 `cv_core.py`，错误鉴定为实际阳性类。

### 6.3 ESM/sequence prior 可能成为 generator detector

pReLoc 的 positive 与 negative 多数共享同一 peptide backbone，ESM site-context 主要表达局部修饰偏好。
当前 positive 是天然 target peptide，而 Q 被要求不在 target/contaminant exact-or-L/I substring；加入 ESM
会让模型直接学习 sequence naturalness 或 FASTA membership 的代理。除非候选已在语言模型似然、
组成、长度、酶切、蛋白组近邻距离上严格匹配，并通过跨 generator/真实 entrapment 验证，否则不应把
ESM feature 加入主模型。

### 6.4 上游 parent score 泄漏

对 Q 复制 P 的 q-value、search score、library score 或 heavy-confirmation score，会给错误候选注入正确
parent 的标签信息；若不复制而让 synthetic 值缺失，又可能产生 missingness shortcut。当前 registry
把 `q_value` 排除是正确做法。pReLoc 的 Bayesian peptide prior 不适用于身份被替换的 Q。

### 6.5 local de novo candidate 不一定真错

DIA 谱图可能是 chimeric；一个解释 light peaks 很好的 Q 可能来自共洗脱真实肽段、未收录 isoform、
variant、PTM 或 contaminant。当前 exact/L/I target exclusion 也明确不是完整 proteome-neighbour scan
（[设计规范](../specs/2026-09-01-counterfactual-hard-negatives.md#validity-versus-hardness)）。因此 local
candidate 应继续标为 Silver，不能因“不是 parent P”就升级为 Gold negative。必要时增加更完整的
proteome/variant/PTM exclusion 和多 raw 重复证据审计。

### 6.6 peptide-level split key

当前 `parent_id` 由 `parent_sequence + charge` 生成，随后 child 使用它作为 `group_id`
（[counterfactual_negatives.py](../../tools/counterfactual_negatives.py#L542)）。这能绑定一个具体
sequence/charge family，也会把同一 sequence/charge 跨 raw 绑定在一起；但同一 peptide 的不同 charge
可能落入不同 CV fold。pReLoc 报告明确采用 peptide-level split。Parent preparation 现已生成
`peptide_group_id = hash(L/I-normalized parent sequence)` 并传播到所有 child。正式训练必须用它或包含
它的 connected grouping 做 CV group；`candidate_family_id` 继续表示具体 parent hypothesis，二者不要
混为一个字段。

### 6.7 hard-negative mining 的循环选择

如果先用完整数据训练模型，再选“模型最容易误信的 Q”，最后在同一折上报告提升，会造成选择泄漏。
正确做法是：每个 outer training fold 内用 inner/OOF score 挖 hard negatives，outer fold 从不参与该折的
选择；immutable heldout 只在方案冻结后评一次。generator 的 observed-evidence score 也应固定定义，
不能根据最终测试表现反复调。

## 7. 建议的下一阶段实验顺序

1. **准备受控 parent。** 对上游过滤 JSON 和 raw split manifest 运行 `tools.counterfactual_parents`，检查
   `label_type=positive` 真值规则、input fingerprint 和 `peptide_group_id` audit。
2. **跑 v1 基线。** 在真实 SILAC train raw 生成三类 Q，提取普通 `feature_type=0` light/heavy 特征；
   不改模型，验证各 source 的产量、物理 gate、source shortcut 和真实 entrapment 增益。
3. **实现 `synthetic_local_observed_anchor_v2`。** 先用 light fragment/XIC anchors 枚举局部 Q，再用 Q-specific
   heavy coordinates 提取配对证据。保留 v1 作为独立对照，不静默改变 generator 语义。
4. **做 source/tier ablation。** 比较 Gold-only、Gold+各 source、Gold+混合、不同 hardness tier；所有
   实验使用相同正确样本、相同 grouped folds 和相同 immutable test。
5. **最后测试 family-aware objective。** 只有当 observed-anchor Q 在真实 entrapment 上稳定带来收益，
   才比较 binary LightGBM、pairwise/listwise 和 contrastive+trust-head；不要在数据尚未验证前先扩大模型。
6. **replicate domain 后置。** 真实 SILAC 上先验证 negative mechanism；后续 rep 只替换 partner-coordinate
   adapter（跨 raw、identity m/z、aligned RT），不改 candidate family、truth、split 和评价契约。

## 8. 最终判断

pReLoc 对当前工作的最强启示不是“使用某个特定深度模型”，而是：**错误候选必须是同一观测对象的
局部、合理、会参与真实竞争的替代解释；统计 decoy、监督负例和最终评估真值必须分层处理。**

当前仓库的 inherited raw/mz/RT、Q-specific heavy shift、parent-child grouping、Silver provenance、
physical signal gate 和 immutable entrapment test 已经搭好了正确骨架。最实质的下一步是把
`local_mass_gap_v1` 深化为 DIA-SILAC observed-anchor generator，并在 peptide-level nested split 下证明
它改善真实错误识别；不是立即复制 pReLoc 的 ESM、Bayesian peptide prior 或 FLR decoy 公式。
