# 设计：无标记位点(no-label-site)过滤 — 标记方案感知，落地 extract_common

状态：已批准（2026-06-11）。实现范围：extract_common 的 JSON 生成阶段。

## 1. 背景与目标

SILAC 轻重标验证只能作用在**有重标位点**的肽上：轻、重必须有可分辨的质量差。一条**没有标记位点**的肽，其"重标版"与轻标完全相同（轻≡重），本工具**原理上无法验证**——它落在工具能力边界之外（见 speclib 设计 spec §12「工具能力边界」的第 4 类）。

这类肽应在**数据生成阶段（extract_common 写 JSON 时）**就剔除，对 **target 和 trap 都剔**（无标记位点与标签无关，正负例都无法验证）。

关键点：「有没有标记位点」取决于**代谢标记方案**，不能写死成 K/R。

## 2. 标记方案与"标记位点"判据

代码已有 `HeavyType`（`spectrum/psm_info.py`）三种方案：

| 方案 | 重标掺入位置 | 「有标记位点」判据 |
|---|---|---|
| **SILAC** | 仅 K(+8.014)/R(+10.008) 变重（`get_SILAC_increase_mass`） | 序列含 **K 或 R** |
| **CHEAVY (¹³C)** | **每个碳**变 ¹³C（`composition['C']×1.00336`） | 含碳 → **恒为真** |
| **NHEAVY (¹⁵N)** | **每个氮**变 ¹⁵N（`composition['N']×0.99704`） | 含氮 → **恒为真** |

**结论**：只有 **SILAC** 会真正剔除肽（无 K/R 者）；**CHEAVY/NHEAVY 永远不剔**——全原子代谢标记下，任何肽（含骨架 N-Cα-C）都含 C 和 N，必然被标记。这就是「C13/N15 过滤对应的」的含义：各自判据（含 C / 含 N）恒成立 → 等价于 no-op。

## 3. 架构与组件

### 3.1 共享判据 `has_label_site`（`spectrum/psm_info.py`）

```python
def has_label_site(sequence: str, heavy_type: HeavyType = HeavyType.SILAC) -> bool:
    """该肽在给定标记方案下是否有重标位点（能否被轻重标验证）。
    SILAC → 序列含 K 或 R；CHEAVY/NHEAVY（全原子标记）→ True（每条肽都含 C 和 N）。
    空序列 → False。"""
```

- 放在 `psm_info.py`，紧邻 `HeavyType` 与 `get_SILAC_increase_mass`/`get_heavy_increase_mass`（标记领域逻辑集中）。
- 大小写归一（`sequence.upper()`）。

### 3.2 标记方案 config

- 键：`[extract] labeling`，**缺省 `silac`**（向后兼容）。
- 大小写不敏感，接受别名：`silac`→SILAC；`c13`/`13c`/`cheavy`→CHEAVY；`n15`/`15n`/`nheavy`→NHEAVY。
- extract_common 内小解析函数 `_parse_labeling(config) -> HeavyType`；非法值抛清晰 `ValueError`（列出合法值）。

### 3.3 extract_common 过滤 `filter_by_label_site(psms, heavy_type)`

- 遍历 psms，**正负例都查** `has_label_site(seq, heavy_type)`，False 者整条剔除。
- **默认开、无条件运行**（不依赖 `[entrapment]` 段；是 SILAC 工具的硬边界）。
- 日志：打印剔除的 positive / negative 数（与 `filter_by_entrapment` 风格一致）。
- 调用点：`main()` 中、各 PSM `label_type` 已设置之后、写 JSON 之前；与 entrapment L0/L1 过滤并列（顺序无关，均为删行）。

### 3.4 trap_domain_filter 复用

- 现有 `tools/trap_domain_filter.has_label_site(sequence)`（写死 SILAC）改为**委托** `psm_info.has_label_site(seq, HeavyType.SILAC)`，行为不变，消除重复。
- `beyond_tool_limit(..., has_kr=...)` 不变。

## 4. 数据流

```
原始引擎结果 → extract_common 构造带 label_type 的 PSMInfo
  → filter_by_label_site(psms, heavy_type)        # 正负例都剔无标记位点（本设计）
  → (可选) filter_by_entrapment(..., drop_levels) # 负例 L0/L1
  → 写 datasets/*.json
```

## 5. 错误处理与边界

- 非法 `labeling` 值 → `ValueError`（fail-fast，不静默退回）。
- 空序列 → 无标记位点 → 剔。
- 修饰不影响判据：SILAC 重标在 K/R 骨架，修饰叠加其上仍是重标；C13/N15 同理。
- 缺省 silac → 现有所有数据集行为：剔无 K/R（正负例）。

## 6. 测试

- **`has_label_site`**：SILAC 下含/不含 K/R 正确；CHEAVY/NHEAVY 下任意肽（含无 K/R）→ True；空序列 → False。
- **`filter_by_label_site`**：SILAC 下正负例无 K/R 均剔、含 K/R 留；c13/n15 下一条不剔（no-op）；日志计数正确。
- **`_parse_labeling`**：silac/c13/n15 及别名/缺省/非法值。
- **trap_domain_filter 复用**：委托后其既有测试仍通过。

## 7. 范围与非目标

- **范围**：仅 extract_common 的标记位点过滤 + 共享判据 + config 键。
- **非目标（本期不做）**：不让 `single_work`（特征提取）改成 config 驱动标记类型——它暂仍写死 `HeavyType.SILAC`，作为后续单独工作。
- **非目标**：out-of-window（类3）已是提取后概念，不在此设计内。

## 8. 文档

- 更新 speclib 设计 spec §12：类4 现落地 extract_common、scheme-aware，C13/N15 为 no-op。
- 本设计文档归档于 `docs/specs/`。
