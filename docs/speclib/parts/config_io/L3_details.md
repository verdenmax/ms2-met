# config_io — 细节

复刻自 pFindSDK：`fastaparser.cpp`、`Reader.cpp`（`ReadMod`/`ReadAA`/`ReadElementInfo`）。

## FASTA 解析（`CFastaParser::ReadOnePrteinEntry`）

- 条目以 `>` 行起始，后续非 `>` 行 `strip()` 后拼成 `sequence`。
- AC 切分：取 `>` 后到**第一个空格 / Tab** 之间；额外规则——第一个 `|` 仅当其位置 `> 15` 时才作为分隔符（uniprot 头 `sp|P12345|NAME` 的首个 `|` 在 index 3 < 15，故不分隔，AC 取到第一个空格 = `sp|P12345|NAME`）。
- `pro_id` = 蛋白在文件中的出现顺序（list 下标）。

## modification.ini（`CReader::ReadMod`）

- 逐行，仅取含 `=` 的"数据行"，**按文件顺序**赋 1-based `mod_id`。
- 跳过（**不占 id**）：`key` 以 `name` 开头、`@NUMBER_MODIFICATION`、`label_name`、`Met-loss+Acetyl[ProteinN-termM]`、`key` 以 `Label_` 开头。
- 数据行格式 `名称=位点 类型 单同位素质量 平均质量 NL数 ... 元素组成`；`mono_mass` 取第 3 个 token（`parts[2]`）。
- C++ 中 `m_vAllMod` 按质量排序但 `m_vID2Mod` 仍以 read-order 的 `m_nID` 索引，故二进制 `mod_id` = 过滤后 read-order 1-based。实测 Carbamidomethyl[C]=9、Oxidation[M]=46。

## element.ini（`CReader::ReadElementInfo`）

- 行 `E<n>=名称|质量列表|丰度列表|`；取**丰度最高**同位素对应的质量。CHNOS 最高丰度即最轻同位素（≈ 单同位素质量）。

## aa.ini（`CReader::ReadAA`）

- 行 `R<n>=氨基酸|组成|`，组成如 `C(3)H(5)N(1)O(1)S(0)`。
- 残基质量 = Σ 元素质量 × 个数，**不含水**。水在质量校验时单独加（`water_mass = 2H + O = 18.0105646837`）。

## 边界 / 坑

- 全部用 `latin-1` 打开（modification.ini 等可能含非 UTF-8 字节）。
- `B/J/O/U/X/Z` 等在 aa.ini 中 `C(0)` → 质量 0；真实肽段不含（`X` 已被上游剔除）。

## 验证

各解析对真实 `../puku/*.ini`、`merge_human_ecoli_yeast.fasta` 跑通：water=18.0105646837、Gly=57.02146、Lys=128.09496、Ala=71.03711，AC 切分与 C++ 一致。
