# pepdata — 细节

复刻自 `pFindSDK/Reader.cpp:264` `CReader::ReadPepData`。所有字段小端、x64 位宽。

## 二进制布局

逐"蛋白-肽段"条目循环到 EOF：

| 段 | struct | 字节 | 字段 |
|---|---|---|---|
| 头 | `'<IIbbbbIQ'` | 24 | `pro_id`u32, `pep_start`u32, `pep_len`i8, `pro_nc`i8, `enz`i8, `miss`i8, `mod_pep_num`u32, `mod_pep_bytes`u64 |
| 变体 ×mod_pep_num | `'<db'` | 9 | `mass`f64, `mod_cnt`i8 |
| 修饰 ×mod_cnt | `'<bi'` | 5 | `pos`i8, `mod_id`i32 |

- 序列还原：`proteins[pro_id].sequence[pep_start : pep_start+pep_len]`。
- `mod_id` → `mods_by_id`（modification.ini read-order）；找不到时 name=""、mono_mass=0。
- C++ 字段逐个 `file.read`，故二进制流**无 padding**；用 `'<'` 强制标准对齐复现。

## 关键点 / 边界

- **M = Σ mod_pep_num**：一个头条目可含多个修饰变体，每个变体 yield 一个 `LibPeptide`（与 RT/MS2 逐变体对齐的基础）。
- **`mod_pep_bytes` 自校验**：累计变体块消耗字节须等于 `mod_pep_bytes`，否则 `ValueError`。该检查同时确认 `size_t`=8 字节（若位宽错则字节数不符）。
- **`mod_pep_num==0`**：合法，消耗 24B 头、不产出肽段（生成器自然跳过）。
- **生成器而非 list**：真实库 ~312 万变体，全量物化成对象会吃 GB 级内存，故核心用 `iter_pepdata` 逐条产出、即用即弃。用 `mmap` 读（非 `fh.read()`），早退调用方无需读完整 91MB；`LibPeptide`/`ModSite` 用 `slots=True` 加速创建、省内存。
- decoy 判定用蛋白 AC 前缀 `REV_`（`pi.nDecoy` 在 `gen_decoy=false` 下恒 0，库内 decoy 靠 fasta 的 REV_ 条目体现）。

## 验证

对真实 `lib-2th/pepdata.pdb`（91,816,563 B）流式解析：M=3,124,520、`mod_pep_bytes` 全部自洽、序列+修饰重算中性质量与存储 `mass` 100% 吻合（max_err=0）。
