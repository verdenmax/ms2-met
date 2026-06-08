# predictions — 细节

复刻 `pPred/pPredRT.cpp::predict`（RT）与 `pPred/pPredMS2.cpp::predict`（MS2）的二进制写出。

## RT：pepdata.rt.predb

- 紧凑 `M × f32`，顺序与 pdb 肽段变体顺序一致（每变体一个）。值为真实 RT（分钟，写出前已 `norm2real`）。
- `read_rt_pred` 用 `array('f').frombytes`，~12MB 全量加载、按下标随机访问。

## MS2：pepdata.ms2.predb = `[二进制记录区][文本尾巴]`

### 二进制记录（每"肽段-电荷"一条）

| 段 | struct | 字节 | 字段 |
|---|---|---|---|
| 头 | `'<h'` | 2 | `n_size`i16 |
| 离子 ×n_size | `'<bbf'` | 6 | `pos`i8, `iontype`i8, `inten`f32 |

- 记录顺序：肽段外层、电荷 `1..chg_max` 内层（共 M×chg_max 条）。
- iontype 编码：偶=`b`、奇=`y`；`frag_charge = iontype//2 + 1`（0=b⁺,1=y⁺,2=b²⁺,3=y²⁺,…）。
- `pos` 0-indexed 切割位；离子按强度降序、已截断（`MAX_ION_OUTPUT=1000` 上限）。

### 文本尾巴（必须跳过）

- 二进制区之后紧跟 `M` 行 ASCII：`"1\t0\t2\t0\t…\tchg_max\t0\t\n"`。
- **成因**：`pPredMS2.cpp:868-873` 的收尾 `fprintf` 循环在 `if(binary)` 之外，二进制模式下 `curr_pep_id=0/curr_pep_chg=1` 未更新，对每肽段误写一行文本。pFind 引擎只读前 M×chg_max 条、忽略尾巴。
- **停止规则**：尾巴首 2 字节 `'1'(0x31)'\t'(0x09)` 作小端 i16 = 0x0931 = 2353 > 1000，故读取遇 `n_size<0 或 >max_ions` 即停。
- **流式实现**：`iter_ms2_records` 用 `mmap`（非 `fh.read()`），真实文件 ~4.4GB 时保持 O(1) 常驻内存；并对 `off + n_size*6 > 文件尾` 做截断防御（干净停止）。
- **RT 字节序**：`read_rt_pred` 用 `array('f')`（本机字节序），文件为小端；非小端主机上 `byteswap()` 纠正。

## 边界 / 坑

- **`n_size==0` 空记录照样存在**（每"肽段-电荷"恒有 2 字节头），`iter_ms2_records` 产出 `[]`，绝不可当 EOF 跳过，否则后续全部错位。
- `read_chg_max_from_trailer`：读末尾 `tail_bytes`，按 `\n` 切行、`reversed` 取最后一条形如 `"1\t0\t…\tC\t0\t"` 的干净行；charges 在偶数 token 下标，要求恰为 `1..C` 连续。需库 ≥2 个肽段（保证至少一条尾巴行不被前面二进制字节污染）。

## 验证

真实 `lib-2th/pepdata.ms2.predb`（4,465,990,502 B）：守卫停在尾巴 → 有效记录 12,498,080 = 4×M，尾巴 17×M = 53,116,840 B；chg_max=4；rt.predb=3,124,520 个 f32 = M，值 7–77 min。
