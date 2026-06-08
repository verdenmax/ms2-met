# predictions — 职责与接口

## 一句话职责

读取预测 RT（全量数组）与预测 MS2（流式逐记录，自动跳过文件末尾的文本尾巴），并从尾巴解析 `chg_max`。

## 对外接口

| 符号 | 签名 | 简述 |
|---|---|---|
| `read_rt_pred(path)` | → `array('f')` | M 个 float32（分钟），全量加载 |
| `iter_ms2_records(path, max_ions=1000)` | → 生成器[`list[FragIon]`] | 逐记录产出离子；遇文本尾巴停 |
| `read_chg_max_from_trailer(path, tail_bytes=8192)` | → `int` | 从尾巴行解析 chg_max |
| `FragIon` | dataclass(`ion_type`,`frag_pos`,`frag_charge`,`intensity`) | 一个碎片离子 |
| `MAX_ION_OUTPUT` | `int` = 1000 | 单记录离子上限（亦作尾巴判据）|

## 依赖

- 依赖：标准库 `struct`/`array`/`os`。
- 被依赖：`speclib`（锁步流式用 `read_rt_pred`+`iter_ms2_records`+`read_chg_max_from_trailer`）。

## 输入 / 输出

- 输入：`pepdata.rt.predb`、`pepdata.ms2.predb` 路径。
- 输出：RT 数组 / MS2 离子记录流 / chg_max 整数。
