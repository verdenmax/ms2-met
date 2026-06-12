# 设计：支持 PFB 谱图文件格式（pFind/pXtract 二进制）

> 文档版本：1.0（2026-06-12）
> 状态：设计已评审通过（格式经真实样例实测验证）；待写实现计划
> 关联：
> - 集成点：`manager/data_manager.py:get_dia_data_object`（raw 路径 → DIAData 的唯一入口）、`spectrum/dia_data.py`（`_load_from_mzml` / `_process_single_spectrum` / `_preallocate_arrays`）
> - 现有测试网：`tests/test_dia_data_load_mzml.py`（19）、`tests/test_dia_data_window.py`（9）、`tests/test_dia_cache.py`、`tests/test_centroid_*`
> - 实测样例：`~/share/2026_04_27_kongweisa_diann_ZHOUHUdataset/2th/20190830_HF_ZHW_hela_SILAC_DDIA_*_2Da_Rep*.pfb`（与现有 mzML 同批 HeLa SILAC DDIA 数据）

---

## 1. 背景与动机

主流程把 DIA raw 数据读成 `DIAData`（扁平峰数组 + 逐谱索引 + DIA 窗口信息），目前**仅支持 mzML**（`spectrum/dia_data.py:_load_from_mzml`，经 pyteomics）。`manager/data_manager.py:get_dia_data_object` 是 raw 路径 → `DIAData` 的唯一入口。

需求：让 config 里的 `raw_path_*` 可以直接填 **`.pfb`** 文件，走同一条流水线（`DataManager → DIAData`），与 mzML **完全等价**——下游特征提取、XIC、DIA 窗口逻辑无感知。

PFB 是 pFind/pXtract 的二进制谱图格式：一次写入全部一级/二级谱（峰列表 + 属性串），文件尾部带逐谱索引（footer addr_list）支持随机访问。

---

## 2. 目标与非目标

**目标**
1. 新增纯解析模块 `spectrum/pfb_reader.py`，把 PFB 解析成统一的逐谱记录。
2. `DIAData._load_from_pfb(path)`：用 PFBReader 做 2-pass 加载，产出与 `_load_from_mzml` **等价**的 `DIAData`。
3. 抽取 mzML 路径中**与格式无关**的两段逻辑为共享方法（`_record_spectrum` / `_finalize_arrays`），mzML 与 PFB 共用（DRY）。
4. `get_dia_data_object` 按扩展名分派 `.pfb` / mzML。
5. npz 缓存复用（源文件身份 = `.pfb` 的 mtime/size）。

**非目标（YAGNI）**
- 基于 footer 的**随机访问按 scan id 取谱**（全量加载只需顺序读；footer 仅可选做一次完整性自检）。
- **写出** PFB。
- 离子淌度 / TIMS（目标数据为 FTMS，`has_mobility` 保持默认 False）。
- 对 PFB 再做 on-load 质心化（PFB 已是 peak-picked，见 §6 决策）。
- MS2 缺 `activation_window` 时的默认窗宽（目标数据均有；缺失则明确报错，见 §7）。

---

## 3. PFB 格式（经真实样例实测验证）

全部为小端（little-endian，x86 native）。下列结构与字节数已用 `~/share/.../Rep1.pfb`（813,753,733 字节，scan_num=80096）逐字段核对通过。

### 3.1 Header（24 字节）
| 字段 | 类型 | 字节 | 说明 |
|---|---|---|---|
| Empty_Property_1/2/3 | int×3 | 12 | 预留，实测均为 0 |
| Addr_List_Addr | long long | 8 | footer addr_list 的首地址 |
| Scan_Num | int | 4 | 谱图总数（MS1+MS2） |

struct 格式：`"<iiiqi"`。**Header 共 24 字节**（实测 `addr_list[0]==24`，即第一个谱紧接 header 之后；原始规范注释"期望=20"有误）。

### 3.2 Loop Body（重复 Scan_Num 次）
| 字段 | 类型 | 说明 |
|---|---|---|
| Property_Str_Len | int | 属性串字节长度 |
| Property_Str | char[len] | UTF-8，`\t` 分隔；尾部可能含 `\x00`，需 `rstrip("\x00")` |
| Peak_Num | int | 谱峰数 |
| All_Peak_Mz | double[Peak_Num] | 质荷比 |
| All_Peak_Intensity | **double**[Peak_Num] | 强度（**实测为 double**；原始规范写入片段的 `sizeof(float)` 是笔误，读取片段与表格均为 double，且按 double 解析时偏移与 footer 完全对齐） |

单谱字节数 = `4 + len + 4 + Peak_Num*8 + Peak_Num*8`。

### 3.3 Property_Str 字段布局（按 token 位置，0-based）
解析时**先取 token[1] = MsType** 决定布局：

MS1（MsType=1，**4 字段**）：
`[0]Scan(int) [1]MsType(int)=1 [2]RT(float) [3]InstrumentType(str)`

MS2（MsType=2，**13 字段**）：
MS1 四项 + `[4]Charge(int) [5]MH+(float) [6]IonInjectionTime(float) [7]ActivationCenter(float) [8]ActivationType(str) [9]PrecursorScan(int) [10]ActivationWindow(double) [11]CollisionEnergy_NCE(double) [12]MonoisotopicMz(double)`

实测样例（MS2 scan 2）：`2  2  0.4538569  FTMS  2  1000.993  63  501  HCD  1  2  27.00  501` → ActivationCenter=501、ActivationWindow=2 → DIA 窗口 [500,502]（与文件名 2Da 吻合）；PrecursorScan=1 链到前一张 MS1。

> 兼容性说明：`ActivationWindow` / `NCE` 是 pXtract 3 新增；WIFF 数据可能缺失（spec 标注"无法提取"）。本版**只保证 Thermo FTMS DIA（属性齐全）**；MS2 缺 `ActivationWindow` → §7 明确报错。

### 3.4 Footer（重复 Scan_Num 次）
`Scan_Idx_Addr: long long[Scan_Num]` —— 每张谱图在文件中的起始偏移。实测 `文件大小 - Addr_List_Addr == Scan_Num*8`（640768 = 80096×8）。本版加载**不依赖** footer（顺序读即可）；仅可选做一次自检（`addr_list[0]==24`）。

### 3.5 RT 单位
实测首/末谱 RT = 0.197 → 7200.175，跨度恰为 120 分钟的秒数 → **PFB 的 RT 已是秒**，与 `DIAData.rt_values`（`_get_retention_time` 转换为秒）一致，**无需换算**。

---

## 4. 架构与组件

### 4.1 新增 `spectrum/pfb_reader.py`（纯解析，无 numpy 数组构建知识）
```python
@dataclass
class PFBSpectrum:
    scan: int
    ms_level: int            # 1 或 2
    rt: float                # 秒
    instrument_type: str
    mz: np.ndarray           # float64
    intensity: np.ndarray    # float64
    # MS2 专有（MS1 为 None）
    charge: int | None = None
    mh_plus: float | None = None
    ion_injection_time: float | None = None
    activation_center: float | None = None
    activation_type: str | None = None
    precursor_scan: int | None = None
    activation_window: float | None = None
    nce: float | None = None
    monoisotopic_mz: float | None = None

def read_header(fh) -> tuple[int, int]:        # (addr_list_addr, scan_num)；跳 3 空 int
def parse_property_str(s: str) -> dict:         # split('\t')，按 token[1] 决定 MS1/MS2 布局
def iter_spectra(fh, scan_num) -> Iterator[PFBSpectrum]   # 顺序读 loop body
def read_footer(fh, addr_list_addr, scan_num) -> list[int]  # 可选，自检用
```
解析器只负责"读字节 → 类型化字段"，**不**做 DIA 窗口推导或 -1 哨兵映射（那是加载器的职责，§4.3）。

### 4.2 `spectrum/dia_data.py`：抽取共享核心（extract-method，纯搬移）
- **`_record_spectrum(self, spectrum_idx, current_peak_index, *, scan_id, rt, precursor_scan_id, isolation_lower, isolation_upper, mz_array, intensity_array) -> (mz_array, intensity_array)`**
  —— 现 `_process_single_spectrum` 尾部（538–561 行）：写 `precursor_scan_ids / rt_values / _scan_id_to_index / _peak_start_idx_list / _peak_stop_idx_list / _precursor_lower_mz / _precursor_upper_mz`。
- **`_finalize_arrays(self, mz_chunks, int_chunks)`**
  —— 现 `_load_from_mzml` 循环后（620–660 行）：concat → float32、min/max mz、`ms1_indexs`(`==-1`)/`ms2_indexs`(`!=-1`) 及其 rt、`frame_max_index`、`_cycle_left_precursor`、centroid-empty 日志（PFB 计数恒 0，共用无副作用）。
- `_load_from_mzml` / `_process_single_spectrum` 改为调用上述共享方法（行为不变，由 §8 测试守护）。质心化、`has_mobility`、`has_ms1` 等**格式相关**逻辑留在各自的解析侧。

### 4.3 新增 `DIAData._load_from_pfb(self, path)`
2-pass，结构对齐 `_load_from_mzml`：
- **Pass 1**：`read_header` 取 `scan_num`；顺序读每谱的 `Property_Str`（用 `Peak_Num` seek 跳过峰数组，不读峰）累计 `total_spectra` 与 `max_scan_id`（= 最大 Scan 号）。
- `_preallocate_arrays(total_spectra=scan_num, max_scan_id=max_scan)`。
- **Pass 2**：`iter_spectra` 逐谱 → 映射统一字段（§4.4）→ `_record_spectrum(...)`，收集 mz/int chunk；`ms_level==1` 时置 `has_ms1=True`。
- `_finalize_arrays(mz_chunks, int_chunks)`。

### 4.4 字段映射（PFBSpectrum → DIAData 统一字段）
| DIAData 统一字段 | PFB 来源 |
|---|---|
| `scan_id` | `scan` |
| `rt`（秒） | `rt`（已是秒，不换算） |
| `precursor_scan_id` | MS1 → **-1**；MS2 → `precursor_scan` |
| `isolation_lower` | MS2：`activation_center - activation_window/2`；MS1：NaN/None |
| `isolation_upper` | MS2：`activation_center + activation_window/2`；MS1：NaN/None |
| `mz_array` / `intensity_array` | `mz` / `intensity`（double → 存为 float32，与 mzML 一致） |

MS1/MS2 的区分完全交给收尾里的 `precursor_scan_ids == -1`，与 mzML 同源、无需额外标志。

### 4.5 `manager/data_manager.py:get_dia_data_object` 分派
```python
ext = os.path.splitext(tot_raw_path)[1].lower()
if ext == ".pfb":
    dia_data._load_from_pfb(tot_raw_path)
else:
    dia_data._load_from_mzml(tot_raw_path)
```
`_centroid_enabled/_threshold` 仍照常注入（PFB 侧不使用，见 §6）。

---

## 5. 数据流与缓存

config `raw_path_*=X.pfb` → `DataManager.get_dia_data_object` →（扩展名）`_load_from_pfb` → PFBReader 2-pass → `_record_spectrum`×N → `_finalize_arrays` → `DIAData` 就绪（与 mzML 等价）。

**缓存**：复用现有 npz 机制——`save_to_file(source_path=X.pfb)` 写入源文件 mtime/size，`validate_cache_params` 据此失效。机制与格式无关，PFB 自动适用。**验证点**：确保缓存编排（`workflows/flow_utils.py` / `DataManager`）传入的 `source_path` 为 `.pfb` 路径而非 mzML。

---

## 6. 质心化决策

PFB（pXtract 导出）已是 peak-picked 峰列表，**跳过** on-load 质心化（再质心化可能误删峰）。即 `_load_from_pfb` 不调用 `centroid_spectrum`，等价于把"已质心化数据"直送数组。`_centroid_enabled` 对 PFB 路径无效（仅 mzML 路径生效）。

---

## 7. 错误处理

| 情形 | 行为 |
|---|---|
| 文件不存在 | 清晰报错（路径） |
| 文件截断 / 读到 EOF 但谱未读完 | 报错并带**谱序号 + 文件偏移** |
| 实读谱数 < header 的 Scan_Num | 报错（数量不符） |
| Property token 数与 MsType 不符 | 报错（期望 4/13，实得 N） |
| **MS2 缺 ActivationWindow** | 明确报错（DIA 需要窗宽；config 默认窗宽留作未来） |
| 空文件 / Scan_Num=0 | 产出空 `DIAData`（与空 mzML 一致） |
| Property_Str 非 UTF-8 | `decode("utf-8")` + `rstrip("\x00")`（解码失败按需 `errors` 处理并报警） |
| 字节序 | 假定小端（struct `<`） |

---

## 8. 测试（TDD）

- **`tests/test_pfb_reader.py`**：测试内**合成** `.pfb`（自写 header + 1 MS1 + 1 MS2 + footer，已知值）→ 校验 `read_header`、`parse_property_str`（MS1/MS2 两种布局）、`iter_spectra` 的 scan/rt/ms_level/各属性/mz/intensity。边界：空文件（Scan_Num=0）、截断文件→报错、MS2 缺 pXtract3 字段、double 往返精度。
- **`tests/test_dia_data_load_pfb.py`**：合成 `.pfb` → `_load_from_pfb` → 断言 `ms1_indexs`/`ms2_indexs`/`rt_values`/`_precursor_lower_mz`/`_precursor_upper_mz`/`_peak_start_idx_list`/`_peak_stop_idx_list`/`_scan_id_to_index`/min/max mz/`_cycle_left_precursor` 与预期一致（仿 `test_dia_data_load_mzml.py` 结构）。
- **真实文件 opt-in 慢测**：仅当 `~/share/.../Rep1.pfb` 存在才跑（`pytest.mark.skipif` + 路径判断），校验 `scan_num==80096`、首张 MS1 RT≈0.197、首张 MS2 窗口==[500,502]。CI 不依赖 813MB 大文件。
- **重构安全网**：抽取 `_record_spectrum`/`_finalize_arrays` 后，现有 `test_dia_data_load_mzml.py`(19) + `test_dia_data_window.py`(9) + cache/centroid 测试须全绿。
- **分派测试**：`get_dia_data_object("x.pfb")` 路由到 `_load_from_pfb`（monkeypatch/spy 验证）；`.mzML` 仍走 `_load_from_mzml`。

---

## 9. 影响面与风险

- **改动文件**：新增 `spectrum/pfb_reader.py`、`tests/test_pfb_reader.py`、`tests/test_dia_data_load_pfb.py`；修改 `spectrum/dia_data.py`（抽取 2 个共享方法 + 新增 `_load_from_pfb`）、`manager/data_manager.py`（分派）。
- **主要风险**：抽取共享方法触碰已审计的 mzML 路径 → 由 28+ 现有测试 + 真实文件等价性兜底。
- **假设**：目标 PFB 为 Thermo FTMS DIA，属性齐全、RT 为秒、小端、intensity 为 double——均已实测确认。
