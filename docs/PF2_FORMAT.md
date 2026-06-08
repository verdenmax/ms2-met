# pFind `.pf2` / `.pf1` 二进制格式说明（含 `.idx` / `.idxL` 索引）

本文档描述 pFind 工具链中实际使用的谱图二进制文件 (`.pf2` / `.pf1`)
以及配套的旧/新两种索引文件 (`.pf2idx` / `.pf2idxL` / `.pf1idx` / `.pf1idxL`)。

> 适用范围：pParse 输出、pBuild / pFind 读取的所有 `pf` 系列文件。

---

## 1. 全局约定

| 项目             | 取值                                                     |
|------------------|----------------------------------------------------------|
| 字节序           | **小端序 (Little-Endian)**                               |
| 对齐 / padding   | **紧凑无填充** (C ABI tightly-packed)                    |
| 文件头 / 尾部    | **均无**（无 magic、无版本号、无校验、无脚注）           |
| 终止条件         | 索引文件读到 EOF 即停止（pBuild 用异常捕获实现）         |
| 整数类型         | `int32` / `uint32` / `uint64`，C# `BinaryReader` 顺序读  |
| 浮点类型         | IEEE-754 `double` (`f64`)                                |

---

## 2. 文件配对关系

每个 `raw` 文件经 pParse 处理后生成一对（或两对）文件：

```
xxx.raw
 ├── xxx.pf2          ← MS2 谱图数据（顺序拼接所有 MS2 记录）
 ├── xxx.pf2idx       ← MS2 索引（旧，32 位 offset，8 B / 条）
 ├── xxx.pf2idxL      ← MS2 索引（新，64 位 offset，12 B / 条）
 ├── xxx.pf1          ← MS1 谱图数据
 ├── xxx.pf1idx       ← MS1 索引（旧）
 └── xxx.pf1idxL      ← MS1 索引（新）
```

`.pf2idxL` / `.pf1idxL` 是为突破 4 GiB 数据文件上限而引入的扩展索引；
数据文件 `.pf2` / `.pf1` **本身格式不变**，仅索引升级。

路径推断逻辑见 `Bean/Task.cs:261-294`：
取 `.pf2` 路径去后缀 + `".pf2idx"` 即为索引路径。

---

## 3. 谱图数据文件 `.pf2` / `.pf1` 记录格式

`.pf2` / `.pf1` 是**多条变长记录顺序拼接**，无任何分隔符。
解析单条记录需事先知道其起始 offset（由索引提供）。

对应代码：`MainWindow.xaml.cs:1397-1444 read_peaks(...)`。

### 3.1 单条记录字段表

| 字段                  | 类型      | 字节数        | 说明                                       |
|-----------------------|-----------|---------------|--------------------------------------------|
| `scan_num`            | `i32` LE  | 4             | 扫描编号，可与索引中的 key 比对做自校验    |
| `peak_num` = N        | `i32` LE  | 4             | 谱峰数量                                   |
| `peaks[0..N-1]`       | —         | 16 × N        | N 组谱峰，每组：                           |
| &nbsp;&nbsp;`mz`      | `f64` LE  | 8             | 谱峰 m/z                                   |
| &nbsp;&nbsp;`intensity` | `f64` LE | 8            | 谱峰绝对强度                               |
| `pre_all_num` = M     | `i32` LE  | 4             | 母离子候选数（MS1 = 0；MS2 ≥ 1）           |
| `precursors[0..M-1]`  | —         | 12 × M        | M 组母离子信息，每组：                     |
| &nbsp;&nbsp;`pep_mass`| `f64` LE  | 8             | 母离子单同位素质量 / m/z                   |
| &nbsp;&nbsp;`charge`  | `i32` LE  | 4             | 母离子电荷                                 |

**单条记录总长**：

```
len(record) = 4 + 4 + 16 * peak_num + 4 + 12 * pre_all_num
            = 12 + 16 * peak_num + 12 * pre_all_num
```

### 3.2 字节布局示意

```
offset  0   4   8         24       8+16N   12+16N
        ┌───┬───┬─────────┬───╌╌╌──┬───────┬─────────┬───╌╌╌─┐
        │SN │PN │ mz[0]   │inten[0]│  ...  │ pre_all │  pre  │
        │i32│i32│  f64    │  f64   │ × N-1 │   i32   │ × M   │
        └───┴───┴─────────┴────────┴───────┴─────────┴───────┘
                                              │
                                              ▼
                                   每组: pep_mass(f64) + charge(i32) = 12 B
```

### 3.3 MS1 vs MS2 的差异

| 维度          | `.pf1` (MS1)                        | `.pf2` (MS2)                              |
|---------------|--------------------------------------|-------------------------------------------|
| 谱峰区        | 仍是 `(mz, intensity) × peak_num`   | 同左                                      |
| `pre_all_num` | 通常 = 0                            | ≥ 1（被选中碎裂的母离子候选）             |
| 后置母离子段  | 无（M=0 即没有）                    | 长 12·M 字节                              |

> 因此 MS1/MS2 共用同一种 record schema，结构上的唯一区别就是
> `pre_all_num` 是否为 0 以及后续 12·M 字节是否存在。

### 3.4 `read_peaks` 的读取语义（关键细节）

```csharp
// 仅读到“第 pre_num 个母离子”为止，覆盖式赋值，前面的被丢弃
for (int i = 0; i <= pre_num; ++i) {
    pep_mass = br.ReadDouble();
    charge   = br.ReadInt32();
}
// 若 pre_num >= pre_all_num 直接返回 0.0
```

- `pre_num` 越界（`>= pre_all_num`）会被判定为"该母离子不存在"，返回 0。
- 谱峰强度在返回前会被**整体归一化到最大值 = 100**：
  `peaks[i].Intensity = 100 * intensity / max_inten`。

### 3.5 `read_peaks2` 的用法

`MainWindow.xaml.cs:1352-1390`：给定一张 MS1 scan 号，沿 scan 号
向后**连续命中索引**的所有 MS2 谱图（即由该 MS1 触发的子谱），
收集每张 MS2 的 `pep_mass` 用于"哪些 MS1 谱峰被碎裂"的可视化。

---

## 4. 索引文件 — 旧格式 `.pf2idx` / `.pf1idx`

对应代码：`Tools/File_Help.cs:552-601 read_ms2_index / read_ms1_index`。

### 4.1 记录布局（8 字节 / 条）

| 偏移 | 字段       | 类型     | 字节数 | 说明                                |
|------|------------|----------|--------|-------------------------------------|
| 0    | `scan_num` | `i32` LE | 4      | 谱图扫描号                          |
| 4    | `offset`   | `i32` LE | 4      | 该谱图在 `.pf2`/`.pf1` 中的字节偏移 |

### 4.2 文件整体

```
+----------+----------+----------+--- ... ---+----------+
| entry 0  | entry 1  | entry 2  |           | entry N-1|
+----------+----------+----------+--- ... ---+----------+
file_size == 8 × N
```

### 4.3 局限性

- pBuild 使用 `BinaryReader.ReadInt32()` 读 offset，类型为**有符号 32 位**。
- 实际可寻址范围：**0 ~ 2 GiB - 1**（语言层面）；
  按无符号解释也只能到 **4 GiB - 1**。
- 一旦 `.pf2` / `.pf1` 超过这个上限，旧索引无法正确指位 → 需 `.idxL`。

---

## 5. 索引文件 — 新格式 `.pf2idxL` / `.pf1idxL`

由 `conver_idx_to_idxL/main.go` 工具从旧索引转换而来；
亦可由新版 pParse 直接生成。文件名末尾的 `L` = **Long (64-bit) offset**。

### 5.1 记录布局（12 字节 / 条）

| 偏移 | 字段       | 类型     | 字节数 | 说明                                       |
|------|------------|----------|--------|--------------------------------------------|
| 0    | `scan_num` | `u32` LE | 4      | 与旧格式语义完全一致                       |
| 4    | `offset`   | `u64` LE | 8      | 64 位无符号偏移，理论上限 16 EiB           |

### 5.2 文件整体

```
+------------+------------+------------+--- ... ---+------------+
|  entry 0   |  entry 1   |  entry 2   |           | entry N-1  |
+------------+------------+------------+--- ... ---+------------+
file_size == 12 × N
```

### 5.3 与旧格式的数值对应

由 `.pf2idx` 转换出的 `.pf2idxL`：

- `scan_num` 原样拷贝；
- `offset` 由 `u32` **零扩展 (zero-extend)** 为 `u64`，高 4 字节恒为 `0x00`，数值不变；
- 记录数 N 不变，文件体积变为原来的 **1.5×**。

---

## 6. 三类文件的尺寸关系

记某 raw 的谱图数为 `N`：

| 文件          | 大小                                       |
|---------------|--------------------------------------------|
| `.pf2idx`     | `8 × N`                                    |
| `.pf2idxL`    | `12 × N`                                   |
| `.pf2`        | `Σ (12 + 16·peak_num_i + 12·pre_all_num_i)`|

`.pf2` 体积取决于每张谱图的峰数与母离子候选数，不能由 N 直接推出。

---

## 7. 读取流程（伪代码）

### 7.1 装载索引

```text
// 使用 idxL（推荐）
打开 file.pf2idxL
hash = {}
while not EOF:
    scan_num = ReadU32()
    offset   = ReadU64()
    hash[scan_num] = offset

// 使用旧 idx
打开 file.pf2idx
while not EOF:
    scan_num = ReadU32()
    offset   = ReadU32()        // 注意 32 位上限
    hash[scan_num] = offset
```

### 7.2 按需读取单张谱图

```text
offset = hash[scan_num]
打开 file.pf2:
    Seek(offset)
    s_num       = ReadI32()        // == scan_num（自校验）
    peak_num    = ReadI32()
    peaks = []
    for i in 0..peak_num:
        mz    = ReadF64()
        inten = ReadF64()
        peaks.append((mz, inten))
    pre_all_num = ReadI32()
    precursors = []
    for i in 0..pre_all_num:
        pep_mass = ReadF64()
        charge   = ReadI32()
        precursors.append((pep_mass, charge))
```

### 7.3 Go 参考片段（读 `.pf2idxL`）

```go
type Entry struct {
    Scan   uint32
    Offset uint64
}

func ReadIdxL(path string) ([]Entry, error) {
    f, err := os.Open(path)
    if err != nil { return nil, err }
    defer f.Close()
    var out []Entry
    var e Entry
    for {
        if err := binary.Read(f, binary.LittleEndian, &e.Scan); err != nil {
            if errors.Is(err, io.EOF) { return out, nil }
            return out, err
        }
        if err := binary.Read(f, binary.LittleEndian, &e.Offset); err != nil {
            return out, err
        }
        out = append(out, e)
    }
}
```

### 7.4 C# 参考片段（读 `.pf2idxL`）

```csharp
public static Dictionary<int, long> ReadIdxL(string path) {
    var hash = new Dictionary<int, long>();
    using var br = new BinaryReader(File.OpenRead(path));
    try {
        while (true) {
            int  scan   = br.ReadInt32();      // 4B
            long offset = br.ReadInt64();      // 8B
            hash[scan] = offset;
        }
    } catch (EndOfStreamException) { }
    return hash;
}
```
