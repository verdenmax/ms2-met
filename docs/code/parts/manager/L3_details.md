# manager — 细节

`manager` 子系统把"如何加载/缓存数据"与"数据本身的解析逻辑"解耦：基类管缓存生命周期，子类管解析。

## BaseManager 提供的通用能力

- **状态变量**：`_path`（pickle 路径）、`_is_loaded_from_file`、`figure_path`、`_version`（取 `manager.__version__ = "0.0.1-dev"`）。
- **构造即加载**：`__init__` 中若 `load_from_file=True` 则调用 `load()`。
- **原子写（save）**：`path` 为 `None` 时直接返回；否则写入 `path + ".tmp"` 并 `flush()`，再 `os.replace()` 原子替换，避免写一半导致缓存损坏；异常时清理临时文件并 `logging.error`。
- **带版本校验的读（load）**：`path` 为 `None` 或文件不存在 → 记 info 日志并返回（走初始化路径）；存在则 `pickle.load`，仅当 `loaded_state._version == self._version` 时用 `self.__dict__.update(...)` 回填并置 `is_loaded_from_file=True`，版本不符记 warning，异常记 exception。

## 各子类职责

### DataManager（原始谱图）
- 持有 `config` 与 `stats`（`stats` 在 `super().__init__` 之前初始化，因为加载可能覆盖 `__dict__`）。
- `get_centroid_params()`：从 `[General]` 段读 `centroid_enabled`/`centroid_rel_threshold`，缺段或缺键回落到 `DIAData` 模块级默认（`DEFAULT_CENTROID_ENABLED`/`DEFAULT_CENTROID_REL_THRESHOLD`）。作为 **centroid 参数的单一真相源**，同时供 `get_dia_data_object` 与 `workflows.flow_utils.data_to_npz` 校验 npz 缓存使用（消除 P0-3 审计指出的重复，2026-06-03）。
- `get_dia_data_object()`：新建 `DIAData`，注入 centroid 参数后**按文件扩展名分派**：`os.path.splitext(tot_raw_path or "")[1].lower() == ".pfb"` → `_load_from_pfb(tot_raw_path)`，否则 → `_load_from_mzml(tot_raw_path)`（`tot_raw_path` 为 `None` 时走 mzML 分支）。PFB 已 peak-picked，centroid 参数对其无实际作用。

### LightResultManager（搜索结果）
- 结构与 `DataManager` 一致（`stats` 前置、持有 `config`）。
- `get_light_result_object()`：读 `[Input]` 段 `search_engine_type`（fallback=1）分派：
  - `0` → `_load_from_pkl`
  - `1` → DIA-NN，`_load_from_dia_nn_input`
  - `2` → AlphaDIA，`_load_from_alphadia_input`
  - `3` → pFind，`_load_from_pfind_input`
  - 其它 → **抛 `ValueError`**（仅支持 0/1/2/3）。
  - 类型 1/2/3 额外读 `pfind_qvalue_threshold`（fallback=0.01）作为 `qvalue_threshold`。

## 设计取舍：为何要 manager 抽象

- **缓存与解析分离**：所有"读文件、写缓存、版本校验"集中在 `BaseManager`，子类只关心把输入变成 `DIAData` / `LightResult`，避免每个数据源重复实现持久化。
- **原子写 + 版本校验**：保证缓存文件不被半写损坏，且跨版本不误用旧 pickle。
- **配置驱动**：子类不写死参数，统一从 `ConfigParser` 取值并提供 fallback，便于 workflows 复用同一真相源。

## 边界 / 坑

- `save()`/`load()` 在 `path=None` 时是**静默 no-op**（仅记日志），适合"无缓存即初始化"的场景。
- 版本不匹配时 `load` **不抛异常**、不回填，对象保持构造时的空状态。
- `stats = {}` 必须在 `super().__init__` 之前赋值，否则 `load()` 回填 `__dict__` 时可能被覆盖/缺失。
- `DataManager` 当前并不缓存 `DIAData` 实例本身，每次 `get_dia_data_object` 都新建并按扩展名重新读原始文件（mzML 或 PFB）。
