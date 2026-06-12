# manager — API 参考

## manager/__init__.py

模块级版本号：`__version__ = "0.0.1-dev"`，被 `BaseManager` 用于 pickle 版本校验。

## manager/base_manager.py

### `class BaseManager`

#### `__init__(self, path: None | str = None, load_from_file: bool = True, figure_path: None | str = None)`
保存 `path`、`figure_path`、`_version`，初始化 `is_loaded_from_file=False`；当 `load_from_file=True` 时调用 `self.load()`。

#### `path` → `str | None`（@property，只读）
pickle 文件路径。

#### `is_loaded_from_file` → `bool`（@property + @setter）
是否成功从 pickle 加载。

#### `save(self) -> None`
把 `self` 序列化到 pickle。`path` 为 `None` 时直接返回。写入 `path + ".tmp"` 并 `flush()` 后 `os.replace()` 原子替换；异常时删除临时文件并 `logging.error`。

#### `load(self) -> None`
从 pickle 加载。`path` 为 `None` 或文件不存在 → 记 info 日志并返回。否则 `pickle.load`，仅当 `_version` 一致时用 `__dict__.update` 回填并置 `is_loaded_from_file=True`；版本不一致记 warning；异常记 exception。

## manager/data_manager.py

### `class DataManager(base_manager.BaseManager)`

#### `__init__(self, config: None | configparser.ConfigParser = None, path: None | str = None, load_from_file: bool = False, figure_path: None | str = None)`
先置 `self.stats = {}`（须早于 `super().__init__`），调用基类构造，再保存 `self._config`。

#### `get_centroid_params(self) -> tuple[bool, float]`
返回 `(enabled, threshold)`。`_config` 为 `None` 或无 `[General]` 段时返回模块默认 `(DEFAULT_CENTROID_ENABLED, DEFAULT_CENTROID_REL_THRESHOLD)`；否则从 `General.centroid_enabled` / `General.centroid_rel_threshold` 读取（各自 fallback 到默认）。centroid 参数单一真相源。

#### `get_dia_data_object(self, tot_raw_path: None | str = None) -> DIAData`
新建 `DIAData`，将 `get_centroid_params()` 注入其 `_centroid_enabled` / `_centroid_rel_threshold`，再**按扩展名分派**加载：`.pfb`（大小写不敏感）→ `_load_from_pfb`，否则 → `_load_from_mzml`（`tot_raw_path` 为 None 也走 mzML 分支）。返回该对象。PFB 路径忽略 centroid 参数（已 peak-picked）。

## manager/light_result_manager.py

### `class LightResultManager(BaseManager)`

#### `__init__(self, config: None | configparser.ConfigParser = None, path: None | str = None, load_from_file: bool = False, figure_path: None | str = None)`
先置 `self.stats = {}`（须早于 `super().__init__`），调用基类构造，再保存 `self._config`。

#### `get_light_result_object(self, light_result_path: None | str = None) -> LightResult`
新建 `LightResult`，按 `[Input].search_engine_type`（fallback=1）分派加载：

| `search_engine_type` | 加载方法 | 额外参数 |
|---|---|---|
| `0` | `_load_from_pkl(path)` | — |
| `1` | `_load_from_dia_nn_input(path, qvalue_threshold=...)` | `Input.pfind_qvalue_threshold`（fallback=0.01） |
| `2` | `_load_from_alphadia_input(path, qvalue_threshold=...)` | 同上 |
| `3` | `_load_from_pfind_input(path, qvalue_threshold=...)` | 同上 |
| 其它 | 记 `logging.error`（支持 0/1/2/3） | — |

返回 `LightResult` 对象。
