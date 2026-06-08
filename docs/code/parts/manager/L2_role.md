# manager — 职责与接口

## 一句话职责

为各类外部数据（原始谱图、搜索结果）提供**统一的加载入口与 pickle 缓存生命周期**：基类 `BaseManager` 负责"从文件读 / 原子写"，各子类负责把具体输入解析成内存对象。

## 对外接口

| 符号 | 签名 | 简述 |
|---|---|---|
| `BaseManager` | `__init__(path=None, load_from_file=True, figure_path=None)` | 缓存基类；构造时按需 `load()` |
| `BaseManager.path` | `@property -> str\|None` | pickle 文件路径（只读） |
| `BaseManager.is_loaded_from_file` | `@property/@setter -> bool` | 是否成功从 pickle 加载 |
| `BaseManager.save()` | `-> None` | 把 `self` 写入 pickle（临时文件 + 原子替换） |
| `BaseManager.load()` | `-> None` | 从 pickle 加载并校验版本，回填 `__dict__` |
| `DataManager` | `__init__(config=None, path=None, load_from_file=False, figure_path=None)` | 原始谱图管理器 |
| `DataManager.get_centroid_params()` | `-> tuple[bool, float]` | 从 config 解析 centroid 参数（单一真相源） |
| `DataManager.get_dia_data_object(tot_raw_path=None)` | `-> DIAData` | 注入 centroid 参数后从 mzML 读取 DIA 数据 |
| `LightResultManager` | `__init__(config=None, path=None, load_from_file=False, figure_path=None)` | 搜索结果（light）管理器 |
| `LightResultManager.get_light_result_object(light_result_path=None)` | `-> LightResult` | 按搜索引擎类型加载搜索结果 |

## 依赖

- 依赖：标准库（`pickle`/`os`/`logging`/`configparser`）；`spectrum.dia_data.DIAData`、`spectrum.light_result.LightResult`；`constant.keys.ConfigKeys`。
- 被依赖：`workflows.pair_flow`（构造两个子类）、`workflows.flow_utils`（用 `DataManager.get_centroid_params` / `get_dia_data_object`）。

## 输入 / 输出

- 输入：`configparser.ConfigParser` 配置、mzML 原始路径、搜索结果路径（pkl / DIA-NN / AlphaDIA / pFind）、可选 pickle 缓存路径。
- 输出：`DIAData`、`LightResult` 内存对象；以及对自身状态的 pickle 持久化（`save`/`load`）。
