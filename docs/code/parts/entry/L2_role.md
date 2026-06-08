# entry — 职责与接口

## 一句话职责

程序入口：解析命令行与 `config.ini`、初始化日志、打印 banner，然后构造并运行 `PairFlow` 工作流；附带常量键（`constant/keys.py`）与 banner 输出（`banner/banner.py`）两个支持模块。

## 对外接口

| 符号 | 签名 | 简述 |
|---|---|---|
| `main()` | `main() -> None` | 入口函数：argparse 解析 → configparser 读配置 → 配置 logging → `show_start_banner()` → `PairFlow(...).run()` → `show_end_banner()` |
| `ConfigKeys` | `class`（`metaclass=ConstantsClass`，不可改写） | 访问 config 的字符串常量集合（见下表） |
| `ConfigKeys.GENERAL` / `.WORK_DIRECTORY` | `= "general"` / `"work_directory"` | `[general]` section 名与工作目录 option 名 |
| `ConfigKeys.INPUT` / `.RAW_NUM` / `.RAW_PATH` | `= "input"` / `"raw_num"` / `"raw_path"` | 输入 section 与原始谱图键 |
| `ConfigKeys.SEARCH_ENGINE_TYPE` / `.LIGHT_RESULT_PATH` | `= "search_engine_type"` / `"light_result_file"` | 搜索引擎类型、轻标结果文件键 |
| `show_start_banner()` | `() -> None` | 记录“开始运行”并打印欢迎 banner（读 `./banner/welcome.txt`，失败回退内置文本）|
| `show_end_banner()` | `() -> None` | 记录 `finished` |

## 依赖

- 依赖：`workflows.pair_flow.PairFlow`（实际工作流执行）、`rich.logging.RichHandler`（终端彩色日志）、标准库 `argparse` / `configparser` / `logging` / `os`。
- 被依赖：`main.py` 为顶层入口，不被其他模块导入；`constant/keys.py` 的 `ConfigKeys` 被各处读取 config 的模块共享。

## 输入 / 输出

- 输入：
  - 命令行参数 `--configpath`（默认 `./config.ini`）、`--logpath`（默认 `./ms2.log`）。
  - 配置文件 `config.ini`：含 `[input]`、`[general]` 等 section（见 L3）。
- 输出：日志同时写入终端（rich）与日志文件；banner 文本；副作用为运行 `PairFlow` 工作流。
