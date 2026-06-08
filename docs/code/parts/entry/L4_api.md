# entry — API 参考

## `main.py`

程序入口模块。

### `main() -> None`

唯一函数，亦为 `if __name__ == "__main__"` 调用目标。流程：

1. `argparse` 解析 `--configpath`（默认 `./config.ini`）、`--logpath`（默认 `./ms2.log`）。
2. `configparser.ConfigParser().read(configpath)`：空返回 → `FileNotFoundError`；`sections()` 为空 → `ValueError`。
3. 必要时 `os.makedirs(log_dir, exist_ok=True)`。
4. `logging.basicConfig(level=INFO, handlers=[RichHandler(), FileHandler(logpath, encoding="utf-8")])`，文件格式 `%(asctime)s | %(levelname)s | %(name)s | %(message)s`。
5. `banner.show_start_banner()`。
6. `work_path = config.get("general", "work_directory", fallback="./workspace")`，捕获 `configparser.NoSectionError` 回退 `./workspace`。
7. `PairFlow(workname="main", config=config, work_path=work_path).run()`。
8. `banner.show_end_banner()`。

导入：`argparse`、`configparser`、`os`、`logging`、`rich.logging.RichHandler`、`banner.banner`、`workflows.pair_flow.PairFlow`。

---

## `constant/keys.py`

共享常量键模块。

### `class ConstantsClass(type)`

常量元类。

- `__setattr__(self, name, value)`：抛 `TypeError("Constants class cannot be modified")`，禁止改写常量。
- `get_values(cls) -> list[str]`：返回类内所有用户定义（非 `__` 开头）且为 `str` 的值列表。

### `class ConfigKeys(metaclass=ConstantsClass)`

config 访问用字符串常量。

| 常量 | 值 | 说明 |
|---|---|---|
| `INPUT` | `"input"` | 输入 section |
| `RAW_NUM` | `"raw_num"` | 原始谱图数量 |
| `RAW_PATH` | `"raw_path"` | 原始谱图路径前缀 |
| `LIGHT_RESULT_PATH` | `"light_result_file"` | 轻标结果文件 |
| `SEARCH_ENGINE_TYPE` | `"search_engine_type"` | 搜索引擎类型 |
| `PFIND_QVALUE_THRESHOLD` | `"pfind_qvalue_threshold"` | pfind FDR 阈值（pfind 特有）|
| `GENERAL` | `"general"` | 通用 section |
| `WORK_DIRECTORY` | `"work_directory"` | 工作目录 |
| `MASS_TOL_PPM` | `"mass_tol_ppm"` | 质量容差（ppm）|
| `XIC_CYCLE_WINDOW` | `"xic_cycle_window"` | XIC 周期窗口 |
| `RESULT_FILE` | `"result_file"` | 输出结果文件 |
| `FEATURE_TYPE` | `"feature_type"` | 特征生成模式 |
| `RANDOM_SEED` | `"random_seed"` | 随机种子 |
| `CENTROID_ENABLED` | `"centroid_enabled"` | mzML centroiding 开关（`manager/data_manager.py` 读取）|
| `CENTROID_REL_THRESHOLD` | `"centroid_rel_threshold"` | centroid 相对强度阈值（同上）|

---

## `banner/banner.py`

启动/结束 banner 输出模块。

### `show_start_banner() -> None`

`logging.info("开始运行")`；尝试读取 `./banner/welcome.txt`（UTF-8）并 `logging.info` 其内容；任何异常时回退 `print` 内置 ANSI 彩色文本（`ms2 check v0.0.1`）。

### `show_end_banner() -> None`

`logging.info("finished")`。
