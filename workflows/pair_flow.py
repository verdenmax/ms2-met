import configparser
import copy
import logging
import os
import pandas as pd
from collections import defaultdict
from itertools import combinations


from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from rich.progress import Progress

import workflows.flow_utils as flow_utils
import manager.data_manager as data_manager

from workflows.flow_utils import data_to_npz
from workflows.flow_utils import process_batch_pair_shuffle
from workflows.flow_utils import process_batch_single, process_batch_pair
from workflows.single_work import multi_batch_work
from manager.light_result_manager import LightResultManager
from spectrum.psm_info import PSMInfo

from constant.keys import ConfigKeys

BATCH_SIZE = 5000


class PairFlow:
    """
    轻重标匹配的工作流
    这个工作流将会完成
    1. 质谱数据文件读取，通过 data_manager
    2. 搜索结果文件读取，需要支持多种格式
    3. 取出每一条搜索结果，将其在谱图中找到对应的轻重标配对
    4. 得到通过检验的结果，存储到文件中
    """

    RAW_DATA_MANAGER_PICKLE = "raw_data_manager.pkl"
    LIGHT_RESULT_MANAGER_PUCKEL = "light_result_manager.pkl"

    def __init__(
        self,
        workname: str,
        config: None | configparser.ConfigParser = None,
        work_path: str = "./Pairworkspace",
    ) -> None:
        """
        初始化工作流
        """
        self.workname: str = workname

        self._config: configparser.ConfigParser = config

        self._workpath: str = work_path

        self._nametoDATA = {}

        # 创建不存在的目录
        for path in [self._workpath]:
            if path and not os.path.exists(path):
                logging.info(f"Creating folder {path}")

                os.makedirs(
                    path,
                    exist_ok=True,
                )

    def load(
        self
    ) -> None:
        # 读取light result
        self._light_result_manager = LightResultManager(
            self._config,
            path=os.path.join(
                self._workpath, self.LIGHT_RESULT_MANAGER_PUCKEL),
        )

        light_result_path = (
            self._config[ConfigKeys.INPUT][ConfigKeys.LIGHT_RESULT_PATH])
        self._light_result = self._light_result_manager.get_light_result_object(
            light_result_path)

        # 从配置文件中加载所需信息
        self._raw_file_manager = data_manager.DataManager(
            self._config,
            path=os.path.join(self._workpath, self.RAW_DATA_MANAGER_PICKLE),
        )
        self._raw_file_manager.save()

    def multi_handle(
        self,
        psm1: PSMInfo,
        psm2: PSMInfo,
        label: int,
    ):
        """ 进行多线程地处理 """

        # 获取 dia 数据，当之后想要多进程读数据时，可以直接将 multi_handle 多进程即可_
        # dia_data = self._raw_file_manager.get_dia_data_object(raw_file_path)
        light_dia_data = self._nametoDATA[psm1._raw_title]
        heavy_dia_data = self._nametoDATA[psm2._raw_title]

        # TODO: 计算出信息
        tot_features = multi_batch_work(
            psm1=psm1,
            dia_data1=light_dia_data,
            psm2=psm2,
            dia_data2=heavy_dia_data,
            config=self._config,
        )

        return {
            "sequence": psm1._sequence,
            "charge": psm1._charge,
            "precursor_mz": psm1._precursor_mz,
            "raw_title1": psm1._raw_title,
            "raw_title2": psm2._raw_title,
            "protein_names": psm1._protein_names,
            "sequence_len": len(psm1._sequence),
            "label": label,
            ** tot_features
        }

    def pharse_data(self, tot_raw_path: str):
        """ 从 manager 中解析出数据"""
        tot_raw_file_name = flow_utils.get_filename_stem(tot_raw_path)

        tot_dia_data = self._raw_file_manager.get_dia_data_object(
            tot_raw_path)

        return tot_raw_file_name, tot_dia_data

    def _process_group(self, group):
        ans = []

        # 这个其实是在重复样本中找到重复出现的
        for a, b in combinations(group, 2):
            # label 设置为 1,记为正样本
            res = self.multi_handle(a, b, 1)
            ans.append(res)

        # DEPRECATED: 以下 in-silico 负样本生成（重复样本两两组合 + heavy
        # RT +10 人为错位）已弃用。现行流程直接使用陷阱库(entrapment)作为
        # 负例并提取特征（单流程 feature_type 0）。此路径仅为兼容旧配置保留。
        # 在重复样本中，找到负样本
        for a, b in combinations(group, 2):
            # label 设置为 0,记为负样本
            # Bug #3: 'b' is shared across multiple combinations(group, 2)
            # iterations; mutating b._rt += 10 in place caused offsets to
            # accumulate (10, 20, 30...) across pairs. Shift a copy instead.
            b_shifted = copy.copy(b)
            b_shifted._rt = b._rt + 10
            res = self.multi_handle(a, b_shifted, 0)
            ans.append(res)

        return ans

    @staticmethod
    def _resolve_raw_paths(config, raw_file_nums):
        """读取 raw_path_1..N；缺失时给出清晰错误（而非裸 KeyError/NoSectionError）。"""
        paths = []
        for i in range(raw_file_nums):
            key = f"{ConfigKeys.RAW_PATH}_{i + 1}"
            if not config.has_option(ConfigKeys.INPUT, key):
                raise ValueError(
                    f"配置缺少 {key}（raw_num={raw_file_nums}，请检查 [input] 下 "
                    f"raw_path_* 的数量是否与 raw_num 匹配）")
            paths.append(config[ConfigKeys.INPUT][key])
        return paths

    @staticmethod
    def _build_raw_tasks(psm_groups, name_to_shared, feature_type,
                         pred_store=None):
        """构建任务列表；raw_title 不在配置 raw 中则跳过计数。pred_store 非空
        （speclib 开启）时给 feature_type=0 的每个任务 dict 附 pred_frags
        （命中=预测碎片 dict，未命中=None）。返回 (tasks, n_skipped)。"""
        from workflows.pred_store import normalize_key
        tasks = []
        n_skipped = 0
        if feature_type == 0:
            for group in psm_groups.values():
                for a in group:
                    if a._raw_title not in name_to_shared:
                        n_skipped += 1
                        continue
                    d = a.to_dict()
                    if pred_store is not None:
                        rec = pred_store.get(
                            normalize_key(a._sequence, a._modify, a._charge))
                        d["pred_frags"] = rec["frags"] if rec is not None else None
                    tasks.append((d, name_to_shared[a._raw_title]))
        else:  # feature_type 1 或 2 (Phase 2b 再附 pred_frags)
            for group in psm_groups.values():
                for a, b in combinations(group, 2):
                    if (a._raw_title not in name_to_shared
                            or b._raw_title not in name_to_shared):
                        n_skipped += 1
                        continue
                    shared_a = name_to_shared[a._raw_title]
                    shared_b = name_to_shared[b._raw_title]
                    tasks.append((a.to_dict(), b.to_dict(), shared_a, shared_b, 1))
                    tasks.append((a.to_dict(), b.to_dict(), shared_a, shared_b, 0))
        return tasks, n_skipped

    def _build_pred_store(self):
        """[speclib] speclib_dir 配置了 → 一遍流式扫库建 PredStore；否则 None。"""
        if not self._config.has_option(ConfigKeys.SPECLIB, ConfigKeys.SPECLIB_DIR):
            return None
        speclib_dir = self._config[ConfigKeys.SPECLIB][ConfigKeys.SPECLIB_DIR].strip()
        if not speclib_dir:
            return None
        from spectrum.speclib import SpecLib
        from workflows.pred_store import build_pred_store, normalize_key
        fasta = self._config[ConfigKeys.SPECLIB][ConfigKeys.SPECLIB_FASTA]
        mod = self._config[ConfigKeys.SPECLIB][ConfigKeys.SPECLIB_MOD]
        lib = SpecLib.open_dir(speclib_dir, fasta_path=fasta, mod_path=mod)
        wanted = {normalize_key(p._sequence, p._modify, p._charge)
                  for p in self._light_result.psm_info}
        logging.info("speclib: 扫库建 PredStore（wanted=%d）...", len(wanted))
        store = build_pred_store(lib, wanted)
        logging.info("speclib: PredStore hit=%d miss=%d", store.n_hit, store.n_miss)
        return store

    def distribute(self):
        # 处理每一个任务
        # 对于每一个文件，需要传递给他一个质谱数据、一个输入数据、config
        raw_file_nums = self._config.getint(
            ConfigKeys.INPUT, ConfigKeys.RAW_NUM, fallback=1)

        # 进行多进程
        with ProcessPoolExecutor(
                max_workers=min(raw_file_nums, 25)) as executor:

            futures = []

            # 记录所有 shared 文件对应信息，就是 name 对应的 shared_path
            shared_files = []
            raw_paths = self._resolve_raw_paths(self._config, raw_file_nums)
            for tot_raw_path in raw_paths:
                # 分发任务，去分配不同的进程进行读取 mz 数据
                futures.append(executor.submit(
                    data_to_npz,
                    self._raw_file_manager, tot_raw_path, self._workpath))

            for future in as_completed(futures):
                tot_raw_file_name, shared_path = future.result()
                shared_files.append((tot_raw_file_name, shared_path))

            name_to_shared = {name: path for name, path in shared_files}

        logging.info("开始psm 任务分配")
        # NOTE: 处理信息，让 psm 信息两两分组
        psm_groups = defaultdict(list)

        # 对每一个 psm ，都映射他们的 key 到一个相同的 groups
        for psm in self._light_result.psm_info:
            key = PSMInfo.get_key(psm)
            psm_groups[key].append(psm)

        tasks = []

        # 生成特征的模式,详细定义看 config 文件
        feature_type = self._config.getint(
            ConfigKeys.GENERAL, ConfigKeys.FEATURE_TYPE, fallback=0)

        # raw_title 不在配置 raw 文件中的 PSM 跳过并计数（而非裸 KeyError 中断）
        pred_store = self._build_pred_store()   # None when speclib disabled
        tasks, n_skipped_unknown_raw = self._build_raw_tasks(
            psm_groups, name_to_shared, feature_type, pred_store=pred_store)
        if n_skipped_unknown_raw:
            logging.warning(
                f"跳过 {n_skipped_unknown_raw} 个 PSM/对：raw_title 不在配置的 "
                f"raw 文件中（检查 [input] raw_path_* 与结果中的 Run/RawName 一致性）")

        if feature_type == 0:
            # 单文件：按 shared1 分组
            task_groups = defaultdict(list)
            for psm_dict, shared1 in tasks:
                task_groups[shared1].append((psm_dict,))
        elif feature_type == 1 or feature_type == 2:
            # 双文件：按 (shared1, shared2) 分组（注意顺序？若无序可用 frozenset）
            task_groups = defaultdict(list)
            for psm1_dict, psm2_dict, shared1, shared2, label in tasks:
                key = (shared1, shared2)
                task_groups[key].append((psm1_dict, psm2_dict, label))

        # 进行计算
        ans = []
        pool_workers = min(os.cpu_count() or 4, 25)
        logging.info(f"启动进程池 workers={pool_workers}, batch_size={BATCH_SIZE}")

        with ProcessPoolExecutor(
                max_workers=pool_workers,
                max_tasks_per_child=4) as executor:

            multi_sample_futures = {}

            if feature_type == 0:
                for shared_path, batch_tasks in task_groups.items():
                    # 切分成每批最多 BATCH_SIZE 个
                    for i in range(0, len(batch_tasks), BATCH_SIZE):
                        chunk = batch_tasks[i:i + BATCH_SIZE]
                        fut = executor.submit(
                            process_batch_single,
                            shared_path, chunk, self._config)
                        multi_sample_futures[fut] = len(chunk)
            elif feature_type == 1:
                for (shared1, shared2), batch_tasks in task_groups.items():
                    for i in range(0, len(batch_tasks), BATCH_SIZE):
                        chunk = batch_tasks[i:i + BATCH_SIZE]
                        fut = executor.submit(
                            process_batch_pair,
                            shared1, shared2,
                            chunk, self._config)
                        multi_sample_futures[fut] = len(chunk)
            elif feature_type == 2:
                for (shared1, shared2), batch_tasks in task_groups.items():
                    for i in range(0, len(batch_tasks), BATCH_SIZE):
                        chunk = batch_tasks[i:i + BATCH_SIZE]
                        fut = executor.submit(
                            process_batch_pair_shuffle,
                            shared1, shared2,
                            chunk, self._config)
                        multi_sample_futures[fut] = len(chunk)

            with Progress() as progress:
                rich_task_progress = progress.add_task(
                    "[cyan] 处理进度 ...", total=len(multi_sample_futures))

                # 收集结果（as_completed 可尽早获取已完成任务）；整批失败按
                # chunk_size 计入丢失，避免静默丢数据（见 _collect_batch_results）
                ans, n_total_errors, n_total_attempted, pool_broken = (
                    self._collect_batch_results(
                        multi_sample_futures, progress, rich_task_progress))

        # 汇总错误率
        if n_total_attempted > 0:
            error_rate = n_total_errors / n_total_attempted
            if error_rate > 0.01:
                logging.warning(
                    f"批次错误率过高: {n_total_errors}/{n_total_attempted} "
                    f"({error_rate:.1%}) — 检查日志中的 PSM 处理失败明细"
                )
            else:
                logging.info(
                    f"批次完成: {len(ans)}/{n_total_attempted} 成功 "
                    f"({n_total_errors} 错误)"
                )

        # NOTE: 保存结果
        ans_df = pd.DataFrame(ans)

        result_file = self._config.get(
            ConfigKeys.GENERAL, ConfigKeys.RESULT_FILE,
            fallback="result.csv")

        result_dir = os.path.dirname(result_file)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)

        logging.info(f"保存结果文件 {result_file}")
        ans_df.to_csv(result_file, sep=',', index=False)

        # Bug #14/H2: 进程池崩溃时写一个 sidecar 标记，
        # 让下游工具能检测到 CSV 是不完整的部分结果
        if pool_broken:
            marker = result_file + ".PARTIAL_INCOMPLETE"
            try:
                with open(marker, "w") as f:
                    f.write(
                        "WARNING: distribute() exited early due to "
                        "BrokenProcessPool.\n"
                        "This CSV contains only the batches that completed "
                        "before the crash.\n"
                        f"completed_psms={len(ans)}\n"
                        f"attempted_psms={n_total_attempted}\n"
                        f"errors_within_completed_batches={n_total_errors}\n"
                    )
                logging.error(
                    f"!!! 结果不完整 — sidecar 标记已写入 {marker} "
                    f"(完成 {len(ans)} / 尝试 {n_total_attempted} PSM)"
                )
            except Exception as e:
                logging.error(
                    f"无法写入 PARTIAL_INCOMPLETE 标记 {marker}: {e}")

    @staticmethod
    def _collect_batch_results(future_to_size, progress=None, task_id=None):
        """收集各批次 future 结果并统计成功/丢失数。

        关键修复：整批 future 抛异常时，丢失的是整个 chunk（最多 BATCH_SIZE 个
        PSM），必须把 chunk_size 同时计入 n_total_errors 与 n_total_attempted，
        否则错误率分母漏掉这些 PSM、警告永不触发 → 静默丢数据。

        返回 (ans, n_total_errors, n_total_attempted, pool_broken)。
        """
        ans = []
        n_total_errors = 0
        n_total_attempted = 0
        pool_broken = False
        for future in as_completed(future_to_size):
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1)
            chunk_size = future_to_size[future]
            try:
                result = future.result()
                if isinstance(result, tuple) and len(result) == 2:
                    batch_results, batch_errors = result
                    ans.extend(batch_results)
                    n_total_errors += batch_errors
                    n_total_attempted += len(batch_results) + batch_errors
                else:
                    # Backward-compat fallback (list return)
                    ans.extend(result)
                    n_total_attempted += len(result)
            except BrokenProcessPool:
                logging.error(
                    "进程池崩溃（可能内存不足），尝试减少 workers 数量或数据量")
                pool_broken = True
                break
            except Exception as e:
                logging.error(
                    f"批次处理异常，丢失 {chunk_size} 个 PSM: {e}")
                n_total_errors += chunk_size
                n_total_attempted += chunk_size
        return ans, n_total_errors, n_total_attempted, pool_broken

    def run(self) -> None:
        logging.info(f"运行任务 {self.workname}")

        # 加载DIA-NN 结果，加载 Data manager
        self.load()

        # TODO: 根据不同的谱图标题，分配到不同的任务，多进程执行
        # 分配不同的进程运行
        self.distribute()

        # TODO: self.save()
