"""PairFlow 批次结果收集的错误/丢失计数测试。"""
from concurrent.futures import Future
from workflows.pair_flow import PairFlow


def test_collect_counts_success_and_errors_within_batch():
    f = Future()
    f.set_result(([{"a": 1}, {"a": 2}], 3))   # 2 成功, 3 批内错误
    ans, n_err, n_att, broken = PairFlow._collect_batch_results({f: 5})
    assert len(ans) == 2
    assert n_err == 3
    assert n_att == 2 + 3
    assert broken is False


def test_collect_counts_lost_psms_on_whole_batch_failure():
    """整批 future 抛异常时，丢失的 chunk_size 必须计入错误数与尝试数，
    否则错误率分母漏掉这些 PSM → 静默丢数据。"""
    f_ok = Future()
    f_ok.set_result(([{"a": 1}], 0))
    f_fail = Future()
    f_fail.set_exception(ValueError("boom"))
    ans, n_err, n_att, broken = PairFlow._collect_batch_results(
        {f_ok: 1, f_fail: 5000})
    assert len(ans) == 1
    assert broken is False
    assert n_err == 5000            # 丢失的 5000 计入错误（而非 +1）
    assert n_att == 1 + 5000        # 尝试数包含丢失批次
