"""测试 speclib_inspect CLI 的 summarize（用合成 fixture，不依赖大文件）。"""
from tools.speclib_inspect import summarize


def test_summarize_runs_on_fixture(lib_files):
    text = summarize(
        library_dir=str(lib_files),
        fasta_path=str(lib_files / "db.fasta"),
        mod_path=str(lib_files / "modification.ini"),
        element_path=str(lib_files / "element.ini"),
        aa_path=str(lib_files / "aa.ini"),
        n_samples=1, tol=1e-4)
    assert "peptides: 2" in text
    assert "chg_max: 2" in text
    assert "mass pass: 2/2" in text
