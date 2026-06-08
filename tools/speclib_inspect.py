"""流式加载 pFind 谱库并打印摘要 + 质量交叉校验，用于真实文件验证。

用法:
  python -m tools.speclib_inspect --library-dir DIR \
      --fasta merge_human_ecoli_yeast.fasta --mod modification.ini \
      [--element element.ini --aa aa.ini] [--n-samples 5] \
      [--tol 0.01] [--mass-limit N]
"""
import argparse

from spectrum.speclib import SpecLib


def summarize(*, library_dir: str, fasta_path: str, mod_path: str,
              element_path: str | None = None, aa_path: str | None = None,
              n_samples: int = 5, tol: float = 0.01,
              mass_limit: int | None = None) -> str:
    lib = SpecLib.open_dir(library_dir, fasta_path=fasta_path,
                           mod_path=mod_path)
    lines = []
    lines.append(f"peptides: {lib.num_peptides}")
    lines.append(f"chg_max: {lib.chg_max}")
    if len(lib.rt):
        lines.append(f"rt range (min): {min(lib.rt):.3f} .. {max(lib.rt):.3f}")

    # 流式取前 n_samples 个肽段（含 RT/MS2），不全载
    samples = []
    for pep in lib.iter_peptides():
        samples.append(pep)
        if len(samples) >= n_samples:
            break
    for pep in samples:
        modstr = ",".join(f"{m.pos}:{m.name}" for m in pep.mods) or "-"
        top = sorted((ion for ions in pep.pred_ms2.values() for ion in ions),
                     key=lambda x: x.intensity, reverse=True)[:3]
        topstr = " ".join(
            f"{i.ion_type}{i.frag_pos}^{i.frag_charge}={i.intensity:.2f}"
            for i in top)
        lines.append(
            f"  {pep.sequence} mods=[{modstr}] mass={pep.neutral_mass:.4f} "
            f"rt={pep.pred_rt:.2f} top_ms2=[{topstr}]")

    if element_path and aa_path:
        rep = lib.validate_masses(element_path, aa_path, tol=tol,
                                  limit=mass_limit)
        lines.append(f"mass pass: {rep.passed}/{rep.total} "
                     f"(max_abs_err={rep.max_abs_error:.5f}, tol={tol})")
        for idx, seq, computed, stored, err in rep.failures[:5]:
            lines.append(f"  FAIL #{idx} {seq} computed={computed:.4f} "
                         f"stored={stored:.4f} err={err:.4f}")
    else:
        lines.append("mass validation skipped (no --element/--aa)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Inspect a pFind spectral library")
    ap.add_argument("--library-dir", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--mod", required=True)
    ap.add_argument("--element", default=None)
    ap.add_argument("--aa", default=None)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--mass-limit", type=int, default=None,
                    help="只校验前 N 条质量（加速）")
    args = ap.parse_args()
    print(summarize(
        library_dir=args.library_dir, fasta_path=args.fasta,
        mod_path=args.mod, element_path=args.element, aa_path=args.aa,
        n_samples=args.n_samples, tol=args.tol, mass_limit=args.mass_limit))


if __name__ == "__main__":
    main()
