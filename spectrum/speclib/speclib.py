"""SpecLib 锁步流式 loader：pdb+RT+MS2 同序逐肽段产出，并提供质量自校验。

体量大（ms2 ~4.4GB），不全量物化：iter_peptides() 流式 yield 已填好
pred_rt/pred_ms2 的 LibPeptide，由调用方即用即弃。
"""
import os
from dataclasses import dataclass, field

from .config_io import (parse_fasta, parse_modifications,
                        parse_element_masses, parse_residue_masses, water_mass)
from .pepdata import iter_pepdata, LibPeptide
from .predictions import read_rt_pred, iter_ms2_records, read_chg_max_from_trailer


@dataclass
class MassValidationReport:
    total: int
    passed: int
    failed: int
    max_abs_error: float
    failures: list = field(default_factory=list)  # (index, seq, computed, stored, err)


class SpecLib:
    def __init__(self, *, pepdata_path, rt_path, ms2_path,
                 proteins, mods_by_id, rt, chg_max):
        self.pepdata_path = pepdata_path
        self.rt_path = rt_path
        self.ms2_path = ms2_path
        self.proteins = proteins
        self.mods_by_id = mods_by_id
        self.rt = rt
        self.chg_max = chg_max

    @property
    def num_peptides(self) -> int:
        return len(self.rt)

    @classmethod
    def open(cls, *, pepdata_path: str, rt_path: str, ms2_path: str,
             fasta_path: str, mod_path: str) -> "SpecLib":
        proteins = parse_fasta(fasta_path)
        mods_by_id = {m.mod_id: m for m in parse_modifications(mod_path)}
        rt = read_rt_pred(rt_path)
        chg_max = read_chg_max_from_trailer(ms2_path)
        if not (1 <= chg_max <= 6):
            raise ValueError(f"chg_max {chg_max} out of range [1,6]")
        return cls(pepdata_path=pepdata_path, rt_path=rt_path,
                   ms2_path=ms2_path, proteins=proteins,
                   mods_by_id=mods_by_id, rt=rt, chg_max=chg_max)

    @classmethod
    def open_dir(cls, library_dir: str, *, fasta_path: str,
                 mod_path: str) -> "SpecLib":
        return cls.open(
            pepdata_path=os.path.join(library_dir, "pepdata.pdb"),
            rt_path=os.path.join(library_dir, "pepdata.rt.predb"),
            ms2_path=os.path.join(library_dir, "pepdata.ms2.predb"),
            fasta_path=fasta_path, mod_path=mod_path)

    def iter_peptides(self):
        """锁步流式：pdb+RT+MS2 同序逐肽段 yield（已填 pred_rt/pred_ms2）。"""
        ms2 = iter_ms2_records(self.ms2_path)
        n_rt = len(self.rt)
        i = -1
        for i, pep in enumerate(iter_pepdata(
                self.pepdata_path, self.proteins, self.mods_by_id)):
            if i >= n_rt:
                raise ValueError(
                    f"peptide count exceeds RT count {n_rt}")
            pep.pred_rt = self.rt[i]
            d = {}
            for chg in range(1, self.chg_max + 1):
                try:
                    d[chg] = next(ms2)
                except StopIteration:
                    raise ValueError(
                        f"ms2 records exhausted at peptide {i} charge {chg}")
            pep.pred_ms2 = d
            yield pep
        if i + 1 != n_rt:
            raise ValueError(
                f"peptide count {i + 1} != RT count {n_rt}")
        # 对称校验：MS2 不应多于 chg_max×M（过供 = 与 pdb 错位）
        try:
            next(ms2)
        except StopIteration:
            pass
        else:
            raise ValueError(
                "ms2 has more records than chg_max * peptides")

    def validate_masses(self, element_path: str, aa_path: str,
                        tol: float = 0.01, limit: int | None = None
                        ) -> MassValidationReport:
        em = parse_element_masses(element_path)
        res = parse_residue_masses(aa_path, em)
        water = water_mass(em)
        failures = []
        max_err = 0.0
        passed = total = 0
        for pep in iter_pepdata(self.pepdata_path, self.proteins,
                               self.mods_by_id):
            computed = (water
                        + sum(res.get(a, 0.0) for a in pep.sequence)
                        + sum(m.mono_mass for m in pep.mods))
            err = abs(computed - pep.neutral_mass)
            if err > max_err:
                max_err = err
            if err <= tol:
                passed += 1
            elif len(failures) < 20:
                failures.append((total, pep.sequence, computed,
                                 pep.neutral_mass, err))
            total += 1
            if limit is not None and total >= limit:
                break
        return MassValidationReport(
            total=total, passed=passed, failed=total - passed,
            max_abs_error=max_err, failures=failures)
