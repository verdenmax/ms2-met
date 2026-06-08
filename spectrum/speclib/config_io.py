"""解析 pFind 谱库依赖的文本配置：FASTA、modification.ini、element.ini、aa.ini。

逻辑复刻自 pFindSDK：fastaparser.cpp / Reader.cpp(ReadMod/ReadAA/ReadElementInfo)。
"""
from dataclasses import dataclass


@dataclass
class Protein:
    ac: str
    description: str
    sequence: str

    @property
    def is_decoy(self) -> bool:
        return self.ac.startswith("REV_")


@dataclass
class ModEntry:
    mod_id: int
    name: str
    mono_mass: float
    sites: str
    mod_type: str


def _parse_fasta_header(line: str):
    """复刻 CFastaParser::ReadOnePrteinEntry 的 AC/DE 切分。"""
    s = line.rstrip("\r\n")

    def find(ch: str) -> int:
        i = s.find(ch)
        return i if i != -1 else len(s)

    tpos = find(" ")
    t2 = find("\t")
    if t2 < tpos:
        tpos = t2
    t2 = find("|")
    if t2 < tpos and t2 > 15:
        tpos = t2
    ac = s[1:tpos]
    de = s[tpos + 1:]
    return ac, de


def parse_fasta(path: str) -> list[Protein]:
    """按文件顺序解析蛋白条目；返回的 list 下标即 pdb 中的 pro_id。"""
    proteins: list[Protein] = []
    ac = de = None
    seq_parts: list[str] = []
    with open(path, encoding="latin-1") as fh:
        for raw in fh:
            if raw.startswith(">"):
                if ac is not None:
                    proteins.append(Protein(ac, de, "".join(seq_parts)))
                ac, de = _parse_fasta_header(raw)
                seq_parts = []
            else:
                seq_parts.append(raw.strip())
    if ac is not None:
        proteins.append(Protein(ac, de, "".join(seq_parts)))
    return proteins


def parse_modifications(path: str) -> list[ModEntry]:
    """复刻 CReader::ReadMod 的过滤与 read-order id 赋值。"""
    mods: list[ModEntry] = []
    with open(path, encoding="latin-1") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key.startswith("name"):
                continue
            if key == "@NUMBER_MODIFICATION":
                continue
            if key == "label_name":
                continue
            if key == "Met-loss+Acetyl[ProteinN-termM]":
                continue
            if key.startswith("Label_"):
                continue
            parts = val.split()
            mods.append(ModEntry(
                mod_id=len(mods) + 1,
                name=key,
                mono_mass=float(parts[2]),
                sites=parts[0],
                mod_type=parts[1],
            ))
    return mods


def parse_element_masses(path: str) -> dict[str, float]:
    """复刻 CReader::ReadElementInfo：取丰度最高同位素的质量。"""
    masses: dict[str, float] = {}
    with open(path, encoding="latin-1") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line.startswith("E"):
                continue
            value = line[line.find("=") + 1:]
            p1 = value.find("|")
            name = value[:p1]
            p2 = value.find("|", p1 + 1)
            mass_str = value[p1 + 1:p2]
            p3 = value.find("|", p2 + 1)
            ab_str = value[p2 + 1:p3]
            ms = [x for x in mass_str.split(",") if x != ""]
            ab = [x for x in ab_str.split(",") if x != ""]
            max_i, max_a = 0, -1.0
            for i, a in enumerate(ab):
                av = float(a)
                if av > max_a:
                    max_a, max_i = av, i
            masses[name] = float(ms[max_i])
    return masses


def parse_residue_masses(path: str, element_masses: dict[str, float]) -> dict[str, float]:
    """复刻 CReader::ReadAA：残基质量 = Σ 元素质量 × 个数（不含水）。"""
    residues: dict[str, float] = {}
    with open(path, encoding="latin-1") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line.startswith("R"):
                continue
            value = line[line.find("=") + 1:]
            p1 = value.find("|")
            name = value[:p1]
            if not name or not name[0].isupper():
                continue
            p2 = value.find("|", p1 + 1)
            comp = value[p1 + 1:p2]
            total = 0.0
            for elem in comp.split(")"):
                if "(" not in elem:
                    continue
                sym, _, cnt = elem.partition("(")
                if sym == "" or cnt == "":
                    continue
                total += element_masses.get(sym, 0.0) * int(cnt)
            residues[name] = total
    return residues


def water_mass(element_masses: dict[str, float]) -> float:
    return 2 * element_masses["H"] + element_masses["O"]
