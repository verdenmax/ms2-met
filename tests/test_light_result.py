import unittest

from spectrum.light_result import parse_alphadia_peptide_modify
from spectrum.psm_info import PSMInfo


class ParseAlphaDiaPeptideModifyTests(unittest.TestCase):
    def test_maps_tmt_variants_to_unimod_ids(self):
        modifications = parse_alphadia_peptide_modify(
            "TMT6plex@K;TMTpro Zero@Any N-term;Oxidation@M",
            "4;1;3",
        )

        self.assertEqual(modifications, [(3, 737), (0, 2017), (2, 35)])

    def test_skips_unsupported_modifications(self):
        modifications = parse_alphadia_peptide_modify(
            "Unsupported@K;TMT10plex@K",
            "2;4",
        )

        self.assertEqual(modifications, [(3, 737)])

    def test_tmt_modifications_contribute_to_fragment_mass(self):
        modifications = parse_alphadia_peptide_modify(
            "TMTpro@Any N-term;TMT6plex@K",
            "1;4",
        )
        psm = PSMInfo(
            sequence="PEPK",
            charge=2,
            modify=modifications,
            rt=1.0,
            precursor_mz=500.0,
            raw_title="raw",
            protein_names="protein",
        )

        self.assertGreater(psm.get_modify_mass(0), 0)
        self.assertGreater(psm.get_modify_mass(3), psm.get_modify_mass(0))


if __name__ == "__main__":
    unittest.main()
