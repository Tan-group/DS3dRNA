"""Regression tests for the consensus utilities in both design examples.

Run with: python -m unittest discover -s tests -p 'test_consensus*.py'
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULES = []
for example in ('Design', 'MultiState_Design'):
    spec = importlib.util.spec_from_file_location(
        example, ROOT / 'Examples' / example / 'DS3dRNA_consensus_script.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    MODULES.append((example, module))


class ChainBoundaryTests(unittest.TestCase):
    def test_conversion_resets_at_filename_chain_boundary(self):
        # Five A bases end chain A; chain B must still begin with its preferred A.
        for example, module in MODULES:
            with self.subTest(example=example), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / 'target_A:5_B:8.csv'
                source.write_text('Designed_seq,count,E_fine(kBT)\nAAAAA&AAA,9,-10\nUUUUU&UUU,1,-9\n')
                consensus_dir, emin_dir = root / 'consensus', root / 'emin'
                consensus_dir.mkdir()
                emin_dir.mkdir()
                result = module.convert_one_csv(
                    source, consensus_dir, emin_dir, model_seeds=[1],
                    method_prefix='test', use_count=True, max_run=5)
                consensus = json.loads(Path(result['consensus_json']).read_text())
                self.assertEqual([r['rna']['sequence'] for r in consensus['sequences']],
                                 ['AAAAA', 'AAA'])
                emin = json.loads(Path(result['E_min_json']).read_text())
                self.assertEqual([r['rna']['sequence'] for r in emin['sequences']],
                                 ['AAAAA', 'AAA'])

    def test_single_chain_cap_is_preserved(self):
        for example, module in MODULES:
            with self.subTest(example=example):
                self.assertEqual(module.build_count_consensus(
                    [('AAAAAAAA', 9), ('UUUUUUUU', 1)], max_run=5), 'AAAAAUAA')

    def test_cap_still_applies_inside_each_chain(self):
        for example, module in MODULES:
            with self.subTest(example=example):
                self.assertEqual(module.build_count_consensus(
                    [('A' * 12, 9), ('U' * 12, 1)], max_run=5,
                    chain_specs=[('A', 6), ('B', 12)]), 'AAAAAUAAAAAU')

    def test_disabled_cap_preserves_profile_consensus(self):
        for example, module in MODULES:
            with self.subTest(example=example):
                self.assertEqual(module.build_count_consensus(
                    [('A' * 8, 9), ('U' * 8, 1)], max_run=0,
                    chain_specs=[('A', 5), ('B', 8)]), 'A' * 8)

    def test_chain_spec_length_must_match_sequence(self):
        for example, module in MODULES:
            with self.subTest(example=example), self.assertRaises(ValueError):
                module.build_count_consensus([('AAAA', 1)], max_run=5,
                                            chain_specs=[('A', 5)])


if __name__ == '__main__':
    unittest.main()
