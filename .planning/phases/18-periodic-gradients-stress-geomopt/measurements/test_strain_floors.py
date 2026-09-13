"""Harness regression tests; production/upstream source is never edited."""
import ast
import unittest

from pyscf.pbc.dft import numint
from strain_floors import StableEnergyReduction, energy_reduction


class EnergyReductionTests(unittest.TestCase):
    def test_only_named_energy_dot_is_rewritten(self):
        rewrite = StableEnergyReduction()
        tree = rewrite.visit(ast.parse("den.dot(exc)\nao.dot(dm)\nden.dot(other)"))
        self.assertEqual(rewrite.replacements, 1)
        self.assertEqual(ast.unparse(tree), "_phase18_energy_fsum(den * exc)\nao.dot(dm)\nden.dot(other)")

    def test_upstream_is_untouched(self):
        original = numint.nr_rks
        with energy_reduction("upstream"):
            self.assertIs(numint.nr_rks, original)
        self.assertIs(numint.nr_rks, original)

    def test_restored_after_exception(self):
        original = numint.nr_rks
        with self.assertRaisesRegex(RuntimeError, "test exception"):
            with energy_reduction("stable"):
                self.assertIsNot(numint.nr_rks, original)
                raise RuntimeError("test exception")
        self.assertIs(numint.nr_rks, original)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            with energy_reduction("invalid"):
                self.fail("invalid mode entered")


if __name__ == "__main__":
    unittest.main()
