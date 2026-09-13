"""Record each upstream strain-component assertion without changing its bound.

Run from the repository root, using .venv and one OMP/BLAS thread.
This initial measurement covers upstream's own fixtures, not the five port cells.
"""
import argparse
import ast
from contextlib import contextmanager
import inspect
import json
import math
from pathlib import Path
import sys
import types
import unittest

import pyscf

assert pyscf.__version__ == "2.12.1", pyscf.__version__


class StableEnergyReduction(ast.NodeTransformer):
    """Change only the periodic nr_rks energy dot, failing closed on drift."""

    def __init__(self):
        self.replacements = 0

    def visit_Call(self, node):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "den" and node.func.attr == "dot"
                and len(node.args) == 1 and not node.keywords
                and isinstance(node.args[0], ast.Name) and node.args[0].id == "exc"):
            self.replacements += 1
            return ast.copy_location(ast.Call(
                func=ast.Name(id="_phase18_energy_fsum", ctx=ast.Load()),
                args=[ast.BinOp(left=node.func.value, op=ast.Mult(), right=node.args[0])],
                keywords=[],
            ), node)
        return node


@contextmanager
def energy_reduction(mode):
    """Temporarily stabilize the FD reference, never the analytic derivative.

    Upstream mode is untouched. Stable mode replaces den.dot(exc) by fsum of
    the same binary64 products, in memory only. Assertions and steps are unchanged.
    """
    if mode == "upstream":
        yield
        return
    if mode != "stable":
        raise ValueError("unknown energy reduction: " + mode)
    from pyscf.pbc.dft import numint

    original = numint.nr_rks
    rewrite = StableEnergyReduction()
    tree = rewrite.visit(ast.parse(inspect.getsource(original)))
    if rewrite.replacements != 1:
        raise RuntimeError("expected exactly one den.dot(exc) in periodic nr_rks")
    namespace = dict(numint.__dict__)
    namespace["_phase18_energy_fsum"] = math.fsum
    exec(compile(ast.fix_missing_locations(tree), "<phase18-stable-energy>", "exec"), namespace)
    numint.nr_rks = namespace["nr_rks"]
    try:
        yield
    finally:
        numint.nr_rks = original


class RecordAssertions(ast.NodeTransformer):
    def visit_Assert(self, node):
        self.generic_visit(node)
        test = node.test
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Lt):
            test.left = ast.Call(
                func=ast.Name(id="_record_residual", ctx=ast.Load()),
                args=[test.left, test.comparators[0], ast.Constant(node.lineno)],
                keywords=[],
            )
        return node


def record_residual(value, bound, line):
    print(json.dumps({"line": line, "residual": float(value), "bound": float(bound)}), flush=True)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-reduction", choices=("upstream", "stable"), default="upstream",
                        help="stable modifies only the FD reference energy reduction in memory")
    args = parser.parse_args()
    source = Path(pyscf.__file__).parent / "pbc/grad/test/test_rks_stress.py"
    print(json.dumps({"pyscf": pyscf.__version__, "source": str(source),
                      "energy_reduction": args.energy_reduction}), flush=True)
    tree = ast.fix_missing_locations(RecordAssertions().visit(ast.parse(source.read_text())))
    module = types.ModuleType("phase18_strain_components")
    module.__file__ = str(source)
    module._record_residual = record_residual
    sys.modules[module.__name__] = module
    exec(compile(tree, str(source), "exec"), module.__dict__)
    names = (
        "ovlp", "kin", "weight", "coulG", "eval_ao_cart", "eval_ao_sph",
        "eval_ao_deriv1_cart", "eval_ao_deriv1_sph", "eval_ao_grid_response",
        "lattice_vector_derivatives", "get_vxc_lda", "get_vxc_gga",
        "get_vxc_mgga", "get_j", "get_nuc", "get_pp",
    )
    suite = unittest.TestSuite(module.KnownValues("test_" + name) for name in names)
    with energy_reduction(args.energy_reduction):
        result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2, failfast=True).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
