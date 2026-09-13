"""Phase 18-01: reproduce vendored gradient anchors and measure FD floors.

Run from the repository root with its .venv interpreter, OMP_NUM_THREADS=1
and OPENBLAS_NUM_THREADS=1. JSON output includes all raw energies and steps.
"""
import argparse
import importlib
import json
import sys
import unittest

import numpy as np
import pyscf
from pyscf import lib
from pyscf.pbc import scf, dft

assert pyscf.__version__ == "2.12.1", pyscf.__version__


def anchors():
    """Run upstream assertions while reporting their actual fingerprints."""
    original = lib.fp

    def fingerprint(a):
        value = original(a)
        print(json.dumps({"fingerprint": float(value), "shape": list(np.shape(a))}), flush=True)
        return value

    lib.fp = fingerprint
    try:
        for name in ("krhf", "kuhf", "krks", "kuks", "krkspu", "kukspu"):
            module = importlib.import_module("pyscf.pbc.grad.test.test_" + name)
            print(json.dumps({"suite": name}), flush=True)
            result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(
                unittest.defaultTestLoader.loadTestsFromModule(module)
            )
            if not result.wasSuccessful():
                raise RuntimeError("upstream gradient suite failed: " + name)
    finally:
        lib.fp = original


def sweep():
    module = importlib.import_module("pyscf.pbc.grad.test.test_krhf")
    module.setUpModule()
    try:
        cell = module.cell.copy()
        cell.verbose = 0
        for name in ("krhf", "pbe"):
            kpts = module.kpts if name == "krhf" else cell.make_kpts([1, 1, 3])
            mf = scf.KRHF(cell, kpts, exxdiv=None) if name == "krhf" else dft.KRKS(cell, kpts, xc="pbe", exxdiv=None)
            mf.max_cycle = 200
            mf.conv_tol = 1e-12
            mf.conv_tol_grad = 1e-8
            mf.kernel()
            if not mf.converged:
                raise RuntimeError("unconverged reference " + name)
            gradient = mf.nuc_grad_method().kernel()
            scanner = mf.as_scanner()
            rows = []
            for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
                energies = []
                for sign in (1, -1):
                    coords = cell.atom_coords().copy()
                    coords[1, 2] += sign * h / 2
                    displaced = cell.set_geom_(coords, unit="Bohr", inplace=False)
                    energies.append(float(scanner(displaced)))
                    if not scanner.converged:
                        raise RuntimeError("unconverged displaced SCF")
                fd = (energies[0] - energies[1]) / h
                row = {"method": name, "full_step": h, "half_step": h / 2,
                       "energy_plus": energies[0], "energy_minus": energies[1],
                       "analytic": float(gradient[1, 2]), "fd": fd,
                       "residual": abs(fd - float(gradient[1, 2])),
                       "cancellation_estimate": 2 * np.finfo(float).eps * abs(mf.e_tot) / h}
                rows.append(row)
                print(json.dumps(row), flush=True)
            print(json.dumps({"method": name, "minimum": min(rows, key=lambda row: row["residual"])}), flush=True)
    finally:
        module.tearDownModule()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("anchors", "sweep"))
    args = parser.parse_args()
    print(json.dumps({"pyscf": pyscf.__version__, "source": pyscf.__file__}), flush=True)
    {"anchors": anchors, "sweep": sweep}[args.mode]()
