"""Gate E and 18-17 screening, on identical converged gamma PBE solutions."""
import argparse
import json
import time
import numpy as np
import pyscf
from pyscf.pbc import dft
from pyscf.pbc.dft.multigrid import MultiGridNumInt2
from pyscf.pbc.grad import rhf
from reference_cells import NAMES, reference_cell

assert pyscf.__version__ == "2.12.1", pyscf.__version__


def measure(name):
    cell = reference_cell(name)
    print(json.dumps({"cell": name, "mesh": cell.mesh.tolist(), "nao": cell.nao,
                      "precision": cell.precision}), flush=True)
    mf = dft.RKS(cell, xc="pbe")
    mf._numint = MultiGridNumInt2(cell)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("reference SCF did not converge: " + name)
    grad = mf.nuc_grad_method()
    original = rhf._contract_vhf_dm
    results = {}
    timings = {False: [], True: []}
    try:
        for repeat in range(4):
            for screen in ([False, True] if repeat % 2 == 0 else [True, False]):
                def contract(*args, **kwargs):
                    kwargs["screen"] = screen
                    return original(*args, **kwargs)
                rhf._contract_vhf_dm = contract
                start = time.perf_counter()
                results[screen] = grad.kernel()
                elapsed = time.perf_counter() - start
                if repeat:
                    timings[screen].append(elapsed)
    finally:
        rhf._contract_vhf_dm = original
    print(json.dumps({"cell": name, "screening_difference": float(np.max(abs(results[True]-results[False]))),
                      "screened_seconds": timings[True], "unscreened_seconds": timings[False],
                      "screened_over_unscreened": float(np.median(timings[True])/np.median(timings[False]))}), flush=True)
    scanner = mf.as_scanner()
    full_step = 2e-4
    residuals = []
    for atom in range(cell.natm):
        for component in range(3):
            energies = []
            for sign in (1, -1):
                coords = cell.atom_coords().copy()
                coords[atom, component] += sign*full_step/2
                displaced = cell.set_geom_(coords, unit="Bohr", inplace=False)
                displaced.mesh = cell.mesh.copy()
                energies.append(float(scanner(displaced)))
                if not scanner.converged:
                    raise RuntimeError("displaced SCF did not converge")
            fd = (energies[0]-energies[1])/full_step
            residual = abs(fd-results[True][atom, component])
            residuals.append(residual)
            print(json.dumps({"cell": name, "atom": atom, "component": component,
                              "full_step": full_step, "energies": energies,
                              "analytic": float(results[True][atom, component]),
                              "fd": fd, "residual": float(residual)}), flush=True)
    print(json.dumps({"cell": name, "gamma_fd_max": float(max(residuals))}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell", choices=NAMES)
    args = parser.parse_args()
    print(json.dumps({"pyscf": pyscf.__version__, "source": pyscf.__file__}), flush=True)
    measure(args.cell)
