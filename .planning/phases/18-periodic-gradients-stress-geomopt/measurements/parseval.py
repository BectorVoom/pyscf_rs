"""18-16: compare matrix, real-grid and reciprocal-grid local contractions.

Fixed seeded complex positive density at a nonzero k point; no SCF noise.
FFT normalization is explicit: inverse FFT includes 1/N; weight is vol/N.
"""
import json
import numpy as np
import pyscf
from pyscf.pbc.dft.numint import eval_ao
from pyscf.pbc.gto.pseudo.pp_int import get_gth_vlocG_part1
from pyscf.pbc import tools
from reference_cells import NAMES, reference_cell

assert pyscf.__version__ == "2.12.1", pyscf.__version__


def measure(name):
    cell = reference_cell(name)
    coords = cell.get_uniform_grids()
    ngrid = len(coords)
    weight = cell.vol/ngrid
    kpt = cell.get_abs_kpts(np.array([[.13, -.07, .11]]))[0]
    ao = eval_ao(cell, coords, kpt=kpt)
    rng = np.random.default_rng(18)
    c = rng.normal(size=(cell.nao, cell.nao)) + 1j*rng.normal(size=(cell.nao, cell.nao))
    dm = c @ c.conj().T
    rho = np.einsum('gi,ij,gj->g', ao, dm, ao.conj()).real
    atom_potential = get_gth_vlocG_part1(cell, cell.Gv)
    si = cell.get_SI(cell.Gv)
    # Differentiate each atom's structure factor, as hcore_generator does.
    for atom in range(cell.natm):
        for component in range(3):
            vg = 1j * cell.Gv[:, component] * si[atom] * atom_potential[atom]
            vr = tools.ifft(vg, cell.mesh).real/weight
            matrix = ao.conj().T @ (vr[:, None]*ao) * weight
            matrix_value = np.einsum('ij,ji->', matrix, dm).real
            real_value = np.dot(vr, rho)*weight
            # Transform the real field actually used by the matrix route;
            # this also treats any self-conjugate Nyquist modes identically.
            vg_real = tools.fft(vr, cell.mesh)
            reciprocal_value = np.vdot(vg_real, tools.fft(rho, cell.mesh)).real*weight/ngrid
            scale = max(abs(real_value), abs(reciprocal_value), 1e-30)
            print(json.dumps({"cell": name, "mesh": cell.mesh.tolist(), "atom": atom,
                              "component": component, "matrix": float(matrix_value),
                              "real": float(real_value), "reciprocal": float(reciprocal_value),
                              "absolute_real_reciprocal": float(abs(real_value-reciprocal_value)),
                              "relative_real_reciprocal": float(abs(real_value-reciprocal_value)/scale),
                              "absolute_matrix_real": float(abs(matrix_value-real_value))}), flush=True)


if __name__ == "__main__":
    print(json.dumps({"pyscf": pyscf.__version__, "source": pyscf.__file__}), flush=True)
    for name in NAMES:
        measure(name)
