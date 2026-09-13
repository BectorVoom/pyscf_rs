"""Phase-18 mirrors of Rust test_systems, expressed in Bohr.

Use the port's CODATA-2014 conversion explicitly, not PySCF's Angstrom conversion.
The actual Rust he_fcc builder uses gth-pade despite its older table heading.
"""
import numpy as np
import pyscf
from pyscf.pbc import gto

assert pyscf.__version__ == "2.12.1", pyscf.__version__
NAMES = ("diamond", "si", "lif", "he_fcc", "graphene")
BOHR_ANG = 0.52917721067


def reference_cell(name, basis="gth-szv"):
    edges = {"diamond": 3.5668, "si": 5.4306, "lif": 4.03, "he_fcc": 3.0}
    if name == "graphene":
        a0 = 2.46
        lattice = np.array([[a0, 0, 0], [-a0/2, a0*np.sqrt(3)/2, 0], [0, 0, 20.]])
        atoms = [("C", [0, 0, 0]), ("C", [0, a0/np.sqrt(3), 0])]
    else:
        a0 = edges[name]
        lattice = a0/2 * (np.ones((3, 3)) - np.eye(3))
        if name in ("diamond", "si"):
            symbol = "C" if name == "diamond" else "Si"
            atoms = [(symbol, [0, 0, 0]), (symbol, [a0/4]*3)]
        elif name == "lif":
            atoms = [("Li", [0, 0, 0]), ("F", [a0/2]*3)]
        else:
            atoms = [("He", [0, 0, 0])]
    return gto.M(atom=[(s, np.asarray(r)/BOHR_ANG) for s, r in atoms],
                 a=lattice/BOHR_ANG, unit="Bohr", basis=basis, pseudo="gth-pade",
                 dimension=2 if name == "graphene" else 3, precision=1e-8, verbose=0)
