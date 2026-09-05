"""Shared cell builders for the Phase-16 (16-01) measurement scripts.

Every script that imports this asserts `pyscf.__version__ == "2.12.1"` at
import time, as every Phase 13/14/17 measurement does.
"""

import numpy as np
import pyscf

assert pyscf.__version__ == "2.12.1", pyscf.__version__

from pyscf.pbc import gto as pbcgto

DEFAULT_PRECISION = 1e-8


def diamond(precision=DEFAULT_PRECISION, basis="gth-szv"):
    """PBC-MASTER-PLAN §9.2 `diamond`: C2 fcc a = 3.5668 A, gth-szv/gth-pade."""
    a0 = 3.5668
    q = a0 / 4.0
    cell = pbcgto.Cell()
    cell.atom = [("C", (0.0, 0.0, 0.0)), ("C", (q, q, q))]
    cell.a = np.array([[0.0, a0 / 2, a0 / 2], [a0 / 2, 0.0, a0 / 2], [a0 / 2, a0 / 2, 0.0]])
    cell.basis = basis
    cell.pseudo = "gth-pade"
    cell.unit = "A"
    cell.precision = precision
    cell.verbose = 0
    cell.build()
    return cell


def he_fcc(precision=DEFAULT_PRECISION, basis="gth-szv"):
    """§9.2 `he_fcc`: He fcc a = 3.0 A, all-electron (no pseudo)."""
    a0 = 3.0
    cell = pbcgto.Cell()
    cell.atom = [("He", (0.0, 0.0, 0.0))]
    cell.a = np.array([[0.0, a0 / 2, a0 / 2], [a0 / 2, 0.0, a0 / 2], [a0 / 2, a0 / 2, 0.0]])
    cell.basis = basis
    cell.unit = "A"
    cell.precision = precision
    cell.verbose = 0
    cell.build()
    return cell


def graphene(precision=DEFAULT_PRECISION, basis="gth-szv"):
    """§9.2 `graphene`: C2 hexagonal with 20 A vacuum, dimension = 2."""
    a0 = 2.46
    c = 20.0
    cell = pbcgto.Cell()
    cell.atom = [("C", (0.0, 0.0, 0.0)), ("C", (a0 / 2, a0 / (2 * np.sqrt(3.0)), 0.0))]
    cell.a = np.array(
        [[a0, 0.0, 0.0], [-a0 / 2, a0 * np.sqrt(3.0) / 2, 0.0], [0.0, 0.0, c]]
    )
    cell.basis = basis
    cell.pseudo = "gth-pade"
    cell.unit = "A"
    cell.dimension = 2
    cell.low_dim_ft_type = None
    cell.precision = precision
    cell.verbose = 0
    cell.build()
    return cell


def fmt(x):
    """17 significant digits, the format every earlier measurement used."""
    return repr(float(x))
