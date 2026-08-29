"""Shared cells for the Phase-14 pre-implementation measurements.

Geometry in BOHR (CODATA drift — same note as the pbc-df test fixtures).
"""
from pyscf.pbc import gto


def diamond():
    h, q = 3.37032, 1.68516
    c = gto.Cell()
    c.a = [[0., h, h], [h, 0., h], [h, h, 0.]]
    c.atom = [('C', (0., 0., 0.)), ('C', (q, q, q))]
    c.basis = 'gth-szv'
    c.pseudo = 'gth-pade'
    c.unit = 'Bohr'
    c.verbose = 0
    return c.build()


def he_fcc():
    """ALL-ELECTRON control — reaches get_nuc, which a gth cell never does."""
    h = 2.834589
    c = gto.Cell()
    c.a = [[0., h, h], [h, 0., h], [h, h, 0.]]
    c.atom = [('He', (0., 0., 0.))]
    c.basis = 'sto-3g'
    c.unit = 'Bohr'
    c.verbose = 0
    return c.build()
