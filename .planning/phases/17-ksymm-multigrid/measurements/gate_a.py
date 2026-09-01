#!/usr/bin/env python
"""Gate A -- IBZ integers. Reproduces lib/test/test_kpts_ksymm.py:56-89 on
upstream's own Si cell, then repeats the six configurations on PBC-MASTER-PLAN
§9.2's si/diamond/lif/he_fcc/graphene to see whether the integers travel with
lattice TYPE (17-CONTEXT §3.10) or not.

Run as:
    PYTHONPATH=<repo root> <venv>/bin/python -u gate_a.py
"""
import sys
import numpy as np
import pyscf
assert pyscf.__version__ == "2.12.1", pyscf.__version__
from pyscf.lib.misc import finger
from pyscf.pbc import gto

KMESH = [16, 16, 16]

def six_configs(cell, cell_symm, label):
    print(f"--- {label} ---")
    kpts = cell.make_kpts(KMESH, space_group_symmetry=True)
    print(f"  space_group_symmetry=True                       nkpts_ibz={kpts.nkpts_ibz}  finger={finger(kpts.kpts_ibz)!r}")

    kpts1 = cell_symm.make_kpts(KMESH, space_group_symmetry=True, time_reversal_symmetry=True)
    if kpts1.kpts_ibz.shape == kpts.kpts_ibz.shape:
        delta = f"{abs(kpts1.kpts_ibz - kpts.kpts_ibz).max():.3e}"
    else:
        delta = f"N/A (shape {kpts1.kpts_ibz.shape} vs {kpts.kpts_ibz.shape})"
    print(f"  symmorphic=True, time_reversal_symmetry=True    nkpts_ibz={kpts1.nkpts_ibz}  finger={finger(kpts1.kpts_ibz)!r}  max|kpts_ibz - kpts0.kpts_ibz|={delta}")

    kpts2 = cell_symm.make_kpts(KMESH, space_group_symmetry=True, time_reversal_symmetry=False)
    print(f"  symmorphic=True, time_reversal_symmetry=False   nkpts_ibz={kpts2.nkpts_ibz}  finger={finger(kpts2.kpts_ibz)!r}")

    kpts3 = cell.make_kpts(KMESH, with_gamma_point=False, space_group_symmetry=True)
    print(f"  with_gamma_point=False, space_group_symmetry=True   nkpts_ibz={kpts3.nkpts_ibz}  finger={finger(kpts3.kpts_ibz)!r}")

    kpts4 = cell_symm.make_kpts(KMESH, with_gamma_point=False, space_group_symmetry=True)
    print(f"  with_gamma_point=False, symmorphic=True          nkpts_ibz={kpts4.nkpts_ibz}  finger={finger(kpts4.kpts_ibz)!r}")

    kpts5 = cell.make_kpts(KMESH, time_reversal_symmetry=True)
    print(f"  time_reversal_symmetry=True only                nkpts_ibz={kpts5.nkpts_ibz}")
    sys.stdout.flush()
    return dict(A=kpts.nkpts_ibz, B=kpts1.nkpts_ibz, C=kpts2.nkpts_ibz,
                D=kpts3.nkpts_ibz, E=kpts4.nkpts_ibz, F=kpts5.nkpts_ibz)

results = {}

# --- upstream's own Si cell (test_kpts_ksymm.py:30-40) ---
cell = gto.Cell()
cell.atom = """
    Si  0.0 0.0 0.0
    Si  1.3467560987 1.3467560987 1.3467560987
"""
cell.a = [[0.0, 2.6935121974, 2.6935121974],
          [2.6935121974, 0.0, 2.6935121974],
          [2.6935121974, 2.6935121974, 0.0]]
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.mesh = [20] * 3
cell.space_group_symmetry = True
cell.build()
cell_symm = cell.copy()
cell_symm.build(symmorphic=True)
results['upstream_si'] = six_configs(cell, cell_symm, "upstream Si (a=2.6935121974 Ang half-vectors)")

EXPECTED = dict(A=145, B=145, C=245, D=408, E=816, F=2052)
print()
print("EXPECTED:", EXPECTED)
print("GOT     :", results['upstream_si'])
if results['upstream_si'] == EXPECTED:
    print("GATE A on upstream's own Si cell: REPRODUCED EXACTLY")
else:
    print("GATE A MISMATCH -- STOP THE PHASE")
    sys.exit(1)

# --- PBC-MASTER-PLAN §9.2 si (a = 5.4306 Ang, conventional cubic; build primitive fcc) ---
def diamond_struct_cell(atom_symbol, a_ang, basis, pseudo=None):
    """Diamond structure (space group Fd-3m, NON-symmorphic): 2nd basis atom
    at a/4 * (1,1,1) in Cartesian -- the tetrahedral offset, NOT a/2 (that is
    rocksalt's octahedral offset). This is what upstream's own Si fixture
    uses (1.3467560987 = 2.6935121974/2 = a_conv/4 with a_conv = 5.3870243948)."""
    c = gto.Cell()
    half = a_ang / 2.0
    quarter = a_ang / 4.0
    c.atom = f"{atom_symbol} 0. 0. 0.\n{atom_symbol} {quarter} {quarter} {quarter}"
    c.a = [[0., half, half], [half, 0., half], [half, half, 0.]]
    c.basis = basis
    if pseudo:
        c.pseudo = pseudo
    c.mesh = [20] * 3
    c.space_group_symmetry = True
    c.build()
    return c

def rocksalt_cell(sym1, sym2, a_ang, basis, pseudo=None):
    """Rocksalt structure (space group Fm-3m, symmorphic): 2nd basis atom at
    a/2 * (1,1,1) -- the octahedral offset."""
    c = gto.Cell()
    half = a_ang / 2.0
    c.atom = f"{sym1} 0. 0. 0.\n{sym2} {half} {half} {half}"
    c.a = [[0., half, half], [half, 0., half], [half, half, 0.]]
    c.basis = basis
    if pseudo:
        c.pseudo = pseudo
    c.mesh = [20] * 3
    c.space_group_symmetry = True
    c.build()
    return c

def single_atom_fcc_cell(atom_symbol, a_ang, basis, pseudo=None):
    """Single-atom primitive fcc cell (space group Fm-3m)."""
    c = gto.Cell()
    half = a_ang / 2.0
    c.atom = f"{atom_symbol} 0. 0. 0."
    c.a = [[0., half, half], [half, 0., half], [half, half, 0.]]
    c.basis = basis
    if pseudo:
        c.pseudo = pseudo
    c.mesh = [20] * 3
    c.space_group_symmetry = True
    c.build()
    return c

si = diamond_struct_cell('Si', 5.4306, 'gth-szv', 'gth-pade')
si_symm = si.copy(); si_symm.build(symmorphic=True)
results['si_9_2'] = six_configs(si, si_symm, "PBC-MASTER-PLAN si (a=5.4306 Ang, diamond structure)")

diamond = diamond_struct_cell('C', 3.5668, 'gth-szv', 'gth-pade')
diamond_symm = diamond.copy(); diamond_symm.build(symmorphic=True)
results['diamond'] = six_configs(diamond, diamond_symm, "diamond (a=3.5668 Ang, diamond structure)")

# lif -- rocksalt (symmorphic Fm-3m), a=4.03 Ang
lif = rocksalt_cell('Li', 'F', 4.03, 'gth-szv', 'gth-pade')
lif_symm = lif.copy(); lif_symm.build(symmorphic=True)
results['lif'] = six_configs(lif, lif_symm, "lif (rocksalt a=4.03 Ang)")

he_fcc = single_atom_fcc_cell('He', 3.0, 'gth-szv', 'gth-pade')
he_fcc_symm = he_fcc.copy(); he_fcc_symm.build(symmorphic=True)
results['he_fcc'] = six_configs(he_fcc, he_fcc_symm, "he_fcc (a=3.0 Ang, single-atom fcc)")

# graphene -- hexagonal, 20 Ang vacuum, dimension=2. Right-handed conventional
# hexagonal lattice vectors: a1=(a,0,0), a2=(-a/2, a*sqrt(3)/2, 0), a3=(0,0,vac).
a_gr = 2.46
c_vac = 20.0
a1 = np.array([a_gr, 0.0, 0.0])
a2 = np.array([-a_gr / 2.0, a_gr * np.sqrt(3) / 2.0, 0.0])
atom2 = (a1 + 2 * a2) / 3.0  # fractional (1/3, 2/3, 0)
graphene = gto.Cell()
graphene.atom = f"""
    C  0.0 0.0 0.0
    C  {atom2[0]:.10f} {atom2[1]:.10f} {atom2[2]:.10f}
"""
graphene.a = [list(a1), list(a2), [0.0, 0.0, c_vac]]
graphene.dimension = 2
graphene.basis = 'gth-szv'
graphene.pseudo = 'gth-pade'
graphene.mesh = [20, 20, 60]
graphene.space_group_symmetry = True
try:
    graphene.build()
    graphene_symm = graphene.copy(); graphene_symm.build(symmorphic=True)
    results['graphene'] = six_configs(graphene, graphene_symm, "graphene (dimension=2)")
except Exception as e:
    print(f"graphene FAILED: {type(e).__name__}: {e}")
    results['graphene'] = f"FAILED: {type(e).__name__}: {e}"

print()
print("=== SUMMARY: does the integer set travel with lattice TYPE? ===")
for name, r in results.items():
    print(f"  {name:12s}: {r}")

same_as_upstream = {k: v for k, v in results.items() if v == EXPECTED}
print()
print(f"Systems matching upstream Si's exact integer set {EXPECTED}: {list(same_as_upstream.keys())}")
