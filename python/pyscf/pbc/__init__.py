"""pyscf.pbc overlay — the import path only, no bindings yet (plan 09-09).

This is the pyscf-rs `_native` re-export overlay (DISTINCT from the vendored
upstream `pyscf/pbc/`). It exists so `import pyscf.pbc` resolves against the
overlay package for the whole of the v2.0 PBC milestone, instead of silently
falling through to a half-present namespace once the first binding lands.

**There is deliberately nothing to re-export yet.** PBC-MASTER-PLAN D-PBC-14
puts every periodic PyO3 binding in Phase 20 plan 20-05: the Rust side
(`pyscf-pbc-gto`, `pyscf-pbc-scf`, ...) is built and gated against upstream
crate by crate first, and only then exposed to Python in one pass. Adding
bindings piecemeal would fork the dispatch story that
`python/pyscf/scf/__init__.py` established.

`pyscf/__init__.py` calls `pkgutil.extend_path`, so submodules of the vendored
upstream `pyscf.pbc` that this file does not shadow still resolve — a user can
keep using upstream `pyscf.pbc.*` today and pick up the Rust implementations as
plan 20-05 replaces them.

Phase 9 shipped (Rust-side, no bindings):
  * ``pyscf_pbc_gto::Cell`` — lattice, reciprocal vectors, rcut, mesh
  * ``get_Gv`` / ``get_SI`` / ``get_uniform_grids``
  * ``get_lattice_Ls`` / ``super_cell`` / ``cell_plus_imgs``
  * ``make_kpts`` + ``pyscf_pbc_lib::kpts_helper``
  * ``ewald`` / ``energy_nuc``
  * complex algebra in ``pyscf_algebra`` (``CTensor``, ``zgemm``, ``zeigh``,
    ``oracle_zsum``)

See ``.planning/phases/09-pbc-foundation/09-VERIFICATION.md``.
"""

import pkgutil as _pkgutil

__path__ = _pkgutil.extend_path(__path__, __name__)
del _pkgutil

__all__: list[str] = []
