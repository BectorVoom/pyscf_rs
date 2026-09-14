"""pyscf.pbc overlay — the import path; native module tree registered, gto/df bound.

This is the pyscf-rs `_native` re-export overlay (DISTINCT from the vendored
upstream `pyscf/pbc/`). It exists so `import pyscf.pbc` resolves against the
overlay package for the whole of the v2.0 PBC milestone, instead of silently
falling through to a half-present namespace once the first binding lands.

What is registered (plan 20-08):
  * the NESTED native tree ``pyscf._native.pbc`` with ten children —
    ``gto``, ``scf``, ``dft``, ``df``, ``symm``, ``lib``, ``tools``, ``mp``,
    ``cc``, ``ci`` — each importable by a real import statement
    (``import pyscf._native.pbc.scf``) because each is entered in
    ``sys.modules`` under its full dotted name
    (``crates/pyscf-py/src/pbc/mod.rs``);
  * the complex, k-resolved NumPy boundary those bindings will use
    (``crates/pyscf-py/src/numpy_io.rs``, plan 20-07).

What is bound (plans 20-09, 20-10): ``pyscf.pbc.gto`` (``Cell``, ``M``,
``make_kpts``, ``band_path``, ``super_cell``, ...) and ``pyscf.pbc.df``
(``FFTDF``, ``AFTDF``/``PWDF``, ``GDF``/``DF``, ``MDF``, ``RSDF``/``RSGDF``) are
overlay packages re-exporting the native objects (``python/pyscf/pbc/gto``,
``python/pyscf/pbc/df``); names they do not bind fall through lazily to the
upstream submodules. The other eight children are still EMPTY — 20-12
(``scf``), 20-13 (``dft``), 20-14 (``symm``/``lib``/``tools``) and 20-15
(``mp``/``cc``/``ci``/``ao2mo``) fill them — so ``__all__`` is empty and
``python/pyscf/tests/test_pbc_identity_gate.py`` passes only its 7 gto/df cases.

`pyscf/__init__.py` calls `pkgutil.extend_path`, and so does this file: every
``pyscf.pbc.*`` submodule this overlay does not shadow — all but ``gto`` and ``df`` —
resolves to the vendored upstream pure-Python implementation. Keeping that
passthrough is deliberate; whether an unported family falls through silently
or announces itself is plan 20-17's decision, and removing ``extend_path``
here would break every unported family at once.

The Rust side (``pyscf_pbc_gto::Cell``, the ``PeriodicDf`` builders, the k-point
SCF/DFT/MP2/CC drivers, ...) is built and gated against upstream crate by crate
in Phases 9-19; see ``.planning/phases/20-pbc-python-bindings/20-CONTEXT.md §3``.
"""

import pkgutil as _pkgutil

__path__ = _pkgutil.extend_path(__path__, __name__)
del _pkgutil

__all__: list[str] = []
