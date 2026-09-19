"""The edge of the pyscf-rs PBC port: a machine-readable registry, ``which_impl``,
and the announced upstream passthrough (plan 20-17, decision D-PBC-35).

``python/pyscf/pbc/__init__.py`` keeps ``pkgutil.extend_path``, so every
``pyscf.pbc.*`` module this overlay does not shadow resolves to whichever other
``pyscf`` distribution is on ``sys.path`` — upstream PySCF *Python*, not the
Rust port. This module makes that visible:

* ``FAMILIES`` — one entry per upstream ``pyscf/pbc/<family>`` package
  (vendored 2.12.1 tree), with ``status`` ``"native"``, ``"partial"`` or
  ``"upstream"``, the Rust crate(s), the not-ported pieces and a note;
* ``which_impl(name)`` — ``"native"`` | ``"partial"`` | ``"upstream"`` for a
  family or a dotted name (and ``"refused"`` for a dotted name the overlay
  binds as a raising ``NotImplementedError`` stub);
* a ``sys.meta_path`` finder that, the first time per family per process an
  import resolves OUTSIDE the overlay, emits one ``PbcUpstreamFallthroughWarning``
  (a ``UserWarning``) naming the family. It never changes what is imported.

The finder only announces; it returns ``None`` from ``find_spec`` so normal
resolution continues unchanged. Measured import status of every upstream
module under the overlay: ``docs/pbc-status.md`` and
``.planning/phases/20-pbc-python-bindings/measurements/overlay-passthrough.md``.
"""

import importlib
import importlib.machinery
import importlib.util
import os
import sys
import warnings

__all__ = ["FAMILIES", "PbcUpstreamFallthroughWarning", "which_impl", "fallthrough_getattr",
           "load_upstream_module"]

OVERLAY_DIR = os.path.dirname(os.path.abspath(__file__))

_P19 = "Rust implementation exists (Phase 19), unbound in Python"

# Status is from the Python user's point of view: what `import pyscf.pbc.<family>`
# actually runs. `missing` lists upstream submodules / names with no native
# counterpart in Python; `names` maps dotted names (relative to the family) that
# `which_impl` must not infer from the overlay's `__all__`.
FAMILIES = {
    # ── overlay packages: the family's public surface is re-exported from _native ──
    "gto": {
        "status": "partial",
        "upstream": "pyscf/pbc/gto",
        "rust": ("pyscf-pbc-gto",),
        "bound_by": "20-09",
        "missing": ("ecp", "basis", "pseudo", "cell", "eval_gto", "ewald_methods"),
        "names": {"ecp": "upstream", "basis": "upstream", "pseudo": "upstream",
                  "cell": "upstream", "neighborlist": "upstream", "eval_gto": "upstream",
                  "ewald_methods": "upstream"},
        "note": "Cell/M/make_kpts/band_path/super_cell native; ECP input not bound; explicit "
                "shell-list basis and per-element pseudo dicts raise (20-09 D4); other names "
                "fall through lazily to upstream cell/basis/pseudo/neighborlist",
    },
    "df": {
        "status": "native",
        "upstream": "pyscf/pbc/df",
        "rust": ("pyscf-pbc-df",),
        "bound_by": "20-10",
        "missing": (),
        "names": {},
        "note": "FFTDF/AFTDF(PWDF)/GDF(DF)/MDF/RSDF(RSGDF) + density_fit native; upstream "
                "submodules (incore, outcore, fft_jk, df_jk, ft_ao, ...) fall through",
    },
    "scf": {
        "status": "partial",
        "upstream": "pyscf/pbc/scf",
        "rust": ("pyscf-pbc-scf",),
        "bound_by": "20-12",
        "missing": ("newton_ah", "stability", "cphf", "_response_functions", "scfint",
                    "kuhf_ksymm", "kghf_ksymm", "rsjk", "addons.convert_to_uhf",
                    "addons.convert_to_rhf", "addons.convert_to_ghf", "addons.convert_to_kscf",
                    "addons.canonical_occ_", "addons.project_dm_k2k",
                    "addons.mo_energy_with_exxdiv_none", "addons.smearing"),
        "names": {"newton_ah": "upstream", "newton": "upstream", "stability": "upstream",
                  "cphf": "upstream", "_response_functions": "upstream",
                  "scfint": "upstream", "rsjk": "upstream",
                  "kuhf_ksymm": "refused", "kghf_ksymm": "refused"},
        "note": "KRHF/KUHF/KROHF/KGHF/KsymAdaptedKRHF + gamma shims, smearing, chkfile native; "
                "addons is a shim (20-18): smearing_/project_mo_nr2nr native, convert_to_*/"
                "canonical_occ_/project_dm_k2k/... fall through to upstream addons.py; "
                "Rust newton_ah/stability/cphf/response modules exist (uncommitted working tree, "
                "another session) but are unbound; rsjk still refuses in Rust (20-06); "
                "kuhf_ksymm/kghf_ksymm are raising stubs, never served from upstream",
    },
    "dft": {
        "status": "native",
        "upstream": "pyscf/pbc/dft",
        "rust": ("pyscf-pbc-dft",),
        "bound_by": "20-13",
        "missing": ("cdft", "numint2c", "multigrid.multigrid", "multigrid.multigrid_pair",
                    "multigrid.utils", "multigrid.pp"),
        "names": {},
        "note": "KRKS/KUKS/KROKS/KGKS, ksymm, KRKSpU, grids, numint selectors native; the "
                "multigrid package is a shim (20-18) whose two public names MultiGridNumInt/"
                "MultiGridNumInt2 are native, its submodules upstream; KUKSpU "
                "and ksymm GGA on s-only bases refuse (20-13 D3/D4 crate defects); nlc/VV10 "
                "refused; all 19 upstream submodules fail to import under the overlay (20-13 D9)",
    },
    "symm": {
        "status": "partial",
        "upstream": "pyscf/pbc/symm",
        "rust": ("pyscf-pbc-symm",),
        "bound_by": "20-14",
        "missing": ("pyscf_spglib",),
        "names": {"pyscf_spglib": "upstream", "Symmetry": "upstream"},
        "note": "KPoints/SpaceGroup/make_kpts native; pyscf_spglib deliberately not ported "
                "(native detection); pyscf.pbc.symm.Symmetry stays upstream's class (20-14 D4)",
    },
    "lib": {
        "status": "partial",
        "upstream": "pyscf/pbc/lib",
        "rust": ("pyscf-pbc-lib", "pyscf-pbc-symm"),
        "bound_by": "20-14 / 20-17 shim",
        "missing": ("arnoldi", "linalg_helper", "chkfile", "ktensor",
                    "kpts_helper.members_with_wrap_around", "kpts_helper.conj_mapping",
                    "kpts_helper.get_kconserv_ria", "kpts_helper.loop_kkk",
                    "kpts_helper.round_to_fbz", "kpts_helper.KptsHelper"),
        "names": {"arnoldi": "upstream", "linalg_helper": "upstream", "chkfile": "upstream",
                  "ktensor": "upstream"},
        "note": "kpts_helper and kpts are overlay shims: native functions plus a lazy "
                "fallthrough to the upstream module for the unported names",
    },
    "tools": {
        "status": "partial",
        "upstream": "pyscf/pbc/tools",
        "rust": ("pyscf-pbc-tools", "pyscf-pbc-gto"),
        "bound_by": "20-14 / 20-17 shim",
        "missing": ("k2gamma", "lattice", "pyscf_ase", "pywannier90", "print_funcs", "tril",
                    "make_test_cell", "pbc.precompute_exx", "pbc.get_monkhorst_pack_size",
                    "pbc.get_lattice_Ls", "pbc.check_lattice_sum_range", "pbc.cutoff_to_gs",
                    "pbc.gs_to_cutoff", "pbc.round_to_cell0"),
        "names": {"k2gamma": "upstream", "lattice": "upstream", "pyscf_ase": "upstream",
                  "pywannier90": "upstream", "print_funcs": "upstream", "tril": "upstream",
                  "make_test_cell": "upstream"},
        "note": "fft/ifft/fftk/ifftk/get_coulG/madelung/super_cell/cell_plus_imgs/"
                "cutoff_to_mesh/mesh_to_cutoff native; the pbc.* names listed are ported in Rust "
                "but not bound (20-14 Task 6); make_test_cell is a feature-gated Rust fixture only",
    },
    "mp": {
        "status": "native",
        "upstream": "pyscf/pbc/mp",
        "rust": ("pyscf-pbc-mp",),
        "bound_by": "20-15",
        "missing": ("mp2",),
        "names": {"RMP2": "refused", "MP2": "refused", "UMP2": "refused", "GMP2": "refused",
                  "mp2": "upstream"},
        "note": "KMP2/KsymAdaptedKMP2/KUMP2/KMP2_stagger native; gamma-point RMP2/UMP2/GMP2 "
                "raise (upstream wraps the molecular MP2)",
    },
    "cc": {
        "status": "native",
        "upstream": "pyscf/pbc/cc",
        "rust": ("pyscf-pbc-cc",),
        "bound_by": "20-15",
        "missing": ("ccsd",),
        "names": {"RCCSD": "refused", "CCSD": "refused", "UCCSD": "refused",
                  "GCCSD": "refused", "ccsd": "upstream"},
        "note": "KRCCSD/KUCCSD/KGCCSD/KsymAdaptedRCCSD + (T) + EOM-IP/EA/EE native; gamma "
                "RCCSD/UCCSD/GCCSD raise; frozen= and EOM partition='mp'/'full' raise",
    },
    "ci": {
        "status": "partial",
        "upstream": "pyscf/pbc/ci",
        "rust": ("pyscf-pbc-ci",),
        "bound_by": "20-15",
        "missing": ("cisd",),
        "names": {"cisd": "upstream", "RCISD": "refused", "CISD": "refused",
                  "UCISD": "refused", "GCISD": "refused"},
        "note": "KCIS native; pbc/ci/cisd.py deferred by design (no molecular CI crate)",
    },
    "ao2mo": {
        "status": "native",
        "upstream": "pyscf/pbc/ao2mo",
        "rust": ("pyscf-pbc-ao2mo",),
        "bound_by": "20-15",
        "missing": (),
        "names": {"eris": "upstream"},
        "note": "the seven eris.py wrappers native (always complex, never s2-packed)",
    },
    # ── upstream: no overlay package; `import pyscf.pbc.<family>` is upstream Python ──
    "grad": {
        "status": "upstream", "upstream": "pyscf/pbc/grad", "rust": ("pyscf-pbc-grad",),
        "bound_by": None, "missing": ("*",), "names": {},
        "note": "Rust implementation in progress (Phase 18, uncommitted), unbound in Python",
    },
    "geomopt": {
        "status": "upstream", "upstream": "pyscf/pbc/geomopt", "rust": ("pyscf-pbc-geomopt",),
        "bound_by": None, "missing": ("*",), "names": {},
        "note": "13-line stub crate (Phase 18 planned)",
    },
    "tdscf": {
        "status": "upstream", "upstream": "pyscf/pbc/tdscf", "rust": ("pyscf-pbc-tdscf",),
        "bound_by": None, "missing": ("*",), "names": {}, "note": _P19,
    },
    "tddft": {
        "status": "upstream", "upstream": "pyscf/pbc/tddft", "rust": (),
        "bound_by": None, "missing": ("*",), "names": {},
        "note": "no crate; upstream pbc/tddft is a two-line alias of pbc/tdscf "
                "(Rust pyscf-pbc-tdscf exists, Phase 19, unbound)",
    },
    "gw": {
        "status": "upstream", "upstream": "pyscf/pbc/gw", "rust": ("pyscf-pbc-gw",),
        "bound_by": None, "missing": ("*",), "names": {}, "note": _P19,
    },
    "adc": {
        "status": "upstream", "upstream": "pyscf/pbc/adc", "rust": ("pyscf-pbc-adc",),
        "bound_by": None, "missing": ("*",), "names": {}, "note": _P19,
    },
    "x2c": {
        "status": "upstream", "upstream": "pyscf/pbc/x2c", "rust": ("pyscf-pbc-x2c",),
        "bound_by": None, "missing": ("*",), "names": {}, "note": _P19,
    },
    "eph": {
        "status": "upstream", "upstream": "pyscf/pbc/eph", "rust": ("pyscf-pbc-eph",),
        "bound_by": None, "missing": ("*",), "names": {}, "note": _P19,
    },
    "mpicc": {
        "status": "upstream", "upstream": "pyscf/pbc/mpicc", "rust": ("pyscf-pbc-mpi",),
        "bound_by": None, "missing": ("*",), "names": {},
        "note": "pyscf-pbc-mpi is a 13-line stub crate (20-CONTEXT §1.7)",
    },
    "mpitools": {
        "status": "upstream", "upstream": "pyscf/pbc/mpitools", "rust": ("pyscf-pbc-mpi",),
        "bound_by": None, "missing": ("*",), "names": {},
        "note": "pyscf-pbc-mpi is a 13-line stub crate (20-CONTEXT §1.7)",
    },
}

_STATUSES = ("native", "partial", "upstream")


class PbcUpstreamFallthroughWarning(UserWarning):
    """A ``pyscf.pbc`` import was served by upstream PySCF Python, not the Rust port."""


_WARNED: set = set()


def _normalise(name):
    if not isinstance(name, str) or not name:
        raise TypeError("which_impl(name) takes a non-empty str such as 'grad' or "
                        "'pyscf.pbc.scf.KRHF'")
    for prefix in ("pyscf.pbc.", "pbc."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    parts = name.split(".")
    if parts[0] not in FAMILIES:
        raise KeyError(f"{parts[0]!r} is not an upstream pyscf.pbc family "
                       f"(known: {', '.join(sorted(FAMILIES))})")
    return parts


def _overlay_path(parts):
    """Overlay file for `pyscf.pbc.<parts...>`, or None (no import performed)."""
    base = os.path.join(OVERLAY_DIR, *parts)
    for cand in (os.path.join(base, "__init__.py"), base + ".py"):
        if os.path.isfile(cand):
            return cand
    return None


def which_impl(name):
    """Which implementation serves ``name`` in this process's overlay.

    ``name`` is a family (``"grad"``, ``"pyscf.pbc.scf"``) or a dotted name below
    one (``"scf.KRHF"``, ``"pyscf.pbc.lib.kpts_helper.get_kconserv"``).

    Returns ``"native"`` (the Rust binding), ``"partial"`` (a family or shim
    module mixing native names with an upstream fallthrough) or ``"upstream"``
    (upstream PySCF Python — which may itself fail to import, see
    ``docs/pbc-status.md``). A dotted name the overlay binds as a raising
    stub returns ``"refused"``. Unknown families raise ``KeyError``.
    """
    parts = _normalise(name)
    family = parts[0]
    entry = FAMILIES[family]
    if len(parts) == 1:
        return entry["status"]
    rest = ".".join(parts[1:])
    if rest in entry["names"]:
        return entry["names"][rest]
    if entry["status"] == "upstream" or _overlay_path([family]) is None:
        return "upstream"
    modname = f"pyscf.pbc.{family}"
    for i, part in enumerate(parts[1:], start=1):
        mod = importlib.import_module(modname)
        if part in getattr(mod, "__all__", ()):
            return "native"
        if _overlay_path(parts[: i + 1]) is None:
            return entry["names"].get(".".join(parts[1: i + 1]), "upstream")
        modname = f"{modname}.{part}"
    return getattr(importlib.import_module(modname), "_PYSCF_RS_IMPL", "partial")


def _user_stacklevel():
    """``stacklevel`` (for a ``warnings.warn`` inside ``announce``) of the first frame
    outside importlib and this overlay. ``warnings`` itself skips the frozen
    ``importlib._bootstrap*`` frames when it counts, so those are walked but not counted.
    """
    level = 1
    frame = sys._getframe(2)  # the caller of announce()
    while frame is not None:
        fn = frame.f_code.co_filename
        if "importlib" in fn and "_bootstrap" in fn:
            frame = frame.f_back
            continue
        level += 1
        if not ("importlib" in fn or fn.startswith(OVERLAY_DIR)):
            return level
        frame = frame.f_back
    return 1


def _tree_of(origin):
    if origin is None:
        return "an unknown location"
    if "site-packages" in origin:
        return f"site-packages ({origin})"
    return origin


def announce(family, what, origin=None):
    """Emit the one-time passthrough warning for ``family`` (no-op after the first).

    ``what`` describes the import, e.g. ``"'pyscf.pbc.grad.krhf'"``.
    """
    if family in _WARNED:
        return
    _WARNED.add(family)
    entry = FAMILIES.get(family)
    status = entry["status"] if entry else "not in the pyscf-rs registry"
    note = f" ({entry['note']})" if entry else ""
    warnings.warn(
        f"pyscf-rs: pyscf.pbc.{family} is {status} — {what} is served by upstream "
        f"PySCF Python from {_tree_of(origin)}, not the Rust port{note}. "
        f"Check pyscf.pbc.which_impl({family!r}); many upstream modules fail to import under "
        f"the overlay (docs/pbc-status.md). Shown once per family.",
        PbcUpstreamFallthroughWarning,
        stacklevel=_user_stacklevel(),
    )


class _PassthroughAnnouncer:
    """`sys.meta_path` finder: announces, never resolves (always returns None)."""

    _busy = False

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if cls._busy or not fullname.startswith("pyscf.pbc.") or path is None:
            return None
        parts = fullname.split(".")
        family = parts[2]
        if family in _WARNED or parts[-1].startswith("_upstream_"):
            return None
        cls._busy = True
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            cls._busy = False
        if spec is None or spec.origin is None or spec.origin.startswith(OVERLAY_DIR):
            return None
        announce(family, repr(fullname), spec.origin)
        return None

    @classmethod
    def invalidate_caches(cls):
        return None


def install():
    """Put the announcer at the front of ``sys.meta_path`` (idempotent)."""
    for finder in sys.meta_path:
        if getattr(finder, "__name__", None) == "_PassthroughAnnouncer" or \
                type(finder).__name__ == "_PassthroughAnnouncer":
            return
    sys.meta_path.insert(0, _PassthroughAnnouncer)


def load_upstream_module(fullname, attr=None):
    """Load the upstream file an overlay module named ``fullname`` shadows.

    The module is executed under the private name ``<parent>._upstream_<leaf>``
    (so relative imports still resolve inside the parent package) and cached in
    ``sys.modules``. Announces the passthrough for the family.
    """
    parent_name, _, leaf = fullname.rpartition(".")
    private = f"{parent_name}._upstream_{leaf}"
    if private in sys.modules:
        return sys.modules[private]
    parent = importlib.import_module(parent_name)
    path = None
    for entry in getattr(parent, "__path__", ()):
        if os.path.abspath(entry).startswith(OVERLAY_DIR):
            continue
        for cand in (os.path.join(entry, leaf + ".py"), os.path.join(entry, leaf, "__init__.py")):
            if os.path.isfile(cand):
                path = cand
                break
        if path:
            break
    if path is None:
        raise ImportError(f"no upstream module shadowed by {fullname!r} on {parent_name}.__path__",
                          name=fullname)
    what = f"{fullname}.{attr}" if attr else fullname
    announce(fullname.split(".")[2], f"{what!r} (not bound; upstream module shadowed by the shim)",
             path)
    spec = importlib.util.spec_from_file_location(private, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[private] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(private, None)
        raise
    return mod


def fallthrough_getattr(module_name, upstream_fullname, submodules=()):
    """A PEP 562 ``__getattr__`` for an overlay shim: native names are module
    globals; ``submodules`` import by name; anything else is looked up on the
    upstream module ``upstream_fullname`` shadows (``load_upstream_module``).
    A failed upstream import becomes ``AttributeError`` so ``hasattr`` stays safe.
    """

    def __getattr__(name):
        if name.startswith("__"):
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        if name in submodules:
            return importlib.import_module(f"{module_name}.{name}")
        try:
            upstream = load_upstream_module(upstream_fullname, name)
        except Exception as exc:  # noqa: BLE001
            raise AttributeError(
                f"module {module_name!r} has no native attribute {name!r}, and the upstream "
                f"fallthrough {upstream_fullname} failed to import: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            return getattr(upstream, name)
        except AttributeError:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}") from None

    return __getattr__
