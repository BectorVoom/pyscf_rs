"""Plan 20-17 — the edge of the PBC port: registry, `which_impl`, announced passthrough.

* `pyscf.pbc.which_impl` classifies all 21 upstream `pyscf/pbc/<family>` packages:
  the ten unported ones "upstream", the six partially covered ones "partial".
* The one-time `PbcUpstreamFallthroughWarning` fires once per family per
  process, names the family, and never fires for an overlay (native) import.
  Warning state is per process, so those cases run in a fresh interpreter.
* The 20-17 shims (`lib.kpts_helper`, `lib.kpts`, `tools`, `tools.pbc`,
  `dft.gen_grid`) re-export the SAME native objects.
"""
import json
import os
import subprocess
import sys
import textwrap

import pytest

import pyscf._native as native
from pyscf.pbc import which_impl

OVERLAY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

UPSTREAM = ["grad", "geomopt", "tdscf", "gw", "adc", "x2c", "eph", "mpicc", "mpitools", "tddft"]
PARTIAL = ["scf", "tools", "gto", "symm", "ci", "lib"]
NATIVE = ["df", "dft", "mp", "cc", "ao2mo"]


def _run_fresh(code, tmp_path):
    """Run `code` in a fresh interpreter with the overlay first on sys.path.

    cwd is `tmp_path`, never the repo root (from there cwd '' would put the
    vendored upstream `pyscf/__init__.py` ahead of the overlay).
    """
    env = dict(os.environ, PYTHONPATH=OVERLAY_ROOT)
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("__OUT__"))
    return json.loads(line[len("__OUT__"):])


@pytest.mark.parametrize("family", UPSTREAM)
def test_unported_family_is_upstream(family):
    assert which_impl(family) == "upstream"
    assert which_impl(f"pyscf.pbc.{family}") == "upstream"


@pytest.mark.parametrize("family", PARTIAL)
def test_partial_family(family):
    assert which_impl(family) == "partial"


@pytest.mark.parametrize("family", NATIVE)
def test_native_family(family):
    assert which_impl(family) == "native"


def test_registry_covers_every_upstream_family_exactly():
    from pyscf.pbc import FAMILIES

    assert sorted(FAMILIES) == sorted(UPSTREAM + PARTIAL + NATIVE)
    for family, entry in FAMILIES.items():
        assert entry["status"] in ("native", "partial", "upstream"), family
        assert entry["upstream"] == f"pyscf/pbc/{family}"
    for family in PARTIAL:
        assert FAMILIES[family]["missing"], family


@pytest.mark.parametrize("name,expected", [
    ("scf.KRHF", "native"),
    ("pyscf.pbc.dft.KRKS", "native"),
    ("mp.KMP2", "native"),
    ("scf.newton_ah", "upstream"),
    ("scf.kuhf_ksymm", "refused"),
    ("mp.RMP2", "refused"),
    ("ci.RCISD", "refused"),
    ("tools.k2gamma", "upstream"),
    ("tools.pbc", "partial"),
    ("tools.pbc.super_cell", "native"),
    ("lib.kpts_helper", "partial"),
    ("lib.kpts_helper.get_kconserv", "native"),
    ("lib.kpts_helper.loop_kkk", "upstream"),
    ("lib.arnoldi", "upstream"),
    ("gto.Cell", "native"),
    ("gto.ecp", "upstream"),
    ("symm.KPoints", "native"),
    ("symm.pyscf_spglib", "upstream"),
    ("dft.gen_grid.BeckeGrids", "native"),
    ("grad.krhf", "upstream"),
])
def test_which_impl_dotted(name, expected):
    assert which_impl(name) == expected


def test_which_impl_unknown_family_raises():
    with pytest.raises(KeyError):
        which_impl("not_a_family")


def test_shims_reexport_the_same_native_objects():
    from pyscf.pbc import dft, symm, tools
    from pyscf.pbc.lib import kpts, kpts_helper
    from pyscf.pbc.tools import pbc as tools_pbc

    assert kpts_helper.get_kconserv is native.pbc.lib.kpts_helper.get_kconserv
    assert kpts_helper.kk_adapted_iter is native.pbc.lib.kpts_helper.kk_adapted_iter
    assert kpts.KPoints is native.pbc.lib.kpts.KPoints is symm.KPoints
    assert tools.fft is native.pbc.tools.fft
    assert tools.get_coulG is native.pbc.tools.get_coulG
    assert tools_pbc.super_cell is native.pbc.tools.super_cell
    assert dft.gen_grid.BeckeGrids is native.pbc.dft.BeckeGrids
    assert dft.gen_grid.UniformGrids is native.pbc.dft.UniformGrids


def test_unknown_gto_attribute_is_attribute_error():
    from pyscf.pbc import gto

    assert not hasattr(gto, "certainly_not_a_pbc_gto_name")


WARN_PROBE = """
import importlib, json, warnings
from pyscf.pbc._unported import PbcUpstreamFallthroughWarning
out = {}
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    for mod in MODULES:
        try:
            importlib.import_module(mod)
        except Exception:
            pass
out["messages"] = [str(x.message) for x in w
                   if issubclass(x.category, PbcUpstreamFallthroughWarning)]
out["user_warning"] = all(issubclass(x.category, UserWarning) for x in w)
print("__OUT__" + json.dumps(out))
"""


def test_unported_family_warns_once_naming_the_family(tmp_path):
    code = "MODULES = ['pyscf.pbc.grad', 'pyscf.pbc.grad', 'pyscf.pbc.grad.krhf', " \
           "'pyscf.pbc.eph', 'pyscf.pbc.eph.eph_fd']\n" + WARN_PROBE
    out = _run_fresh(code, tmp_path)
    msgs = out["messages"]
    assert len(msgs) == 2, msgs
    assert "pyscf.pbc.grad is upstream" in msgs[0]
    assert "which_impl('grad')" in msgs[0]
    assert "pyscf.pbc.eph is upstream" in msgs[1]
    assert out["user_warning"]


def test_native_imports_do_not_warn(tmp_path):
    code = "MODULES = ['pyscf.pbc', 'pyscf.pbc.gto', 'pyscf.pbc.df', 'pyscf.pbc.scf', " \
           "'pyscf.pbc.dft', 'pyscf.pbc.dft.gen_grid', 'pyscf.pbc.symm', 'pyscf.pbc.lib', " \
           "'pyscf.pbc.lib.kpts_helper', 'pyscf.pbc.lib.kpts', 'pyscf.pbc.tools', " \
           "'pyscf.pbc.tools.pbc', 'pyscf.pbc.mp', 'pyscf.pbc.cc', 'pyscf.pbc.ci', " \
           "'pyscf.pbc.ao2mo', 'pyscf.pbc.scf.kuhf_ksymm']\n" + WARN_PROBE
    out = _run_fresh(code, tmp_path)
    assert out["messages"] == []


def test_shim_fallthrough_name_warns_once_for_the_family(tmp_path):
    code = """
    import json, warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from pyscf.pbc.lib.kpts_helper import get_kconserv  # native: silent
        n0 = len(w)
        from pyscf.pbc.lib.kpts_helper import loop_kkk      # upstream fallthrough
        from pyscf.pbc.lib.kpts_helper import KptsHelper    # same family: silent
    msgs = [str(x.message) for x in w]
    print("__OUT__" + json.dumps({"n0": n0, "msgs": msgs, "mod": loop_kkk.__module__}))
    """
    out = _run_fresh(code, tmp_path)
    assert out["n0"] == 0
    assert len(out["msgs"]) == 1, out["msgs"]
    assert "pyscf.pbc.lib is partial" in out["msgs"][0]
    assert "loop_kkk" in out["msgs"][0]
    assert out["mod"] == "pyscf.pbc.lib._upstream_kpts_helper"


def test_announcer_does_not_change_resolution(tmp_path):
    code = """
    import json, warnings
    warnings.simplefilter("ignore")
    import pyscf.pbc, pyscf.pbc.scf, pyscf.pbc.eph
    import pyscf._native as n
    print("__OUT__" + json.dumps({
        "scf": pyscf.pbc.scf.__file__,
        "eph": pyscf.pbc.eph.__file__,
        "identity": pyscf.pbc.scf.KRHF is n.pbc.scf.KRHF,
    }))
    """
    out = _run_fresh(code, tmp_path)
    assert out["scf"].startswith(os.path.join(OVERLAY_ROOT, "pyscf", "pbc"))
    assert not out["eph"].startswith(OVERLAY_ROOT)
    assert out["identity"]
