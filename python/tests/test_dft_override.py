"""Wave-0 scaffold — DFT-08: subclass-override dispatch.

A Python subclass of ``pyscf.dft.RKS`` that overrides ``get_veff`` and calls
``define_xc_`` must have those overrides invoked every SCF cycle (the PyO3
``slf.call_method1`` subclass-override dispatch seam established in Phase 3 for
RHF/UHF/GHF, BIND-04).

Upstream reference: ``pyscf/dft/rks.py`` get_veff + ``pyscf/dft/libxc.py``
define_xc_. Owning plan: 04-08 (DFT PyO3 override surface) unskips this and
fills the per-cycle dispatch assertion. Built via maturin before running.
Verify command: ``pytest python/tests/test_dft_override.py`` (maturin build).
"""
import pytest

pytestmark = pytest.mark.skip(reason="Phase 4 04-08: DFT subclass get_veff/define_xc_ override dispatch — unskip when the DFT PyO3 surface lands")


def test_dft_subclass_get_veff_override_invoked_every_cycle():
    # 04-08 fills: define a Python subclass of dft.RKS overriding get_veff (and
    # using define_xc_), run the SCF, and assert the override is invoked on
    # every cycle via the slf.call_method1 dispatch seam (BIND-04 precedent).
    raise NotImplementedError(
        "04-08 fills the DFT subclass-override per-cycle dispatch assertion"
    )
