"""Phase 3 BIND-04 stride-fuzz oracle (NumPy zero-copy boundary discipline).

Covers: BIND-04 — `a` / `a.T` / `a[::2]` / `a[:, 1:5]` all return identical answers
(Pitfall 5 mitigation: `is_standard_layout` → `to_owned` policy).
"""
import pytest


@pytest.mark.xfail(
    reason="Stride-fuzz a/a.T/a[::2]/a[:,1:5] assertion pending — plan 03-10",
    strict=False,
)
def test_stride_fuzz_get_veff_accepts_views():
    assert False, "Phase 3 plan 03-10 must implement BIND-04 stride-fuzz body"
