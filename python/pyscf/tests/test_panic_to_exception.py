"""Phase 3 BIND-09 panic → exception conversion (`PyscfRsError(kind='ConvergenceFailure')`).

Covers: BIND-09 — Rust panic / convergence failure surfaces as PyscfRsError with
`.kind` and `__cause__` chain preserved (Pitfall 14 mitigation).
"""
import pytest


@pytest.mark.xfail(
    reason="ConvergenceFailure → PyscfRsError(kind='ConvergenceFailure') assertion pending — plan 03-10",
    strict=False,
)
def test_panic_to_pyscf_rs_error_preserves_kind_and_cause():
    assert False, "Phase 3 plan 03-10 must implement BIND-09 panic→exception assertion"
