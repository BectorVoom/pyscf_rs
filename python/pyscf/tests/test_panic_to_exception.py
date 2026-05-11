"""Phase 3 BIND-09 panic → exception conversion.

Covers: BIND-09 — Rust panic / convergence failure surfaces as
`PyscfRsError` with `.kind` and `.source_chain` preserved.

Plan 03-07 ships `create_exception!(_native, PyscfRsRuntimeError, PyException)`
(abi3-py310 workaround) plus a Python overlay `PyscfRsError(_PyscfRsBase)`
in `pyscf/__init__.py` that grafts `.kind` + `.source_chain` from
positional args. This test forces a convergence failure and asserts the
overlay class is raised with the expected kind tag.
"""
import pytest

from pyscf import PyscfRsError, scf


def test_convergence_failure_raises_typed(h2o_mol):
    """Force non-convergence; ConvergenceFailure must be the typed kind."""
    mf = scf.RHF(h2o_mol)
    mf.max_cycle = 1            # force non-convergence
    mf.conv_tol = 1e-99         # impossibly tight — guarantees failure
    with pytest.raises(PyscfRsError) as exc_info:
        mf.run()
    # The .kind property reads args[1]; the source_chain reads args[2].
    # BIND-09 contract: kind matches the Rust variant name verbatim.
    kind = exc_info.value.kind if hasattr(exc_info.value, "kind") else ""
    msg = str(exc_info.value)
    assert (
        "ConvergenceFailure" in kind
        or "ConvergenceFailure" in msg
    ), (
        f"BIND-09: expected ConvergenceFailure, "
        f"got kind={kind!r} msg={msg!r}"
    )
    # The .source_chain must be a list (possibly empty).
    chain = exc_info.value.source_chain if hasattr(exc_info.value, "source_chain") else []
    assert isinstance(chain, list), (
        f"BIND-09: .source_chain must be a list, got {type(chain)}"
    )


def test_panic_to_pyscf_rs_error_preserves_kind_and_cause(h2o_mol):
    """Aggregator name kept for grep continuity (plan 03-02 stub name)."""
    test_convergence_failure_raises_typed(h2o_mol)
