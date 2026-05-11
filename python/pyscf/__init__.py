"""pyscf-rs overlay package — Phase 3 BIND-02.

Until plan 03-07 ships the `_native` cdylib via `maturin develop`,
imports from `pyscf._native` raise ImportError; we catch + emit a clear
"pyscf-rs not yet built" message so test collection doesn't crash.
"""
try:
    from pyscf._native import scf  # type: ignore[attr-defined]
except ImportError:
    scf = None  # plan 03-07 fills this in

try:
    from pyscf._native import PyscfRsRuntimeError as _PyscfRsBase  # type: ignore[attr-defined]

    class PyscfRsError(_PyscfRsBase):  # type: ignore[misc, valid-type]
        """Phase 3 BIND-09 panic→exception with .kind and .source_chain attrs.

        Attributes:
            kind: Rust error variant name (e.g., 'ConvergenceFailure').
            source_chain: list of `str(err.source())` walking the Rust error tree.
        """

        @property
        def kind(self) -> str:
            return self.args[1] if len(self.args) > 1 else "Unknown"

        @property
        def source_chain(self) -> list[str]:
            return self.args[2] if len(self.args) > 2 else []
except ImportError:
    PyscfRsError = RuntimeError  # type: ignore[assignment,misc]  # placeholder until plan 03-07
