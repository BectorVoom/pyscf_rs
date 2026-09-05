#!/usr/bin/env python
"""16-03 cross-check — upstream `lib.davidson_nosym1` on the Rust port's own
test fixtures.

    PYTHONPATH=$PWD .venv/bin/python -u \
      .planning/phases/16-periodic-cc-ci/measurements/m7_davidson_ref.py

Plan 16-03 requires no PySCF oracle, and the shipped Rust tests consume none.
This script exists for a different reason: while writing those tests the port
was seen to STALL on a diagonally-dominant random matrix, converging to a
`1.7e-4` residual and then stopping with "linear dependency in trial subspace".
Rather than assume that was a port defect, upstream was run on the same
matrices. It stalls identically, to the last digits.

Two fixtures, both `n = 40`, `nroots = 1`, `tol = 1e-14`,
`tol_residual = 1e-12`, `max_space = 20`, unit-vector guess, diagonal
preconditioner:

  A. dense random (SplitMix64, seed 12345, amplitude 0.05, diag 1+2i)
  B. band, `a[i,j] = 0.05/d² + 0.0075i/d³`, diag 1+2i

and, for B, a coupling sweep showing where the METHOD (not either
implementation) picks the wrong root.
"""

import numpy as np

import pyscf

assert pyscf.__version__ == "2.12.1", pyscf.__version__

from pyscf.lib import linalg_helper as lh


class SplitMix64:
    """The PRNG the Rust tests use, so both sides see the same matrix."""

    def __init__(self, s):
        self.s = s & 0xFFFFFFFFFFFFFFFF

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def unit(self):
        return (self.next() >> 11) / float(1 << 53) - 0.5


def dense_random(n, seed, off):
    r = SplitMix64(seed)
    a = np.zeros((n, n), dtype=complex)
    for j in range(n):                       # column-major fill, as in Rust
        for i in range(n):
            a[i, j] = complex(off * r.unit(), off * r.unit())
    for i in range(n):
        a[i, i] = 1.0 + 2.0 * i
    return a


def band(n, off, slope=2.0):
    a = np.zeros((n, n), dtype=complex)
    for i in range(n):
        a[i, i] = 1.0 + slope * i
        for j in range(n):
            if i != j:
                d = abs(i - j)
                a[i, j] = complex(off / (d * d), 0.15 * off / (d * d * d))
    return a


def run(a, nroots, tol_residual, verbose=0):
    n = a.shape[0]
    diag = a.diagonal().real.copy()
    calls = [0]

    def aop(xs):
        calls[0] += 1
        return [a.dot(x) for x in np.asarray(xs)]

    def precond(dx, e, x0):
        d = diag - e
        d[abs(d) < 1e-12] = 1e-12
        return dx / d

    x0 = [np.eye(n, dtype=complex)[k] for k in range(nroots)]
    conv, e, v = lh.davidson_nosym1(
        aop, x0, precond, tol=1e-14, tol_residual=tol_residual,
        max_cycle=400, max_space=20, nroots=nroots, verbose=verbose,
    )
    resid = [float(np.linalg.norm(a.dot(v[k]) - e[k] * v[k])) for k in range(len(e))]
    return conv, e, resid, calls[0]


if __name__ == "__main__":
    print(f"pyscf {pyscf.__version__} at {pyscf.__file__}\n", flush=True)

    print("=== A. dense random, off = 0.05, n = 40, nroots = 1, "
          "tol_residual = 1e-12 ===")
    a = dense_random(40, 12345, 0.05)
    exact = np.sort(np.linalg.eigvals(a).real)
    conv, e, resid, calls = run(a, 1, 1e-12, verbose=5)
    print(f"  dense lowest {exact[0]!r}")
    print(f"  davidson     {e[0]!r}")
    print(f"  conv {list(conv)}  residual {resid[0]!r}  aop calls {calls}")
    print("  RUST PORT (crates/pyscf-algebra/src/davidson.rs) on the same "
          "matrix produced")
    print("  e = 1.0003286056665817, residual = 1.7050398774100944e-4, "
          "4 cycles, 4 aop calls;")
    print("  upstream's own trajectory is 0.131 -> 0.00171 -> 0.000171 -> "
          "0.000171 then")
    print("  'Linear dependency in trial subspace'. The stall is the METHOD, "
          "not the port.\n", flush=True)

    print("=== B. band fixture (the one the Rust tests ship), "
          "tol_residual = 1e-3 ===")
    for n in (40, 80):
        a = band(n, 0.05)
        exact = np.sort(np.linalg.eigvals(a).real)
        for nroots in (1, 3, 5):
            conv, e, resid, calls = run(a, nroots, 1e-3)
            err = max(abs(e[k] - exact[k]) for k in range(nroots))
            print(
                f"  n {n} nroots {nroots}: conv {all(conv)} "
                f"max|e - dense| {err:.2e} max residual {max(resid):.2e} "
                f"aop calls {calls}",
                flush=True,
            )

    print("\n=== C. coupling sweep — where the METHOD picks the wrong root ===")
    for off in (0.05, 0.2, 0.5):
        a = band(40, off)
        exact = np.sort(np.linalg.eigvals(a).real)
        conv, e, resid, _ = run(a, 1, 1e-12)
        print(
            f"  off {off}: davidson {e[0]!r} vs dense {exact[0]!r} "
            f"-> |err| {abs(e[0] - exact[0]):.2e}",
            flush=True,
        )
    print(
        "  At off >= 0.2 upstream converges to a spurious ~1e-15 eigenvalue "
        "with a\n  unit-vector guess and a plain diagonal preconditioner. The "
        "Rust port does the\n  same, which is why the shipped fixtures stay at "
        "off = 0.05.",
        flush=True,
    )
