"""Open-shell UCCSD make_rdm1 byte-identity vs live PySCF (F-07, plan Task 6).

THE acceptance gate for the open-shell (spin-orbital GCCSD) Λ + RDM surface.

The in-tree α==β collapse (ulambda.rs / urdm.rs) is NECESSARY but NOT SUFFICIENT:
when α == β the spin-orbital↔spin-block transpose/antisymmetry is masked by the
palindromic spin slice, so a transpose-equivalent bug survives it (the F-06
lesson). This script is the sufficient gate — it runs a REAL open-shell doublet
(OH radical, spin=1, STO-3G) where nocc_α (=5) ≠ nocc_β (=4), so the spin-orbital
→ spin-block recovery and the SIGNED λ antisymmetrizer are exercised for real.

Two-venv `.npz` cross-compare (the sandbox idiom, mirroring ump2_open_shell_oracle):
  - `.upstream-venv` (pyscf 2.12.1, real C extensions) computes the reference
    UHF → UCCSD → solve_lambda → make_rdm1 (spin-block (dm1a, dm1b)) and dumps
    `e_corr`, the converged α/β MO arrays, and the molecule spec to a temp `.npz`.
  - `.venv` (the pyscf-rs PyO3 bridge) rebuilds the SAME molecule via the rs
    `gto.M`, wraps the upstream-converged α/β MO arrays in a minimal UHF-shaped
    shim, runs the rs `cc.UCCSD(shim).kernel()` then `.make_rdm1()`.

Acceptance gate (the spin-orbital λ-correctness gate, make_rdm1 ao_repr=False):
  - e_corr               |Δ| ≤ 1e-7   (sanity — already validated by uccsd_smoke)
  - |dm1a_rs − dm1a_up|  ≤ 1e-7   AND  |dm1b_rs − dm1b_up| ≤ 1e-7   ← THE gate
  - Tr(dm1a) + Tr(dm1b)  == 9     (OH nelec; oracle-free, always-on)

A larger dm1 delta means the spin-orbital antisymmetrizer sign/index (ulambda.rs,
Pitfall 2) or the spin-orbital→spin-block recovery (urdm.rs::pack_rdm1) is wrong.

make_rdm2 byte-identity is NOT gated this task (RESEARCH Open Question 3 — dm1
byte-identity is the gate). OPTIONAL BONUS: a make_rdm2 dm2ab slice is also
compared (non-gating) — the only thing that could catch a spin-block-SPECIFIC
AO-repack transpose the in-tree α==β round-trip (Task 3) cannot. It is a bonus,
not a gate.

Per `docs/rust_crate_test_guideline.md`: the specification source is upstream
PySCF 2.12.1 (the byte-identity oracle); the verified scope is the open-shell
spin-orbital λ/RDM surface on OH/STO-3G (SEGMENTED); tolerance + molecule are
stated explicitly; unverified scope (larger basis sets where cintx ordering
differs) is NOT claimed.

Run:
  REPO=/home/user/Documents/workspace/pyscf_rs
  $REPO/.upstream-venv/bin/python $REPO/crates/pyscf-py/tests/ucc_open_shell_oracle.py
(the script re-invokes the rs `.venv` itself for stage 2).
"""

import json
import os
import subprocess
import sys
import tempfile

REPO = "/home/user/Documents/workspace/pyscf_rs"
UPSTREAM_PY = f"{REPO}/.upstream-venv/bin/python"
RS_PY = f"{REPO}/.venv/bin/python"

# Real open-shell doublet: OH radical, spin=1 (one unpaired electron), STO-3G.
# nocc_α = 5, nocc_β = 4 → nocc_α ≠ nocc_β (the non-palindromic spin slice that
# EXERCISES the spin-orbital↔spin-block transpose; the SAME molecule the F-06
# oracle uses).
ATOM = "O 0 0 0; H 0 0 0.97"
BASIS = "sto-3g"
SPIN = 1
TOL = 1e-7
NELEC = 9


def stage_upstream(npz_path: str) -> dict:
    """Run in `.upstream-venv`: reference UHF → UCCSD → λ → make_rdm1; dump."""
    import numpy as np
    from pyscf import gto, scf, cc

    mol = gto.M(atom=ATOM, basis=BASIS, spin=SPIN)
    mf = scf.UHF(mol)
    e_scf = mf.kernel()
    assert mf.converged, "upstream UHF must converge"

    mycc = cc.UCCSD(mf)
    mycc.kernel()
    assert mycc.converged, "upstream UCCSD must converge"
    e_corr = float(mycc.e_corr)
    mycc.solve_lambda()
    dm1 = mycc.make_rdm1()  # spin-block (dm1a, dm1b)
    dm1a = np.asarray(dm1[0])
    dm1b = np.asarray(dm1[1])

    # Optional (non-gating) make_rdm2 dm2ab slice for the bonus assertion.
    dm2 = mycc.make_rdm2()  # (dm2aa, dm2ab, dm2bb)
    dm2ab = np.asarray(dm2[1])

    mo_coeff = np.asarray(mf.mo_coeff)  # [2, nao, nmo]
    mo_energy = np.asarray(mf.mo_energy)  # [2, nmo]
    mo_occ = np.asarray(mf.mo_occ)  # [2, nmo]

    np.savez(
        npz_path,
        mo_coeff=mo_coeff,
        mo_energy=mo_energy,
        mo_occ=mo_occ,
        e_tot=np.array([float(e_scf)]),
        e_corr=np.array([e_corr]),
        dm1a=dm1a,
        dm1b=dm1b,
        dm2ab=dm2ab.reshape(-1),
        dm2ab_shape=np.array(dm2ab.shape),
    )
    return {
        "e_scf": float(e_scf),
        "e_corr": e_corr,
        "tr_dm1a": float(np.trace(dm1a)),
        "tr_dm1b": float(np.trace(dm1b)),
        "mo_coeff_shape": list(mo_coeff.shape),
    }


def stage_rs(npz_path: str) -> dict:
    """Run in the rs `.venv`: rebuild the mol, wrap the upstream-converged UHF
    arrays in a UHF-shaped shim, run rs UCCSD → make_rdm1 / make_rdm2."""
    import numpy as np
    from pyscf import gto
    from pyscf._native import cc as ncc

    data = np.load(npz_path)
    mo_coeff = data["mo_coeff"]
    mo_energy = data["mo_energy"]
    mo_occ = data["mo_occ"]
    e_tot = float(data["e_tot"][0])

    mol = gto.M(atom=ATOM, basis=BASIS, spin=SPIN)

    class UHFShim:
        """Minimal UHF-shaped reference the rs UCCSD bridge snapshots from.

        Carries the upstream-converged α/β MO arrays (3-D [2,nao,nmo] mo_coeff,
        2-D [2,nmo] mo_energy/mo_occ) + the rs mol. The bridge's `mf_is_uhf`
        uses `istype('UHF')`; the snapshot reads the [2,...] α/β slices."""

        def __init__(self):
            self.mol = mol
            self.mo_coeff = mo_coeff
            self.mo_energy = mo_energy
            self.mo_occ = mo_occ
            self.e_tot = e_tot
            self.converged = True

        def istype(self, t):
            return t == "UHF"

        def dumps(self):
            return self.mol.dumps()

    shim = UHFShim()

    mycc = ncc.UCCSD(shim)
    e_corr = float(mycc.kernel())
    dm1a, dm1b = mycc.make_rdm1()
    dm1a = np.asarray(dm1a)
    dm1b = np.asarray(dm1b)
    # Optional bonus: dm2ab slice.
    dm2aa, dm2ab, dm2bb = mycc.make_rdm2()
    dm2ab = np.asarray(dm2ab)

    return {
        "e_corr": e_corr,
        "dm1a": dm1a.tolist(),
        "dm1b": dm1b.tolist(),
        "tr_dm1a": float(np.trace(dm1a)),
        "tr_dm1b": float(np.trace(dm1b)),
        "dm2ab_shape": list(dm2ab.shape),
        "dm2ab": dm2ab.reshape(-1).tolist(),
        "rs_class": type(mycc).__name__,
    }


def main() -> int:
    # Stage 2 entry: invoked with --rs <npz>.
    if len(sys.argv) >= 3 and sys.argv[1] == "--rs":
        out = stage_rs(sys.argv[2])
        print(json.dumps(out))
        return 0

    import numpy as np

    # Stage 1 (upstream) + orchestration.
    with tempfile.TemporaryDirectory() as td:
        npz = os.path.join(td, "uhf_ref.npz")
        up = stage_upstream(npz)
        print(f"[upstream] UHF   e_scf    = {up['e_scf']:.12f}")
        print(f"[upstream] UCCSD e_corr   = {up['e_corr']:.12f}")
        print(f"[upstream] Tr(dm1a)+Tr(dm1b) = {up['tr_dm1a'] + up['tr_dm1b']:.10f}")
        print(f"[upstream] mo_coeff shape = {up['mo_coeff_shape']}")

        res = subprocess.run(
            [RS_PY, os.path.abspath(__file__), "--rs", npz],
            capture_output=True,
            text=True,
            cwd=td,  # avoid the repo-root ./pyscf shadow
        )
        if res.returncode != 0:
            print("[rs] STDERR:\n" + res.stderr, file=sys.stderr)
            print("[rs] STDOUT:\n" + res.stdout, file=sys.stderr)
            print("RESULT: BLOCKED (rs stage failed)")
            return 2
        rs = json.loads(res.stdout.strip().splitlines()[-1])

        ref = np.load(npz)
        up_dm1a = ref["dm1a"]
        up_dm1b = ref["dm1b"]
        up_dm2ab = ref["dm2ab"].reshape(ref["dm2ab_shape"])

        rs_dm1a = np.asarray(rs["dm1a"])
        rs_dm1b = np.asarray(rs["dm1b"])
        rs_dm2ab = np.asarray(rs["dm2ab"]).reshape(rs["dm2ab_shape"])

        d_ecorr = abs(rs["e_corr"] - up["e_corr"])
        d_dm1a = float(np.abs(rs_dm1a - up_dm1a).max())
        d_dm1b = float(np.abs(rs_dm1b - up_dm1b).max())
        tr_rs = rs["tr_dm1a"] + rs["tr_dm1b"]

        print(f"[rs]       class         = {rs['rs_class']}")
        print(f"[rs]       UCCSD e_corr  = {rs['e_corr']:.12f}")
        print(f"[rs]       Tr(dm1a)+Tr(dm1b) = {tr_rs:.10f}")
        print(f"[delta]    e_corr   |Δ|  = {d_ecorr:.3e}  (tol {TOL:.0e})")
        print(f"[delta]    dm1a max|Δ|   = {d_dm1a:.3e}  (tol {TOL:.0e})")
        print(f"[delta]    dm1b max|Δ|   = {d_dm1b:.3e}  (tol {TOL:.0e})")

        # Optional (non-gating) make_rdm2 dm2ab slice.
        d_dm2ab = None
        if rs_dm2ab.shape == up_dm2ab.shape:
            d_dm2ab = float(np.abs(rs_dm2ab - up_dm2ab).max())
            print(f"[bonus]    dm2ab max|Δ| = {d_dm2ab:.3e}  (non-gating)")
        else:
            print(f"[bonus]    dm2ab shape mismatch rs={rs_dm2ab.shape} "
                  f"up={up_dm2ab.shape} (non-gating, skipped)")

        ok_ecorr = d_ecorr <= TOL
        ok_dm1 = d_dm1a <= TOL and d_dm1b <= TOL
        ok_trace = abs(tr_rs - NELEC) < 1e-6

        if ok_ecorr and ok_dm1 and ok_trace:
            print(
                f"RESULT: PASS (make_rdm1 byte-identity gate) — open-shell UCCSD "
                f"make_rdm1 matches live PySCF to <= {TOL:.0e}, Tr=9"
            )
            if d_dm2ab is not None and d_dm2ab > TOL:
                print(f"        NOTE: dm2ab bonus |Δ|={d_dm2ab:.3e} > {TOL:.0e} "
                      "(non-gating — make_rdm2 byte-identity is out of scope this "
                      "task; the gate is make_rdm1 + Tr=9).")
            return 0

        print("RESULT: FAIL — open-shell make_rdm1 gate not met "
              f"(e_corr |Δ|={d_ecorr:.3e}, dm1a |Δ|={d_dm1a:.3e}, "
              f"dm1b |Δ|={d_dm1b:.3e}, Tr={tr_rs:.6f})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
