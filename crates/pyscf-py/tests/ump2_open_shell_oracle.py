"""Open-shell UMP2 / DF-UMP2 byte-identity vs live PySCF (F-06, plan Task 3).

THE acceptance gate for the cross-spin `(o_α v_α | o_β v_β)` ao2mo layout.

The closed-shell `UMP2 == RMP2` guard (`crates/pyscf-mp2/tests/ump2_cross_spin.rs`
Test B) is NECESSARY but NOT SUFFICIENT: when α == β the F-order→C-order swap is
absorbed by the palindromic shape, so a symmetric-equivalent transpose bug
survives it. This script is the sufficient gate — it runs a REAL open-shell
doublet (OH radical, spin=1, STO-3G) where nocc_α ≠ nocc_β and the αβ block has
distinct α/β orbitals, so the F→C repack is exercised for real.

Two-venv `.npy` cross-compare (the sandbox idiom):
  - `.upstream-venv` (pyscf 2.12.1, real C extensions) computes the reference
    UHF → UMP2 and UHF → DF-UMP2 correlation energies + dumps the converged
    α/β MO arrays + the molecule spec to a temp `.npz`.
  - `.venv` (the pyscf-rs PyO3 bridge) rebuilds the SAME molecule via the rs
    `gto.M`, wraps the upstream-converged α/β MO arrays in a minimal UHF-shaped
    shim, and runs the rs `mp.UMP2(shim).kernel()` (conventional, cross-spin
    `int2e`) and the DF-unrestricted path (`dfump2_kernel`).

Acceptance: |e_corr_rs − e_corr_upstream| <= 1e-9 for BOTH conventional UMP2 and
DF-UMP2. A larger delta means the F→C transpose (or the DF cross-spin wiring) is
wrong — do NOT approve; return to Task 1/2.

Per `docs/rust_crate_test_guideline.md`: the specification source is upstream
PySCF 2.12.1 (the byte-identity oracle); the verified scope is the open-shell
cross-spin layout; tolerance + molecule are stated explicitly; unverified scope
(larger basis sets where cintx ordering differs) is NOT claimed.

Run:
  REPO=/home/user/Documents/workspace/pyscf_rs
  $REPO/.upstream-venv/bin/python $REPO/crates/pyscf-py/tests/ump2_open_shell_oracle.py
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
ATOM = "O 0 0 0; H 0 0 0.97"
BASIS = "sto-3g"
SPIN = 1
TOL = 1e-9


def stage_upstream(npz_path: str) -> dict:
    """Run in `.upstream-venv`: reference UHF → UMP2 + DF-UMP2; dump arrays."""
    import numpy as np
    from pyscf import gto, scf, mp

    mol = gto.M(atom=ATOM, basis=BASIS, spin=SPIN)
    mf = scf.UHF(mol)
    e_scf = mf.kernel()
    assert mf.converged, "upstream UHF must converge"

    # Conventional UMP2.
    m = mp.UMP2(mf)
    m.kernel()
    e_corr_ump2 = float(m.e_corr)

    # DF-UMP2: density-fit the UHF reference, then UMP2.
    mf_df = mf.density_fit()
    mdf = mp.UMP2(mf_df)
    mdf.kernel()
    e_corr_dfump2 = float(mdf.e_corr)

    mo_coeff = np.asarray(mf.mo_coeff)  # [2, nao, nmo]
    mo_energy = np.asarray(mf.mo_energy)  # [2, nmo]
    mo_occ = np.asarray(mf.mo_occ)  # [2, nmo]

    np.savez(
        npz_path,
        mo_coeff=mo_coeff,
        mo_energy=mo_energy,
        mo_occ=mo_occ,
        e_tot=np.array([float(e_scf)]),
        e_corr_ump2=np.array([e_corr_ump2]),
        e_corr_dfump2=np.array([e_corr_dfump2]),
    )
    return {
        "e_scf": float(e_scf),
        "e_corr_ump2": e_corr_ump2,
        "e_corr_dfump2": e_corr_dfump2,
        "mo_coeff_shape": list(mo_coeff.shape),
    }


def stage_rs(npz_path: str) -> dict:
    """Run in the rs `.venv`: rebuild the molecule, wrap the upstream-converged
    UHF arrays in a UHF-shaped shim, run rs UMP2 (conventional + DF)."""
    import numpy as np
    from pyscf import gto
    from pyscf._native import mp as nmp

    data = np.load(npz_path)
    mo_coeff = data["mo_coeff"]
    mo_energy = data["mo_energy"]
    mo_occ = data["mo_occ"]
    e_tot = float(data["e_tot"][0])

    mol = gto.M(atom=ATOM, basis=BASIS, spin=SPIN)

    class UHFShim:
        """Minimal UHF-shaped reference the rs MP2 bridge snapshots from.

        Carries the upstream-converged α/β MO arrays + the rs mol. The bridge's
        `mf_is_uhf` falls back to a class-name check (contains 'UHF'); the
        snapshot reads `mo_coeff` (3-D [2,nao,nmo]), `mo_energy`/`mo_occ`
        (2-D [2,nmo]), `e_tot`, `mol`, `converged`."""

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
            # The rs bridge's extract_mole_from_pyany(mf) expects mf to carry a
            # serialisable Mole; delegate to the rs mol's canonical dumps().
            return self.mol.dumps()

    shim = UHFShim()

    # Conventional UMP2 (cross-spin int2e path).
    m = nmp.UMP2(shim)
    e_corr_ump2 = float(m.kernel())

    # DF-unrestricted path: a with_df UHF shim → dfump2_kernel.
    class UHFShimDF(UHFShim):
        def __init__(self):
            super().__init__()
            self.with_df = True

    shim_df = UHFShimDF()
    mdf = nmp.MP2(shim_df)  # factory: UHF+with_df → DF-unrestricted UMP2
    e_corr_dfump2 = float(mdf.kernel())

    return {
        "e_corr_ump2": e_corr_ump2,
        "e_corr_dfump2": e_corr_dfump2,
        "rs_class": type(m).__name__,
        "rs_df_class": type(mdf).__name__,
    }


def main() -> int:
    # Stage 2 entry: invoked with --rs <npz>.
    if len(sys.argv) >= 3 and sys.argv[1] == "--rs":
        out = stage_rs(sys.argv[2])
        print(json.dumps(out))
        return 0

    # Stage 1 (upstream) + orchestration.
    with tempfile.TemporaryDirectory() as td:
        npz = os.path.join(td, "uhf_ref.npz")
        up = stage_upstream(npz)
        print(f"[upstream] UHF e_scf      = {up['e_scf']:.12f}")
        print(f"[upstream] UMP2  e_corr   = {up['e_corr_ump2']:.12f}")
        print(f"[upstream] DFUMP2 e_corr  = {up['e_corr_dfump2']:.12f}")
        print(f"[upstream] mo_coeff shape = {up['mo_coeff_shape']}")

        # Stage 2: re-invoke the rs venv on the same .npz.
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
        print(f"[rs]       class          = {rs['rs_class']} / {rs['rs_df_class']}")
        print(f"[rs]       UMP2  e_corr   = {rs['e_corr_ump2']:.12f}")
        print(f"[rs]       DFUMP2 e_corr  = {rs['e_corr_dfump2']:.12f}")

        d_ump2 = abs(rs["e_corr_ump2"] - up["e_corr_ump2"])
        d_dfump2 = abs(rs["e_corr_dfump2"] - up["e_corr_dfump2"])
        print(f"[delta]    UMP2   |Δe_corr| = {d_ump2:.3e}  (tol {TOL:.0e})")
        print(f"[delta]    DFUMP2 |Δe_corr| = {d_dfump2:.3e}  (tol {TOL:.0e})")

        ok_ump2 = d_ump2 <= TOL
        ok_dfump2 = d_dfump2 <= TOL
        if ok_ump2 and ok_dfump2:
            print("RESULT: PASS — open-shell UMP2 AND DFUMP2 match live PySCF "
                  f"to <= {TOL:.0e}")
            return 0
        print("RESULT: FAIL — cross-spin layout / DF wiring incorrect "
              f"(UMP2 ok={ok_ump2}, DFUMP2 ok={ok_dfump2})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
