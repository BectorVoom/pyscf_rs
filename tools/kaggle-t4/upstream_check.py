#!/usr/bin/env python3
"""Re-run a finished pipeline result in upstream PySCF and compare.

    PYTHONPATH=<repo root> <repo>/.venv/bin/python upstream_check.py CKPT_DIR [--out FILE]

CKPT_DIR holds the `result.json` (and `history.jsonl`) a `yta7o19_bands` run
wrote, on any backend. The same cell (taken in Bohr from the result, so both
sides see the same geometry to the last bit), basis, cutoff, k-mesh and SCF
controls are run through upstream PySCF, stage by stage:

    pre  ->  project  ->  s1  ->  s2  ->  get_bands on the same k-points

and every stage is compared: the energy of each SCF cycle, the final energy,
the free energy, occupations, orbital energies on the k-mesh, and the band
energies and gap.

Upstream is run twice. `stock` is PySCF as a user would run it. `like` removes
the three places where the port's algorithm is known to differ from upstream's,
so that everything else is compared like for like:

* the non-local pseudopotential: the port evaluates it analytically
  (`pp_int.get_pp_nl`), FFTDF evaluates it in reciprocal space on the FFT mesh
  (`fft.py:get_pp`). The two agree at a converged mesh and differ at a low
  `ke_cutoff`; `like` gives upstream the analytic one.
* the level shift of a restricted k-point SCF: upstream passes the full density
  (`khf.py:157`), which lowers the occupied levels by the shift and raises the
  virtuals by it; the port raises the virtuals only. Upstream at HALF the shift
  is the port's Fock matrix minus `shift/2 * S`: the same orbitals and
  occupations, every orbital energy lower by `shift/2`.
* the final step: upstream re-diagonalises the un-shifted Fock matrix after
  convergence (`conv_check`); the port stops at the converged cycle.

Meant for small cells: upstream holds whole AO tables in memory.

Results written by a runner with the three fixes (method.pp_nonlocal ==
"reciprocal") need only the stock arm.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg

HARTREE_EV = 27.211386245988


def build_cell(res, basis):
    from pyscf.pbc import gto
    c = gto.Cell()
    c.unit = "Bohr"
    c.a = res["cell"]["a_bohr"]
    c.atom = [(sym, tuple(r)) for sym, r in res["cell"]["atoms_bohr"]]
    c.basis = basis
    c.pseudo = res["method"]["pseudo"]
    c.ke_cutoff = res["method"]["ke_cutoff_ha"]
    c.verbose = 0
    c.build()
    return c


def analytic_nl_hcore(mf):
    """`get_hcore` with FFTDF's local pseudopotential and the analytic non-local part."""
    from pyscf.pbc import tools
    from pyscf.pbc.dft import numint
    from pyscf.pbc.gto import pseudo
    from pyscf.pbc.gto.pseudo import pp_int

    def get_hcore(cell=None, kpts=None):
        cell = mf.cell if cell is None else cell
        kpts = np.asarray(mf.kpts if kpts is None else kpts).reshape(-1, 3)
        mesh = cell.mesh
        v_g = -np.einsum("ij,ij->j", cell.get_SI(mesh=mesh), pseudo.get_vlocG(cell, cell.get_Gv(mesh)))
        v_r = tools.ifft(v_g, mesh).real
        ao = numint.eval_ao_kpts(cell, mf.with_df.grids.coords, kpts)
        loc = np.asarray([a.conj().T @ (v_r[:, None] * a) for a in ao])
        return np.asarray(cell.pbc_intor("int1e_kin", kpts=kpts)) + loc + np.asarray(pp_int.get_pp_nl(cell, kpts))

    return get_hcore


def make_mf(cell, kpts, xc, controls, conv_tol, max_cycle, diis_space, like):
    from pyscf.dft import libxc
    from pyscf.pbc import dft
    mf = dft.KRKS(cell, kpts, xc=xc)
    mf._numint.libxc = libxc
    spec = controls.get("smearing")
    if spec and spec != "none":
        kind, sigma = spec.split(":")
        mf = mf.smearing(sigma=float(sigma), method={"gauss": "gauss", "gaussian": "gauss", "fermi": "fermi"}[kind])
    mf.damp = controls["damp"]
    mf.level_shift = controls["level_shift"] / 2 if like else controls["level_shift"]
    mf.diis_start_cycle = controls["diis_start"]
    mf.diis_space = diis_space
    mf.conv_tol = conv_tol
    mf.max_cycle = max_cycle
    mf.conv_check = False if like else controls.get("conv_check", True)
    if like:
        mf.get_hcore = analytic_nl_hcore(mf)
    return mf


def run_stage(mf, dm0, init_guess):
    traj = []
    mf.callback = lambda env: traj.append(float(env["e_tot"]))
    mf.init_guess = init_guess
    t = time.time()
    mf.kernel(dm0)
    return {"converged": bool(mf.converged), "cycles": len(traj), "trajectory": traj,
            "e_tot": float(mf.e_tot), "e_free": float(getattr(mf, "e_free", mf.e_tot)),
            "mo_energy": [np.asarray(e).tolist() for e in mf.mo_energy],
            "mo_occ": [np.asarray(o).tolist() for o in mf.mo_occ],
            "seconds": time.time() - t}


def project(cell1, dm1, cell2, s22, kpts):
    """`scf.addons.project_dm_nr2nr` per k-point: D2 = P D1 P^H, P = S22^-1 S21."""
    from pyscf.pbc import gto
    s21 = gto.intor_cross("int1e_ovlp", cell2, cell1, kpts=kpts)
    out = []
    for k in range(len(kpts)):
        p = scipy.linalg.solve(s22[k], s21[k], assume_a="her")
        out.append(p @ dm1[k] @ p.conj().T)
    return np.asarray(out)


def upstream(res, like):
    m = res["method"]
    cell = build_cell(res, m["basis"])
    assert list(map(int, cell.mesh)) == m["mesh"], f"mesh differs: upstream {cell.mesh}, port {m['mesh']}"
    kpts = cell.make_kpts(m["kmesh"])
    out = {}
    dm0 = None
    if m.get("pre_basis") and "scf_pre" in res:
        pre_cell = build_cell(res, m["pre_basis"])
        pre = make_mf(pre_cell, kpts, m["xc"], res["scf_pre"]["controls"], m["pre_conv_tol"],
                      m["max_cycle"], m["diis_space"], like)
        out["pre"] = run_stage(pre, None, m["init_guess"])
        s22 = make_mf(cell, kpts, m["xc"], res["scf_stage1"]["controls"], m["conv_tol"], m["max_cycle"],
                      m["diis_space"], like).get_ovlp()
        dm0 = project(pre_cell, pre.make_rdm1(), cell, s22, kpts)
        out["projection"] = {"electrons": float(np.einsum("kij,kji->", dm0, s22).real / len(kpts))}
    s1 = make_mf(cell, kpts, m["xc"], res["scf_stage1"]["controls"], m["conv_tol"], m["max_cycle"],
                 m["diis_space"], like)
    out["s1"] = run_stage(s1, dm0, m["init_guess"])
    final = s1
    if "scf_stage2" in res:
        s2 = make_mf(cell, kpts, m["xc"], res["scf_stage2"]["controls"], m["conv_tol"], m["max_cycle"],
                     m["diis_space"], like)
        out["s2"] = run_stage(s2, s1.make_rdm1(), m["init_guess"])
        final = s2
    band_k = cell.get_abs_kpts(np.asarray(res["bands"]["kpts_scaled"]))
    t = time.time()
    e_band = final.get_bands(band_k)[0]
    out["bands"] = {"energies": [np.asarray(e).tolist() for e in e_band], "seconds": time.time() - t}
    return out


def gap(bands, nocc):
    e = np.asarray(bands)
    vb, cb = e[:, nocc - 1], e[:, nocc]
    return {"gap_ev": float((cb.min() - vb.max()) * HARTREE_EV), "direct_gap_ev": float((cb - vb).min() * HARTREE_EV),
            "vbm_k": int(vb.argmax()), "cbm_k": int(cb.argmin())}


def compare(res, history, up, like):
    """Port (`res`, `history`) against one upstream arm."""
    nocc = res["method"]["nocc"]
    rows = {}
    for name, key in (("pre", "scf_pre"), ("s1", "scf_stage1"), ("s2", "scf_stage2")):
        if name not in up or key not in res:
            continue
        port, u = res[key], up[name]
        traj = [h["e_tot"] for h in history if h["stage"] == name and not h.get("final")]
        n = min(len(traj), len(u["trajectory"]))
        steps = [abs(a - b) for a, b in zip(traj[:n], u["trajectory"][:n])]
        row = {"cycles_port": port["cycles_total"], "cycles_upstream": u["cycles"],
               "converged_port": port["converged"], "converged_upstream": u["converged"],
               "e_tot_port": port["e_tot_ha"], "e_tot_upstream": u["e_tot"],
               "d_e_tot": abs(port["e_tot_ha"] - u["e_tot"]),
               "d_first_cycle": steps[0] if steps else None,
               "d_trajectory_max": max(steps) if steps else None}
        if port.get("e_free_ha") is not None:
            row["d_e_free"] = abs(port["e_free_ha"] - u["e_free"])
        if "kpts_mo_occ" in port:
            row["d_mo_occ_max"] = float(np.abs(np.asarray(port["kpts_mo_occ"]) - np.asarray(u["mo_occ"])).max())
        # `like` runs upstream at half the shift: its SCF-mesh orbital energies
        # are the port's minus shift/2 (see the module docs).
        offset = port["controls"]["level_shift"] / 2 if like else 0.0
        de = np.abs(np.asarray(port["kpts_mo_energy_ha"]) - np.asarray(u["mo_energy"]) - offset)
        row["d_mo_energy_occupied_max"] = float(de[:, :nocc].max())
        row["d_mo_energy_virtual_max"] = float(de[:, nocc:].max())
        rows[name] = row
    if "projection" in up and "projection" in res:
        rows["projection"] = {"electrons_port": res["projection"]["electrons"],
                              "electrons_upstream": up["projection"]["electrons"]}
    pb, ub = np.asarray(res["bands"]["energies_ha"]), np.asarray(up["bands"]["energies"])
    d = np.abs(pb - ub)
    rows["bands"] = {"k_points": int(pb.shape[0]), "bands": int(pb.shape[1]),
                     "d_all_max": float(d.max()), "d_occupied_max": float(d[:, :nocc].max()),
                     "d_edges_max": float(d[:, nocc - 1:nocc + 1].max()),
                     "port": gap(pb, nocc), "upstream": gap(ub, nocc)}
    rows["bands"]["d_gap_ev"] = abs(rows["bands"]["port"]["gap_ev"] - rows["bands"]["upstream"]["gap_ev"])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt_dir")
    ap.add_argument("--out", help="where to write the comparison (default CKPT_DIR/upstream_check.json)")
    ap.add_argument("--arms", default="stock", help="comma list of like,stock")
    args = ap.parse_args()
    d = Path(args.ckpt_dir)
    res = json.loads((d / "result.json").read_text())
    if "cell" not in res:
        sys.exit("result.json has no `cell` entry: it was written by a runner older than upstream_check.py")
    if not res.get("bands", {}).get("complete"):
        sys.exit("the run is not finished: result.json has no complete band set")
    history = [json.loads(line) for line in (d / "history.jsonl").read_text().splitlines() if line.strip()]

    import pyscf
    report = {"pyscf_version": pyscf.__version__, "pyscf_path": str(Path(pyscf.__file__).parent),
              "system": res["system"], "method": res["method"], "port_backend": res["backend"]}
    for arm in args.arms.split(","):
        t = time.time()
        up = upstream(res, like=(arm == "like"))
        report[arm] = compare(res, history, up, like=(arm == "like"))
        report[arm]["upstream_seconds"] = time.time() - t
        report[arm]["upstream_trajectories"] = {k: v["trajectory"] for k, v in up.items() if "trajectory" in v}
        print(f"== upstream arm `{arm}` ({report[arm]['upstream_seconds']:.0f} s)", file=sys.stderr)
        for stage, row in report[arm].items():
            if isinstance(row, dict) and stage != "upstream_trajectories":
                print(f"  {stage}: " + json.dumps(row), file=sys.stderr)
    out = Path(args.out) if args.out else d / "upstream_check.json"
    out.write_text(json.dumps(report, indent=1))
    print(out)


if __name__ == "__main__":
    main()
