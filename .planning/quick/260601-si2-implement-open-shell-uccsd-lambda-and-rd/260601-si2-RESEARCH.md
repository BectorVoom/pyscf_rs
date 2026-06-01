# F-07: Open-shell UCCSD Λ + RDM + wave-3 hooks — Research

**Researched:** 2026-06-01
**Domain:** Open-shell coupled-cluster response densities (port of PySCF `cc/*`)
**Confidence:** HIGH (formulation + reuse map are source-anchored; verification path mirrors the proven F-06 oracle)

---

## Executive Summary (5 lines)

1. **Formulation: port the SPIN-ORBITAL `gccsd_lambda.py` + `gccsd_rdm.py`, NOT the spin-block `uccsd_lambda.py`/`uccsd_rdm.py`.** The in-tree `uccsd.rs`/`uintermediates.rs` already solve UCCSD via a combined spin-orbital `<pq||rs>` representation (`SpinOrbitalEris`, `uccsd.rs:197-426`), so the single-tensor spin-orbital GCCSD λ/RDM equations are the analog the executor ports — clean 1:1 host-loops, no `ovOV`/`OVoo` mixed-spin block soup.
2. **Scope:** fill `ulambda.rs` (`make_intermediates`+`update_lambda`+`solve_lambda_so`, mirroring `lambda.rs` discipline) and `urdm.rs` (`_gamma1`/`_gamma2`/`make_rdm1`/`make_rdm2` + ao_repr), then wire `hooks.rs` `make_rdm1`/`make_rdm2` (currently `NotYetImplemented{wave:3}`) and add the PyUCCSD bridge entry points.
3. **One prerequisite refactor:** `uccsd_kernel` currently discards the converged spin-orbital `t1` and the `SpinOrbitalEris` (`UccsdResult` keeps only spin-block `t2aa/t2ab/t2bb`). Lambda/RDM consume the spin-orbital `(t1,t2,eris)`; the kernel must surface them.
4. **Verification:** a two-venv live-PySCF oracle (`ucc_open_shell_oracle.py`, mirroring `ump2_open_shell_oracle.py`) on an OH doublet (STO-3G, nocc_α≠nocc_β) comparing `make_rdm1` trace=nelec + λ/RDM byte-identity vs `pyscf.cc.UCCSD` to ≤1e-7; plus always-on in-tree invariants (Tr=nelec; α==β collapse to validated RCCSD λ/RDM).
5. **Biggest risk:** the spin-orbital↔spin-block index/antisymmetry mapping (the same trap that masked F-06: a transpose-equivalent bug is invisible when α==β) — the OH live oracle is the only sufficient gate; the α==β collapse is necessary-but-not-sufficient.

`cargo tree -p pyscf-ccsd` → **0 libxc rows** (verified this session). Test gates will NOT trigger the forbidden ~6h libxc build.

---

## 1. Formulation Decision (CRITICAL)

**The existing open-shell amplitude code is SPIN-ORBITAL internally, spin-block only at the boundary.**

Evidence:
- `uccsd.rs:1-36` module doc: *"the cleanest 1:1 host-loop port… is to assemble the combined spin-orbital antisymmetrized integrals `<pq||rs> = (pr|qs)−(ps|qr)` once… then run the compact spin-orbital CCSD intermediates."* It explicitly chose Stanton/Gauss/Watts/Bartlett spin-orbital CCSD over the production spin-block `uccsd.update_amps`.
- `uccsd.rs:197-426` `build_spin_orbital_eris` builds `SpinOrbitalEris` (`oovv/oooo/vvvv/ovvo/ooov/ovvv` in physicist `<pq||rs>` antisymmetrized form) over interleaved spin-orbital ranges (occ = [α-occ, β-occ], vir = [α-vir, β-vir]).
- `uccsd.rs:509-758` `update_amps_so` is the Stanton 1991 spin-orbital singles/doubles residual (single `t1`/`t2` tensors, `Fae`/`Fmi`/`Fme`/`Wvvvv`/`Woooo`/`Wovvo`/`tau` intermediates in `uintermediates.rs`).
- Spin-block `(t2aa,t2ab,t2bb)` is produced ONLY at the very end by `pack_amplitudes` (`uccsd.rs:787-840`) slicing the converged spin-orbital `t2` by spin label.

**Therefore: port the spin-orbital GCCSD path.** PySCF ships a dedicated spin-orbital module set that is the exact analog:
- `pyscf/cc/gccsd_lambda.py` (219 lines) — single-tensor `make_intermediates` (`v1/v2/v3/v4/v5/w3/woooo/wovvo/wovoo/wvvvo`, lines 34-99) + `update_lambda` (lines 103-168). **No spin-block soup** — exactly the `SpinOrbitalEris` blocks the executor already has.
- `pyscf/cc/gccsd_rdm.py` (297 lines) — `_gamma1_intermediates` (lines 26-43) + `_gamma2_intermediates` (lines 49-104) + `_make_rdm1` (lines 138-176) + `_make_rdm2` (lines 178-254), all single-tensor spin-orbital.

**Why NOT the spin-block `uccsd_lambda.py`/`uccsd_rdm.py`:** those carry ~6 distinct mixed-spin blocks per intermediate (`ovOV`, `OVoo`, `ooOO`, `goOvV`, `gvVvV`, …; see `uccsd_lambda.py:58-106`, `uccsd_rdm.py:266-315`). Porting them would mean re-deriving a *second*, parallel spin-block formulation that does not match the in-tree spin-orbital eris — wrong analog, far more surface, more bug area. They are mathematically equivalent to the GCCSD path the kernel already uses.

**Consistency guarantee:** the in-tree kernel doc (`uccsd.rs:25-29`) asserts the spin-orbital path *"reduces EXACTLY to the 06-03 RCCSD energy for a spin-symmetric (α==β) reference."* The same holds for spin-orbital λ/RDM → α==β must collapse to the validated closed-shell `lambda.rs`/`rdm.rs` results (an always-on in-tree cross-check).

| | In-tree UCCSD amplitudes | Upstream spin-block (`uccsd_*`) | Upstream spin-orbital (`gccsd_*`) ← **PORT THIS** |
|---|---|---|---|
| Representation | spin-orbital `<pq||rs>` | α/β/αβ blocks | spin-orbital `<pq||rs>` |
| Tensors per intermediate | 1 | ~3–6 | 1 |
| Matches `SpinOrbitalEris` | ✓ | ✗ | ✓ |

---

## 2. Exact Equations to Port (spin-orbital GCCSD → in-tree analog)

### 2a. `ulambda.rs` ← `gccsd_lambda.py`

| Upstream (`gccsd_lambda.py`) | In-tree analog to mirror | Notes |
|---|---|---|
| `make_intermediates` (l.34-99): `v1,v2,v3,v4,v5,w3,woooo,wovvo,wovoo,wvvvo` | `lambda.rs::LambdaImds::build` (l.98-122) discipline; intermediates live as a new `ULambdaImds` struct | All from `(t1,t2,SpinOrbitalEris)`. Reuse `uintermediates.rs` where blocks coincide (`make_tau`, `f_vv`/`f_oo`/`f_ov`, `w_oooo`, `w_ovvo`, `w_vvvv`); add the λ-only `v1/v2/v5/w3/wovoo/wvvvo`. |
| `update_lambda` (l.103-168) | `lambda.rs::update_lambda` (l.167-400) | Seed `l1=t1,l2=t2`; symmetrize via `tmp - tmp.transpose(1,0,2,3)` then `- .transpose(0,1,3,2)` (l.132-133,138,143) — the spin-orbital antisymmetrizer (NOT the closed-shell `(1,0,3,2)` symmetric form). Divide by `eia`/`eia+jb` (l.163-165). |
| `kernel` (l.27-32 → `ccsd_lambda.kernel`) | `lambda.rs::solve_lambda` (l.411-493) | Same arena (`wvvvv≈nv⁴` reserved once), dual-criterion `CONV_TOL_NORMT`, `MAX_CYCLE`. |

Key spin-orbital λ structure to honor (from `gccsd_lambda.py`):
- `v1 = imds.v1 - diag(mo_e_v)`, `v2 = imds.v2 - diag(mo_e_o)` (l.110-111) — the energy-diagonal removal exactly as `lambda.rs:107-111` does for `Loo`/`Lvv`.
- `mba = ½ einsum('klca,klcb->ba', l2, t2)`, `mij = ½ einsum('kicd,kjcd->ij', l2, t2)` (l.116-117).
- `m3` carries the `vvvv` tenant: `m3 += ½ einsum('ijcd,cdab->ijab', l2, eris.vvvv)` (l.125) — the `wvvvv` arena tenant.

### 2b. `urdm.rs` ← `gccsd_rdm.py`

| Upstream | In-tree analog | Notes |
|---|---|---|
| `_gamma1_intermediates` (l.26-43) → `(doo,dov,dvo,dvv)` | `rdm.rs::gamma1_intermediates`+`Gamma1` (l.94-189) | Spin-orbital: `doo=-einsum('ie,je->ij',l1,t1) - ½ einsum('imef,jmef->ij',l2,t2)`; `dvv`, `dvo`, `dov=l1`. Single tensor — simpler than the closed-shell `theta` form. |
| `_gamma2_intermediates` (l.49-104) → `(dovov,dvvvv,doooo,doovv,dovvo,dvvov,dovvv,dooov)` | new — heavier than closed-shell `rdm.rs::make_rdm2` step 2 | `miajb`, `goovv/gvvvv/goooo/gooov/govvo/govvv` then the `transpose` antisymmetrizations (l.92-104). |
| `_make_rdm1` (l.138-176) | `rdm.rs::make_rdm1` (l.202-282) | `dm1[:no,:no]=doo+doo.T`, `[:no,no:]=dov+dvo.T`, `[no:,no:]=dvv+dvv.T`, `*=.5`, `+1` on occ diag (l.155-161). **trace=nelec** invariant. ao_repr: `einsum('pi,ij,qj->pq', mo, dm1, mo)` (l.175). |
| `_make_rdm2` (l.178-254) | `rdm.rs::make_rdm2` (l.296-457) | Block placement (l.192-217) + the `with_dm1` separable correction (l.229-244) + the final `dm2.transpose(1,0,3,2)` (l.250) + ao_repr via `ccsd_rdm._rdm2_mo2ao` → in-tree `pyscf_ao2mo::general` (the exact `rdm.rs:433` path). |

**Spin-orbital → spin-block recovery for the PyO3/upstream-comparison boundary:** PySCF's `make_rdm1`/`make_rdm2` for UCCSD return spin-block tuples `(dm1a,dm1b)` / `(dm2aa,dm2ab,dm2bb)`. The in-tree path computes the spin-orbital `dm1`/`dm2`, then slices by the same spin labels `pack_amplitudes` uses (`uccsd.rs:787-840`: α-occ=0..no_a, β-occ=no_a..no, α-vir=0..nv_a, β-vir=nv_a..nv). The executor adds `pack_rdm1`/`pack_rdm2` helpers mirroring `pack_amplitudes`.

---

## 3. Reuse Map

### Consumes (already in-tree)
| Source | Item | Use |
|---|---|---|
| `uintermediates.rs:54-75` | `SpinOrbitalEris` blocks | λ `make_intermediates` + γ intermediates read these directly. |
| `uintermediates.rs` | `make_tau`, `f_vv/f_oo/f_ov`, `w_oooo`, `w_ovvo`, `w_vvvv`, `validate`, index helpers (`oovv_idx` … `t2_idx`) | Shared between amps and λ. |
| `uccsd.rs:787-840` | `pack_amplitudes` spin-label slicing | Template for `pack_rdm1`/`pack_rdm2`. |
| `reference.rs:42-47` | `UccsdReference{alpha,beta}` | The kernel input; λ/RDM ride the converged amplitudes, not the reference directly. |
| `lambda.rs` / `rdm.rs` | the entire host-loop+`oracle_sum`+arena discipline | Mirror per spin-orbital tensor. |
| `pyscf_ao2mo::general` | nmo⁴→nao⁴ back-transform | `make_rdm2(ao_repr)` — same call as `rdm.rs:433`. |

### MUST add / change
1. **`uccsd.rs` — surface the spin-orbital amplitudes + eris.** Currently `UccsdResult.amplitudes:UccsdAmplitudes` packs only `t2aa/t2ab/t2bb` (`uccsd.rs:60-76`); the converged spin-orbital `t1_cur`/`t2_cur` and `SpinOrbitalEris` are dropped at `uccsd_kernel` return (l.970-983). **λ/RDM need all three.** Add fields to `UccsdResult` (e.g. `so_t1: Vec<f64>`, `so_t2: Vec<f64>`, `so_eris: SpinOrbitalEris`, plus `no_a/nv_a/no_b/nv_b`), or expose a `rebuild_spin_orbital_eris` + cache. NOTE: `UccsdAmplitudes` has **no t1 at all** — even the spin-block boundary lost it.
2. **`ulambda.rs` public API** (new):
   ```rust
   pub struct ULambdaAmplitudes { pub l1: Vec<f64>, pub l2: Vec<f64>, pub converged: bool, pub niter: usize }
   pub fn solve_ulambda(t1:&[f64], t2:&[f64], eris:&SpinOrbitalEris, pool:&WorkspacePool) -> Result<ULambdaAmplitudes, PyscfRsError>;
   pub fn update_ulambda(t1,t2,l1,l2,&eris, wvvvv:&mut[f64]) -> Result<(Vec<f64>,Vec<f64>), CcsdError>;
   ```
3. **`urdm.rs` public API** (new):
   ```rust
   pub fn umake_rdm1(t1,t2,l1,l2,&eris, ao_repr:bool, &mo_coeff_a, &mo_coeff_b) -> Result<.., PyscfRsError>;
   pub fn umake_rdm2(t1,t2,l1,l2,&eris, ao_repr, .., pool:&WorkspacePool) -> Result<.., PyscfRsError>;
   ```
   (spin-orbital `mo_coeff` for ao_repr = α/β stacked; see `gccsd_rdm.py:281` `mo_a+mo_b`).
4. **`hooks.rs:65-76`** — `make_rdm1`/`make_rdm2` `NotYetImplemented{wave:3}` seams. These are on `CcsdOverrideHooks` (closed-shell `CcsdReference`). Open-shell RDM does NOT fit that signature (needs `UccsdReference` + spin-orbital amps). Recommend a **separate open-shell entry point** wired in the PyO3 bridge, NOT through `CcsdOverrideHooks` (which is RHF-shaped). Confirm with planner whether to add a `UccsdOverrideHooks` trait or route directly in `cc.rs`.
5. **`pyscf-py/src/cc.rs` PyUCCSD** — `solve_lambda`/`make_rdm1`/`make_rdm2` currently exist only on the closed-shell `PyRCCSD`-style path (cc.rs:434-489 use `solve_lambda`+`ccsd_make_rdm1`). PyUCCSD stores `me.amps = Some((t1,t2,nocc,nvir))` (cc.rs ~413) from the **spin-block** result — insufficient. Bridge must retain spin-orbital amps+eris and call `solve_ulambda`/`umake_rdm*`.

---

## 4. Verification Strategy

### 4a. Live-PySCF oracle (THE sufficient gate)
Mirror `crates/pyscf-py/tests/ump2_open_shell_oracle.py` (the F-06 template, proven 2.3e-10). New `crates/pyscf-py/tests/ucc_open_shell_oracle.py`:
- **System:** OH radical, `spin=1`, STO-3G (nocc_α=5, nocc_β=4 → nocc_α≠nocc_β; small enough for in-sandbox `cargo`/PySCF). Same molecule the F-06 oracle uses.
- **Stage 1 (`.upstream-venv`, PySCF 2.12.1):** `mf=scf.UHF(mol).run()`; `mycc=cc.UCCSD(mf).run()`; `l1,l2=mycc.solve_lambda()`; `dm1=mycc.make_rdm1()`; `dm2=mycc.make_rdm2()`. Dump `e_corr`, α/β `mo_coeff`, and the spin-block `dm1a/dm1b` (+ optionally `dm2ab` trace) to `.npz`.
- **Stage 2 (`.venv`, rs PyO3):** rebuild the mol, wrap the upstream-converged α/β MOs in a UHF shim, run `cc.UCCSD(shim).kernel()` then `.make_rdm1()`.
- **Quantities + tolerances (per `docs/rust_crate_test_guideline.md`):**
  - `e_corr` byte-identity ≤1e-7 (sanity — already validated by `uccsd_smoke.rs`).
  - **`make_rdm1` block byte-identity** `|dm1a_rs - dm1a_up|, |dm1b_rs - dm1b_up| ≤ 1e-7` ← the primary λ-correctness gate (RDM1 is the cheapest full λ-dependent observable).
  - **trace invariant** `Tr(dm1a)+Tr(dm1b)=nelec=9` for OH (always-on, oracle-free).
  - Optionally `make_rdm2` `dm2ab` slice ≤1e-7 (heavier; gate if memory permits).
- **Why OH and not α==β:** α==β masks any spin-orbital↔spin-block transpose bug (the F-06 lesson, AUDIT-FIX pass-5/7). nocc_α≠nocc_β forces a non-palindromic spin slice — the genuinely-exercising case.

### 4b. Always-on in-tree invariants (oracle-free, `cargo test -p pyscf-ccsd`)
1. **`umake_rdm1` trace = nelec** (mirror `rdm.rs:526-546` `make_rdm1_trace_equals_nelec`).
2. **α==β closed-shell collapse:** build a `UccsdReference` with `alpha==beta` (closed-shell), run `solve_ulambda`/`umake_rdm1` and assert ≈ the validated `solve_lambda`/`make_rdm1` on the same RHF reference (the spin-restricted reduction the module docs promise; mirror `uccsd_smoke.rs:82-138`).
3. **Thread invariance (RAYON 1==8 / Pitfall 2):** `update_ulambda` bit-identical across re-runs (mirror `lambda.rs:558-575`).
4. **ShapeMismatch not panic** on truncated block (mirror `lambda.rs:578-588`).
5. **λ symmetry:** spin-orbital `l2` antisymmetric `l2[i,j,a,b] = -l2[j,i,a,b] = -l2[i,j,b,a]` (the gccsd `__main__` self-check, `gccsd_lambda.py:198-199`).

### 4c. Does the PyO3 bridge already expose the entry points?
**No, not for open-shell.** PyUCCSD's `solve_lambda`/`make_rdm1`/`make_rdm2` (if present) route through the closed-shell `solve_lambda`+`ccsd_make_rdm1` (cc.rs:447,483) which take `ChemistsEris`/`CcsdReference` — wrong shape for spin-orbital. The bridge entry points for open-shell **must be added** (item 3.5 above). The closed-shell PyRCCSD pattern (cc.rs:434-489) is the template.

---

## 5. Pitfalls (the delicate parts)

1. **Algebra wall (HARD, confirmed `lambda.rs:14-18`, `uccsd.rs:36`, `uintermediates.rs:19-23`):** every contraction MATERIALIZES the contracted-axis products into a `Vec` then reduces via `pyscf_algebra::oracle_sum` (or `oracle_dot`). **NO bare `+=` across a contracted axis. NO `pyscf_algebra::gemm`** (it is a `NotYetImplemented{phase:2}` stub). This guarantees bit-exact + RAYON-1==8 invariance. The λ/RDM port must follow it verbatim — every `einsum` in gccsd → host loop + `oracle_sum`.
2. **Spin-orbital antisymmetrizer ≠ closed-shell symmetrizer.** `lambda.rs` symmetrizes with the closed-shell `tmp + tmp.transpose(1,0,3,2)` (l.295-380). The spin-orbital λ uses `tmp - tmp.transpose(1,0,2,3)` then `- tmp.transpose(0,1,3,2)` (`gccsd_lambda.py:132-133,138,143`) — a *signed* antisymmetrizer. Getting the sign/index pattern wrong is the most likely silent bug; pin it with invariant 4b.5.
3. **F-order vs C-order (the F-06 trap).** `pyscf_ao2mo::general` returns **F-order** `[n0,n1,n2,n3]` (element at `i+j*n0+k*n0*n1+l*n0*n1*n2`); in-tree tensors are **C-order**. `rdm.rs:418-431` reorders C→F before `general` and `rdm.rs:436-449` reorders the result F→C. The spin-orbital ao_repr back-transform must do the same. A transpose-equivalent bug is invisible when α==β (F-06 pass-5 lesson) → the OH oracle is the only catch.
4. **`UccsdResult` drops `t1`+`eris` (Section 3, item 1).** The single biggest *plumbing* gap — without it the kernel must rebuild `SpinOrbitalEris` (an `int2e`+`ao2mo` re-transform) just to solve λ. Prefer caching on the result.
5. **`wvvvv≈nv⁴` arena tenant.** λ `m3 += ½ l2·vvvv` (`gccsd_lambda.py:125`) and γ2 `gvvvv` are the heaviest buffers. Reserve the `nv⁴` scratch ONCE before the λ loop and reuse (Pitfall 20; `lambda.rs:444-447`, `uccsd.rs:939-942`). HARD `pool.try_reserve` pre-flight, NO downgrade (`rdm.rs:382-393`, T-06-06-OOM). Note: spin-orbital `nv = nv_a+nv_b` (double the per-spin virtual count) → nv⁴ is 16× a single spin channel — budget accordingly.
6. **Frozen-core masking.** `gccsd_rdm._make_rdm1`/`_make_rdm2` embed the active RDM into the full nmo via `get_frozen_mask` (l.163-171, 219-227). In-tree `rdm.rs` notes frozen embedding is a "CCSD-10 follow-on" (l.326-327) and operates active-only. Scope frozen embedding the same way (active-only this task) OR honor it explicitly — confirm with planner; the channel masking already exists (`uccsd.rs:169-185` `channel_counts` via `get_frozen_mask`).
7. **`l~t` canonical seed + dual-criterion convergence.** Seed `l1=t1,l2=t2` (`lambda.rs:449-451`); converge on `||Δl|| < CONV_TOL_NORMT` within `MAX_CYCLE` (the verified 06-03 constants, `lambda.rs:455-483`). Do not invent new tolerances.
8. **`make_rdm2` final transpose.** `gccsd_rdm._make_rdm2` returns `dm2.transpose(1,0,3,2)` (l.250) to put it in chemist-contractable order. Easy to omit — the energy cross-check `einsum('pqrs,pqrs',eri,dm2)*.5 + e1 == e_tot` (`gccsd_rdm.py:292-295`) catches it.

---

## 6. Open Questions (planner/discuss must resolve)

1. **Hook wiring shape:** route open-shell `make_rdm1`/`make_rdm2` through a new `UccsdOverrideHooks` trait, or directly in `cc.rs` PyUCCSD (bypassing the RHF-shaped `CcsdOverrideHooks` whose `wave:3` stubs at `hooks.rs:65-76` cannot carry `UccsdReference`)? **Recommend direct-in-bridge** (least surface; the trait override seam is a v1.x concern).
2. **Frozen-core embedding:** active-only (match `rdm.rs` CCSD-10 deferral) or full-nmo embed this task? Recommend **active-only**, documented, to match the closed-shell precedent.
3. **`make_rdm2` oracle depth:** gate on full `dm2` byte-identity or just `dm1` + the `Tr` invariant + the `e1==e_tot` energy reconstruction? Recommend **dm1 byte-identity + energy reconstruction** as the gate (dm2 byte-identity as a bonus assertion if OH/STO-3G fits the budget).

---

## 7. Project Constraints (from CLAUDE.md / memory)

- **NEVER trigger the libxc_rs ~6h build.** `cargo tree -p pyscf-ccsd` → 0 libxc rows (verified). Use `cargo +nightly test -p pyscf-ccsd --locked` (nightly: workspace manifest needs `profile-rustflags`; pyscf-ccsd's graph excludes libxc).
- **Save full Cargo output to `log/` before investigating build issues** (CLAUDE.md).
- **Test guideline:** read `docs/rust_crate_test_guideline.md` before writing tests; the oracle script must state specification source (PySCF 2.12.1), verified scope (OH/STO-3G open-shell λ/RDM), tolerance, and NOT claim unverified scope (larger bases where cintx ordering differs).
- **Algebra wall + `#![forbid(unsafe_code)]` + ShapeMismatch-not-panic** are non-negotiable (FOUND-06 / T-06-06-SHAPE).
- **Live PySCF oracle IS available in-sandbox** (`.upstream-venv` pyscf 2.12.1 + `.venv` PyO3 via maturin) — the two-venv `.npz` cross-compare is runnable (proven by F-06).

---

## Sources

**Primary (HIGH):**
- `pyscf/cc/gccsd_lambda.py` (l.34-168), `pyscf/cc/gccsd_rdm.py` (l.26-254) — the spin-orbital port targets.
- In-tree: `uccsd.rs` (l.1-983), `uintermediates.rs` (l.1-115), `lambda.rs` (l.1-589), `rdm.rs` (l.1-617), `reference.rs`, `hooks.rs`, `ulambda.rs`/`urdm.rs` stubs.
- `crates/pyscf-py/src/cc.rs` (PyUCCSD + closed-shell solve_lambda/make_rdm pattern), `crates/pyscf-py/tests/ump2_open_shell_oracle.py` (oracle template).
- `.planning/phases/06-ccsd/06-CONTEXT.md` D-03/D-05/D-09; `06-VERIFICATION.md` row 10; `.planning/AUDIT-FIX-2026-06-01.md` F-06/F-07.

**Verified this session:** `cargo tree -p pyscf-ccsd` → 0 libxc rows; spin-orbital formulation confirmed at `uccsd.rs:197-426,509-758`; gccsd spin-orbital modules exist (`ls pyscf/cc/gccsd_*`).

**Confidence:** Formulation HIGH (source-anchored). Reuse map HIGH. Verification path HIGH (mirrors proven F-06). The exact gccsd equation transcription is MEDIUM until the OH oracle runs — that is precisely what the gate is for.
