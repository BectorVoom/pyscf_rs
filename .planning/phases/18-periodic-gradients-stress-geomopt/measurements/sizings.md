# Plan 18-17 measurements — screening default, peak-RSS sizings, clause-5 fusion

**Method (all runs):** vendored PySCF 2.12.1, `PYTHONPATH=. .venv/bin/python`,
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. §9.2 reference cells from
`measurements/reference_cells.py` (diamond, si, lif, he_fcc, graphene),
gth-szv unless noted. Each heavy workload ran in a **fresh child process**;
peak RSS is the child's `ru_maxrss` via `os.wait4` (in-process deltas are
meaningless — `ru_maxrss` is monotonic and the ao-build phase sets an early
high-water mark). **No Rust file created or edited** (`git diff --stat --
crates/` empty, verified at close). No SIGKILL occurred on this machine
(31 GiB RAM); escalation stopped at ao-table builds for cost reasons (below),
so there is **no exit-137 ceiling to record**.

Mesh note: these cells build `mesh = [47, 47, 47]` (`ngrids = 103823`),
not `18-REVIEW §3.1`'s `ngrids = 24389`. All byte figures below are quoted at
the realised mesh; REVIEW's formulae are confirmed, its absolute MiB are
rescaled.

**Gamma-real correction (applies to every k-point AO table):**
`pyscf/pbc/gto/eval_gto.py:158-159` demotes the Γ-point slice to `float64`
(`if abs(kpt).sum() < 1e-9: v = v.real`). On a 2×2×2 mesh 1 of 8 k-slices is
real, so resident totals are `(7·16+1·8)/8/16 = 0.9375×` the all-complex
formula. Measured exactly (Task 2b). `ao2_kpts` in the untagged path is a
**view** into the same buffer (`fft_jk.py:347-349`) — confirmed, no doubling.

---

## Task 1 — `_contract_vhf_dm` screening default (`pbc/grad/rhf.py:30, :90-131`)

Direct calls of `_contract_vhf_dm` with a dense deterministic dm
(`A+A.T`, least-screenable input) and C-order `vhf (3,nao,nao)`, screen
True vs False, 5 timing reps each:

| cell | nao | natm | max\|de_s − de_u\| | t_screen | t_noscreen | ratio s/u |
|---|---|---|---|---|---|---|
| diamond | 8 | 2 | **0.0** | 0.387 ms | 0.023 ms | 16.5 |
| si | 8 | 2 | **0.0** | 0.341 ms | 0.063 ms | 5.4 |
| lif | 6 | 2 | **0.0** | 1.136 ms | 0.016 ms | 73.0 |
| he_fcc | 1 | 1 | **0.0** | 0.211 ms | 0.020 ms | 10.4 |
| graphene | 8 | 2 | **0.0** | 0.132 ms | 0.024 ms | 5.6 |

**Ruling for 18-10 / 18-18 Task 3:** keep upstream default
`SCREEN_VHF_DM_CONTRA = True`. The screening is **free and exact** on these
cells — bitwise-identical output (dense dm ⇒ nothing screened out at this
size). The wall ratio favouring unscreened is pure per-call
`build_neighbor_list_for_shlpairs` construction overhead dominating a
sub-ms contraction at `nao ≤ 8`; it is not the contraction cost and does not
transfer to production sizes. Both branches stay (neighbour list already
exists); the FD gate runs against the screened default.

## Task 2 — memory sizing vs `18-REVIEW §3.1`

Fixture: diamond, gth-szv, 2×2×2 (`nkpts = 8`, `nao = 8`, `natm = 2`,
`ngrids = 103823`), KRKS/PBE `max_cycle = 1`.

### 2a — `krks_stress.get_vxc` per grid block (D-PBC-30 clause 1)

Summed over the yielded k-list (Γ slice real), first block `blk = 97608`:

- `block_loop` AO: **950.5 MiB** (GGA `deriv+1 = 2`, comp 10)
- strain AO (`_eval_ao_strain_derivatives`, `deriv = 1`, 36 comps): **3421.9 MiB**
- block total realised: **4372.4 MiB**
- realised ratio to `block_loop`'s budget: **4.600** — the 4.6× GGA
  under-count (`18-REVIEW §3.2`) reproduces to 3 decimals.
- Full `krks_stress.get_vxc` (GGA): wall **37.0 s**, child peak RSS
  **4784.8 MiB**, `trace(out) = 0.143251` (sanity: symmetric, diagonally
  dominant 3×3).
- `block_loop` chose `blk = 97608` of `103823` grids (nearly full grid) from
  its under-counted budget — the realised 4.37 GiB block fits here but would
  OOM a constrained box. 18-12 must size blocks against the **sum** (clause 1
  stands, now with a realised number).

### 2b — `fft_jk.get_k_e1_kpts` (`fft_jk.py:346-350, :391`)

`ao1_kpts` build-only totals (per-k loop, Γ-corrected):

| basis | mesh | nkpts | nao | measured | all-complex theory |
|---|---|---|---|---|---|
| gth-szv | 2×2×2 | 8 | 8 | **0.371 GiB** | 0.396 GiB |
| gth-dzvp | 2×2×2 | 8 | 26 | **1.207 GiB** | 1.287 GiB |
| gth-dzvp | 3×3×3 | 27 | 26 | **4.264 GiB** | 4.344 GiB |

 Ratio measured/theory = 0.9375–0.98 (Γ-real slice; exactly 1 real slice in
 every mesh). REVIEW's `nkpts·4·ngrids·nao·16` formula is confirmed modulo
 this 0.94× factor. Note the build loop streams (child peak only
 434.7 MiB at dzvp 3×3×3); `get_k_e1_kpts` holds the whole list resident,
 so 4.26 GiB is the residency at dzvp 3×3×3.

Full `get_k_e1_kpts`, diamond szv 2×2×2, default `max_memory` (achieved
`blksize = 8`, verified by `prange` spy — single block):

| route | wall | child peak RSS | fp norm |
|---|---|---|---|
| untagged (`naoj = nao = 8`) | **173.2 s** | **1302.6 MiB** | 1.105745e+00 |
| MO-tagged (`naoj = nocc = 4`) | **90.5 s** | **1028.7 MiB** | 1.105745e+00 |

- Tagged is **1.91× faster** and **273.9 MiB** leaner; tagged-vs-untagged
  `max|dvk| = 2.2e-16` (exact to roundoff — clause 4a factorisation agreed).
- Inner `rho1` per block (blksize 8): untagged
  `3·8·8·103823·16` = **304.2 MiB**; tagged `3·8·4·103823·16` = **159.4 MiB**.
- `vR_dm (3,1,8,103823)` complex = **39.8 MiB** (REVIEW's 30 MiB was at
  `ngrids = 24389`; same formula).
- Tagged peak (1028.7 MiB) ≈ build-phase residency (`ao1` 398.5 + tagged
  `ao2` 47.5 + base): the inner loop never exceeds the build high-water
  mark when tagged — the §8.2 resident term is real and must be in 18-04
  Task 3's peak assertion.

### 2c — `krhf.hcore_generator` `eval_ao_kpts` call counts (clause 6/7)

Counting wrapper around `pbc/grad/krhf.py`'s `eval_ao_kpts` import,
`hcore_deriv(atm_id)` called per atom, diamond 2×2×2:
**16 calls = `natm·nkpts` = 2·8 — PASS.** The `natm×` full-grid AO
re-evaluation is confirmed; D-PBC-31 clause 7 (G-space contraction, 18-05)
removes it rather than hoisting it.

### 2d — inner buffer multiplicity, integer (D-PBC-31 clause 12)

Source (`fft_jk.py:391-396`): `rho1`+`vG` co-resident across the forward
FFT, `vG`+`vR` across the inverse ⇒ predicted integer **2**. Blksize-slope
experiment (identical code path, only `blksize` differs; peaks 757.6 /
757.7 MiB at achieved blksize 1, 845.8 MiB at achieved blksize 2;
rho-unit `3·1·8·103823·16` = 39.8 MiB):

- slope = 88.2 / 39.8 = **2.2 rho-units ⇒ integer 2**, with ~0.2 units
  (~8 MiB) FFT/transpose workspace. The transform runs effectively in
  place; there is no 3–4× blowup.
- **18-04 uses multiplicity 2** (plus small workspace epsilon), keeps the
  `/4 → /2` headroom of the re-derived `blksize`, drops only the doubled
  `mem_now` (`:362` then `:363`) as the defect — exactly the clause-12
  prescription.

### 2e — MO-tagged resident cost (both sides, `fft_jk.py:357-359`)

Diamond szv 2×2×2 (`nocc = 4` spatial): tagged `ao2` resident
`nkpts·nocc·ngrids·16` (Γ-corrected) = **47.5 MiB** beside the deriv-1
table (**398.5 MiB**, stays alive as `ao1_kpts`); transient inner `rho1`
cut 304.2 → 159.4 MiB per block at blksize 8. Overwhelmingly good trade,
peak assertion must cover the resident side (see 2b).

### Escalation log

2×2×2 gth-szv green ⇒ escalated ao-builds to gth-dzvp 2×2×2 and 3×3×3
(table 2b), all green, no SIGKILL. Full `get_k_e1_kpts` / `get_vxc` at
dzvp or 3×3×3 **not attempted** (projected ~35× the 173 s contraction —
cost-prohibitive, not a ceiling). No exit-137 ceiling recorded.

## Task 3 — clause-5 fusion split (one GGA grid block, `blk = 97608`)

Same fixture; dtype-matched no-op (1 Γ slice real, 7 complex):

| piece | min-of-3 wall |
|---|---|
| `_eval_ao_strain_derivatives` (36-comp strain + grid response inputs) | **22414 ms** |
| `ni.block_loop` AO eval (10-comp, same coords) | **10148 ms** |
| no-op alloc + write same `(8,3,3,4,blk,8)` mixed-dtype buffer (~3.4 GiB) | **839 ms** |
| arithmetic share of strain call | **21575 ms (96 %)**; traffic ≈ **4 %** |

**Verdict for 18-11: DO NOT FUSE on traffic grounds.** The repo's prior
("buffer traffic dominates collocation kernels") is **refuted for this
shape** — the strain kernel is exponential/radial-arithmetic dominated
(96 %). Fusing would save at most one ~0.8 s traffic pass over a ~32 s
combined cost; a fusion case built on shared-exponential savings is not
measured here and 18-11 ships the separate strain kernel. Number recorded
either way, as required.

---

## Verification

- Screening difference AND wall ratio recorded per cell (table Task 1),
  including the exact-0.0 outcome — "free and exact" is the result.
- Buffer multiplicity recorded as integer **2** (slope 2.2); 18-04 cites §2d.
- Clause-5 split recorded as two timings + **DO-NOT-FUSE** verdict; 18-11 cites §Task 3.
- No SIGKILL occurred; escalation boundary (ao-builds only at dzvp/3×3×3)
  stated in §Escalation log.
- `git diff --stat -- crates/`: empty (this plan wrote only this file;
  scratch scripts live in `/tmp/opencode/`, outside the repo).
