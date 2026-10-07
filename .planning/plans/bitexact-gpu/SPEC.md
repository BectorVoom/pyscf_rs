---
title: Bit-exact GPU remediation (CUDA/WGPU/ROCm vs PySCF reference)
status: draft
format: markdown
spec_version: 1
updated_at: 2026-09-19T00:00:00Z
source_requirements:
  - ORACLE-07 (.planning/REQUIREMENTS.md:169)
  - ROADMAP Phase 8 success criterion 1 (.planning/ROADMAP.md:364)
  - measurements README §3 bitwise rule (.planning/phases/20-pbc-python-bindings/measurements/README.md:80-103)
---

# SPEC — Bit-exact GPU remediation

## 1. Context

`pyscf_rs` computes PBC band structures / DFT energies that agree with the
vendored PySCF 2.12.1 CPU reference to ~1e-11 Ha (`e_tot`) and ~1e-9
(`mo_energy`, `get_bands`) on a Y-load surrogate (YN rocksalt, KRKS/LDA,
mesh 41³, `conv_tol` 1e-12) — but never 0-ulp bitwise
([INFERRED: prior-session measurements in `/tmp/opencode/yn_*.json`; not
committed to this repo]).
Repo policy already states GPU backends are *not* bit-exact by design:

- [VERIFIED: LOCAL .planning/REQUIREMENTS.md:169]
  `ORACLE-07: GPU backends (CUDA/WGPU/ROCm) are tested at chemical accuracy,
  not bit-exact; tolerance documented per backend` — status Pending
  (`REQUIREMENTS.md:360`).
- [VERIFIED: LOCAL .planning/ROADMAP.md:364]
  `CPU: bit-exact under oracle profile; CUDA: 1e-10 Hartree energy / 1e-8
  gradient; WGPU: chemical accuracy 1e-6; ROCm: 1e-10`.
- [VERIFIED: LOCAL .planning/phases/20-pbc-python-bindings/measurements/README.md:80-103]
  §3 bitwise rule: `to_bits()` assertions are for same-implementation A/B
  only; no gate tighter than 1 ulp of the bounded quantity. Best observed
  anywhere: 221 ulp.

So "fixing" here means: **attribute every GPU-vs-reference ulp to a named
mechanism, eliminate the cheap ones, pin the rest behind per-backend
regression gates, and close ORACLE-07** — not "make CUDA 0-ulp".

> **Correction (2026-09-19, user directive): xcfun is excluded from this
> workstream in every form (`xcfun_rs`, `xcfun_gpu`).** All comparison
> quantities are HF-level analytic integrals — **libcint** on the PySCF side
> (`pyscf/gto/mole.py:21`, `pyscf/gto/moleintor.py:20`) vs **cintx** on the
> pyscf_rs side (`crates/pyscf-gto/Cargo.toml:13-22`, root `Cargo.toml:171`
> path patch). The comparison vehicle is therefore **KRHF + `get_bands`**
> (native binding at `crates/pyscf-py/src/pbc/scf.rs:1761`, via
> `Krhf::get_bands:106`), never KRKS/XC. The KRKS/LDA numbers above remain
> valid background; gates will be set on KRHF.

## 2. Scope and non-goals

In scope: CUDA (T4), WGPU (shader-f64 adapters), ROCm paths for PBC
**KRHF + `get_bands`** (HF-level analytic integrals via libcint/cintx; no XC
functional evaluation on either side); host reduction geometry; device FMA
audit; dm-anchored comparison methodology; CI gates.

Non-goals:

- 0-ulp CUDA/WGPU acceptance gates (contradicts ORACLE-07; stays rejected).
- **xcfun in any form (`xcfun_rs`, `xcfun_gpu`) — excluded by directive;
  no XC-substrate work, no libxc-vs-xcfun unification.**
- A device FFT kernel (absence verified, `pyscf-pbc-tools/src/fft_kernel.rs:1-23`).
- Changing CPU/oracle-profile bit-exactness behavior (must not regress).
- Silent-f32 fallbacks (DFT-11 honesty stays: warn + CPU-f64).

## 3. Dependencies

- cubecl 0.10.0 lockstep + cubek-matmul/reduce 0.2.0
  [VERIFIED: LOCAL Cargo.toml:60-75]; per-backend features
  `cpu/cuda/wgpu/rocm` forwarded `pyscf-runtime → pyscf-algebra →
  pyscf-kernels` [VERIFIED: LOCAL crates/pyscf-kernels/Cargo.toml:26-32].
- `PYSCF_BACKEND` resolver [VERIFIED: LOCAL crates/pyscf-runtime/src/backend.rs:9-74,
  crates/pyscf-algebra/src/select.rs:23-52]; `dispatch_backend!`
  [VERIFIED: LOCAL crates/pyscf-algebra/src/dispatch.rs:53-82].
- Host FMA-freedom: `.cargo/config.toml:14-24` (`-fp-contract=off`,
  `-fma,-fma4`); enforced for host symbols by `xtask check-no-fma`
  [VERIFIED: LOCAL xtask/src/bin/check_no_fma.rs:39-100].
- Ordered host reductions `oracle_sum/oracle_dot`, `PAIRWISE_CHUNK=128`
  [VERIFIED: LOCAL crates/pyscf-algebra/src/oracle.rs:1-94].
- Sibling integral tree: **cintx** (path dep, root `Cargo.toml:171`;
  `pyscf-gto` via `cintx-{core,compat,rs,ops,runtime}`
  [VERIFIED: LOCAL crates/pyscf-gto/Cargo.toml:13-22]) against upstream
  **libcint** (`pyscf/gto/mole.py:21`, `pyscf/gto/moleintor.py:20`).
  cintx pins libcint byte-identity at source for shipped operators
  (e.g. `crates/pyscf-gto/tests/int2e_arity4.rs:12`,
  `int3c2e_auxmol.rs:79`) — the integral floor starts near 0 ulp.
- Hardware: CUDA T4 required for specs S3–S4 acceptance; CPU-only work
  (S1, S2-partial, S5, S6-partial, S7-scaffolding) proceeds without it.

## 4. Typed contracts

- `FROZEN` fixture: `(mesh: [u64;3], kpts: Vec<[f64;3]>, dm: KDms, shape_id: str)`
  with a committed SHA-256 hex digest; every census/micro-test consumes
  `FROZEN` and emits `(max_abs: f64, max_ulp: u64, n: usize)`.
- `BackendReport { backend: BackendKind, e_tot_hex: String,
  mo_hex: Vec<String>, bands_hex: Vec<String>, mesh, conv_tol }` — the
  comparison artifact; `*_hex` via `f64::to_bits` formatting. (No XC
  fields — KRHF vehicle.)
- Error type: reuse `PyscfRsError`; new failures surface as data
  (report rows), never silent fallback.

## 5. Failure-isolated behavioral specifications

### S1 — dm-anchored band comparison harness [draft]

- Rationale: separates SCF fixed-point depth (tolerance artifact) from
  kernel bugs. Prior evidence: `e_tot` 1e-11 coexisting with bands 1e-9.
- Preconditions: converged reference dm committed as `FROZEN` + digest.
- Input: `FROZEN`, `backend: BackendKind`, `mesh` (no XC — KRHF vehicle).
- Output: one-shot `energy_elec` + `get_bands` on the frozen dm (no SCF
  iteration; native `kernel(dm0=…)` support exists
  [VERIFIED: CODEGRAPH crates/pyscf-py/src/scf.rs:319-334 `InitGuessMode::UserDM`]).
- Dependencies: `pyscf-py` PBC bindings (`KRHF.kernel(dm0)`, `get_bands`
  [VERIFIED: LOCAL crates/pyscf-py/src/pbc/scf.rs:1761-1762]).
- Given a frozen dm + identical mesh When one-shot energy+bands run on
  CPU vs CUDA Then per-quantity `max_ulp` is reported; SCF-depth
  attribution is impossible to confuse with kernel error.
- Invariants: harness never iterates SCF; never mutates the frozen dm.
- Acceptance: harness reproduces the known ordering
  (`e_tot` gap ≪ bands gap shrinks under `conv_tol` sweep 1e-8→1e-12 on
  CPU/CPU control) before any GPU claim is made.
- Out of scope: fixing any kernel; any XC functional.
- Traceability: `kscf.rs` convergence semantics; `Krhf::get_bands`
  reuses the converged dm — one-shot by construction
  [VERIFIED: LOCAL crates/pyscf-py/src/pbc/scf.rs:1762].
- Open: exact `FROZEN` system (default: YN surrogate, KRHF, mesh 41³).

### S2 — frozen-input device-vs-host reduce/dot census [draft]

- Rationale: quantify breakage point #1 (device-dependent partial count).
- Preconditions: `FROZEN` vectors covering YN/KRHF mesh-41³ shapes.
- Input: fixed `&[f64]`, backend.
- Output: `(partials_count, max_ulp_vs_oracle_sum)` per backend.
- Dependencies: `launch_reduce_sum`
  [VERIFIED: CODEGRAPH crates/pyscf-algebra/src/reduce.rs:69-114],
  `launch_dot`, `launch.rs` geometry fns.
- Given frozen inputs When reduced on CPU/CUDA/WGPU/ROCm Then partial
  counts and bit distances are tabulated; host tail fold stays sequential.
- Invariants: no kernel math change; census only.
- Acceptance: table shows whether partial count varies by backend and the
  resulting ulp distance on identical inputs.
- Out of scope: changing the reduction (see S3).
- Traceability: `reduce.rs:76-82` (`stride = total_units(&count, dim)`).
- Open: T4 + ROCm hardware access for the CUDA/ROCm columns.

### S3 — device-independent (fixed-lane) reduction [draft]

- Rationale: eliminate breakage point #1 if S2 shows backend-dependent
  partial counts with nonzero ulp distance.
- Preconditions: S2 census complete; CPU oracle behavior pinned by
  `oracle_determinism` tests.
- Input: same as `launch_reduce_sum`/`launch_dot`.
- Output: identical partial tree (fixed lane count) on every backend;
  CPU bit-identity preserved.
- Dependencies: `reduce.rs`, `dot.rs`, `launch.rs`; wall rule (no new
  cubecl deps outside algebra/kernels).
- Given frozen inputs When reduced on any backend Then partial count is
  backend-independent and host-vs-device distance is 0 ulp where the
  kernel math is exact (pure sum; FMA-affected dots excluded — see S4).
- Invariants: `PAIRWISE_CHUNK` untouched; CPU numerical results unchanged
  (existing oracle tests must pass unmodified).
- Acceptance: S2 census re-run shows equal partial counts + 0-ulp pure
  sums on all available backends; full workspace CPU test suite green.
- Out of scope: GEMM/dot-product FMA (S4); integral-layer gaps owned by
  cintx workstreams (S5).
- Traceability: `launch_1d`, `line_size_for`, `reduction_lanes`.
- Open: performance cost of fixed lanes on T4 (measure, don't assume).

### S4 — device FMA audit [draft]

- Rationale: breakage point #2; host `check-no-fma` is blind to PTX/SASS
  [VERIFIED: LOCAL xtask/src/bin/check_no_fma.rs:39-73, x86/aarch64
  mnemonics only].
- Preconditions: T4 hardware; list of kernels under audit
  (`reduce_kernel`, `dot_kernel`, GEMM inner loops, eval-GTO radial sums).
- Input: compiled kernel artifacts for the CUDA target.
- Output: report of `fma.rn.f64`/DFMA presence per kernel + fixed-fixture
  host-vs-device dot/GEMM micro-test table.
- Dependencies: cubecl 0.10.0 CUDA codegen (external; audit only, no fork).
- Given a frozen dot/GEMM fixture When executed on host vs T4 Then the
  ulp table distinguishes FMA contraction from reduction geometry (S2).
- Invariants: no production code change (audit + tests only).
- Acceptance: each audited kernel has an FMA present/absent verdict with
  PTX evidence; micro-tests pin the per-op ulp contribution.
- Out of scope: disabling FMA in cubecl codegen (upstream change; file as
  follow-up if verdict is present).
- Traceability: `dot.rs:63` (`acc += x[i]*y[i]` contractable site).
- Open: whether cubecl exposes per-kernel FP-contract flags.

### S5 — Analytic-integral parity (libcint vs cintx) [draft]

- Rationale: with xcfun excluded, the comparison floor is the integral
  layer itself. cintx claims libcint byte-identity at source per shipped
  operator; S5 pins that claim on exactly the operators the KRHF band path
  consumes (overlap, kinetic, nuclear, GTH PP, ERI/JK lattice sums), so no
  downstream GPU gap can be misattributed to integrals.
- Preconditions: frozen YN cell geometry (+ digest); cintx-ops manifest
  listing shipped families.
- Input: frozen cell, operator list (`int1e_ovlp`, `int1e_kin`,
  `int1e_nuc`, GTH PP parts, `int2e`/JK route used by KRHF).
- Output: per-operator `(max_abs, max_ulp)` libcint-vs-cintx table on CPU.
- Dependencies: `pyscf-gto` intor dispatch, cintx path deps; oracle
  `tests/oracle/test_intor_oracle.py`.
- Given the frozen cell When each operator is evaluated via libcint
  (upstream) and cintx (native, CPU) Then the table shows the integral
  floor — expected 0 ulp where cintx pins byte-identity, else a measured
  floor that caps all downstream attribution.
- Invariants: no operator change; census only. CPU-only (no hardware).
- Acceptance: every KRHF-consumed operator has a table row; any nonzero
  row names the cintx workstream/operator as owner (not this plan).
- Out of scope: extending cintx operator coverage; any XC functional.
- Traceability: `crates/pyscf-gto/Cargo.toml:13-22`; `pyscf/gto/moleintor.py:20`.
- Open: GTH-PP lattice-sum families coverage for Y d-channel on the
  frozen cell (same gap class as the prior Y finding).

### S6 — eigensolver null test [draft]

- Rationale: prove the solver is an amplifier, not a root cause (it is
  host-side on all backends: `krhf.rs` `eig_channel` → `zeigh_gen`;
  DSYEVD-vs-faer agree only ~1e-9
  [VERIFIED: LOCAL crates/pyscf-algebra/tests/lapack_backend_oracle.rs:15]).
- Preconditions: bit-identical Fock+overlap fixtures.
- Input: frozen `(F, S)` pairs incl. YN shapes.
- Output: eigenvalue/eigenvector bit comparison across backends.
- Dependencies: `lapack_backend.rs`, `eigh_gen.rs`, `zeigh.rs`.
- Given identical Fock inputs When diagonalised under any
  `PYSCF_BACKEND` Then outputs are bit-identical (solver never sees the
  device).
- Invariants: no solver change.
- Acceptance: 0-ulp on all backends for frozen inputs (CPU-runnable
  today; CUDA column needs T4 but is expected trivially green).
- Out of scope: improving DSYEVD-vs-faer agreement.
- Open: none.

### S7 — per-backend regression gates / ORACLE-07 closure [draft]

- Rationale: convert findings into CI so the issue stays fixed.
- Preconditions: S1–S6 evidence complete.
- Input: measured floors per backend.
- Output: CI jobs asserting documented tolerances (CUDA 1e-10 energy /
  1e-8 gradient; ROCm 1e-10; WGPU 1e-6 or measured floor), reusing the
  S1 harness; ORACLE-07 checkbox ticked with evidence links.
- Dependencies: `.github/workflows/ci.yml` (`build-gpu` today is
  compile-only [VERIFIED: LOCAL ci.yml:75-88]), hardware runners.
- Given frozen fixtures + backend hardware When CI comparison runs Then
  pass/fail follows the documented per-backend tolerance, and any
  tightening is backed by a measurement entry.
- Invariants: CPU bit-exact gates untouched; no gate tighter than 1 ulp
  of its bounded quantity (measurements §3 rule).
- Acceptance: ORACLE-07 marked complete with gate names + floors;
  Phase 8 plan entry updated from `TBD`.
- Out of scope: procuring hardware runners (operational follow-up).
- Open: runner availability for CUDA-T4 / ROCm / shader-f64 WGPU.

### S8 — WGPU-f64 honesty guard [draft]

- Rationale: regression guard; the fix must never "close gaps" via silent
  precision loss.
- Input: shader-f64-less adapter (existing CI job).
- Output: fallback warning + CPU-f64 numerics, unchanged.
- Acceptance: existing `wgpu-no-f64-fallback` job still passes unmodified.
- Out of scope: everything else.

## 6. Acceptance scenarios

- A1 (methodology): S1 harness on CPU/CPU reproduces
  `e_tot`-≪-bands ordering and its `conv_tol` shrinkage before any GPU
  number is cited.
- A2 (attribution): each of geometry (S2), FMA (S4), integral floor
  (S5) has a measured ulp contribution table, with SCF fixed-point depth
  (S1) controlled; their sum accounts for the observed CUDA gap within
  1 order of magnitude.
- A3 (remediation): S3 implemented if S2 shows nonzero geometry ulps;
  integral-floor gaps (S5) routed to owning cintx workstream, never
  absorbed here.
- A4 (gates): S6 green on all backends; S7 CI jobs green at documented
  tolerances; ORACLE-07 closed.
- A5 (no regression): CPU oracle-profile bit-exact suites + `check-no-fma`
  + dependency-wall lints green throughout.

## 7. Impact scope

`pyscf-algebra` (reduce/dot/launch/select), `pyscf-kernels` (kernel audit
surface only), `pyscf-runtime` (resolver/probes — read-only expected),
`pyscf-pbc-scf`/`pyscf-pbc-dft` (harness consumers), `pyscf-py` (harness
plumbing if `dm0`/hex export gaps appear), `xtask` (possible
`check-no-fma` GPU-asm extension — optional), `.github/workflows/ci.yml`
(new hardware comparison jobs). Impact: local per spec except S7
(operational) and S3 (cross-module: algebra→kernels→method crates).

## 8. Compatibility and migration

No public API change required except possibly one documented
comparison-mode flag (S5 pin). `PAIRWISE_CHUNK`, oracle profile, and
`PYSCF_BACKEND` semantics frozen. Rollback per spec is revert-to-parent
(no migrations).

## 9. Risks and open questions

- R1: T4 hardware unavailable → S2(CUDA col), S4, S5 blocked; plan
  explicitly fronts all CPU-doable work (S1, S2-CPU/WGPU-if-present, S6,
  S7 scaffolding) in Wave 1.
- R2: S3 fixed-lane reduction may cost T4 throughput — measure (Wave 3)
  before merging; keep behind no flag (single code path) only if cost is
  negligible, else record decision.
- R3: cubecl may offer no per-kernel FP-contract control → S4 ends as
  "documented 0.5–1 ulp/term, attributed" (still closes its third of A2).
- R4: YN surrogate small-gap k-points (gaps ~6.5e-4 Ha observed) make SCF
  depth studies noisy — S1 frozen-dm design exists precisely for this.
- Q1–Q6: the T4 measurement questions from the research report
  (partial census, PTX audit, dm-anchored sweep, WGPU/ROCm pattern,
  solver null) — each is owned by its spec's acceptance task. (The
  former XC-trace question is void: xcfun excluded, S5 now integral
  parity, CPU-doable.)

## 10. Traceability and sources

Research report (subagent session `ses_f443a2a91ffepKjJYKvRYF9SyM`,
2026-09-19; full text in SOURCES.md). Directly re-verified this session:
`reduce.rs:69-114`, `krks.rs:311-380`, `REQUIREMENTS.md:169,360`,
`scc.rs kernel(dm0)` via codegraph. Prior-session YN measurements:
`/tmp/opencode/yn_*.json` (UNVERIFIED locally — not in repo).
