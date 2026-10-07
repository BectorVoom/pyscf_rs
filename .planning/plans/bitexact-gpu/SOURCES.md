# SOURCES — bitexact-gpu planning evidence ledger

## Research report

Full Research-Agent report: subagent session `ses_f443a2a91ffepKjJYKvRYF9SyM`
(2026-09-19). Method: `codegraph_explore` + Grep/Read, no files modified.
Citations below marked [R] come from that report; [V] = directly re-verified
this session via codegraph/Read.

## Load-bearing sources ([V] = re-verified)

- [V] `crates/pyscf-algebra/src/reduce.rs:69-114` — `launch_reduce_sum`:
  partials buffer sized from launch geometry, host sequential tail fold.
- [V] `crates/pyscf-pbc-dft/src/krks.rs:311-380` — `fused_band_fock`
  (band-Fock one-shot structure).
- [V] `.planning/REQUIREMENTS.md:169` — ORACLE-07 exact wording; `:360` —
  status Pending.
- [V] `crates/pyscf-py/src/scf.rs:319-334` — `kernel(dm0)` →
  `InitGuessMode::UserDM` (S1 harness hook exists).
- [R] `.planning/ROADMAP.md:364` — per-backend tolerances (CUDA 1e-10 /
  1e-8 grad; WGPU 1e-6; ROCm 1e-10); `:369` Phase 8 `Plans: TBD`.
- [R] `measurements/README.md:80-103` — §3 bitwise rule (same-impl A/B
  only; 1-ulp floor; best 221 ulp).
- [R] `crates/pyscf-algebra/src/oracle.rs:1-94` — FOUND-06, PAIRWISE_CHUNK=128.
- [R] `.cargo/config.toml:14-24` — host FMA-freedom flags.
- [R] `xtask/src/bin/check_no_fma.rs:39-100` — host-only asm scan.
- [R] `crates/pyscf-runtime/src/backend.rs:9-110`,
  `crates/pyscf-algebra/src/select.rs:23-194` — resolver, auto priority,
  WGPU-f64 hard error.
- [R] `crates/pyscf-pbc-scf/src/kscf.rs:95-199` — `|dE|<conv_tol AND
  |g|<grad_tol` on unextrapolated Fock; `kdiis.rs:14-23` Re-only B-entry
  deviation (iteration count only).
- [R] `crates/pyscf-dft/src/xc_backend.rs` + `pyscf-dft/Cargo.toml:12-26` —
  libxc default (CPU-only), 4.7e-7 PBE/Si libxc-vs-xcfun gap.
  [SUPERSEDED by Correction below — retained as research provenance only.]
- [R] `.github/workflows/ci.yml:75-88,108-130,133+,474-489` — build-gpu
  compile-only; oracle-determinism matrix; check-no-fma; wgpu-no-f64-fallback.
- [R] `crates/pyscf-algebra/tests/lapack_backend_oracle.rs:15` — DSYEVD vs
  faer agree ~1e-9, not bitwise.

## Correction (2026-09-19, user directive — supersedes XC items)

- xcfun (`xcfun_rs`, `xcfun_gpu`) is **excluded** from this workstream.
  The research report's suspect #3 (XC substrate swap), research Q3
  (XC trace), and the 4.7e-7 libxc-vs-xcfun gap are **void** for planning
  purposes; SPEC S5 is now analytic-integral parity (libcint vs cintx).
- [VERIFIED: LOCAL `pyscf/gto/mole.py:21`] — "interface to the integral
  library libcint" (upstream side).
- [VERIFIED: LOCAL `pyscf/gto/moleintor.py:20`] — low-level libcint interface.
- [VERIFIED: LOCAL `crates/pyscf-gto/Cargo.toml:13-22`] — pyscf-gto integral
  seam via `cintx-{core,compat,rs,ops,runtime}` path deps.
- [VERIFIED: LOCAL `Cargo.toml:171`] — `[patch.crates-io] cintx = { path = "../cintx" }`.
- [VERIFIED: LOCAL `crates/pyscf-py/src/pbc/scf.rs:1761-1762`] — native PBC
  `get_bands` binding via `Krhf::get_bands:106` (KRHF vehicle).
- [VERIFIED: LOCAL `tests/oracle/test_intor_oracle.py`] — intor oracle
  suite (S5/T7 regression scope).

## Prior-session measurements ([UNVERIFIED] locally — files outside repo)

- `/tmp/opencode/yn_band_dump.py`, `yn_band_compare.py`,
  `yn_basis_blocks.json` — YN surrogate harness + GTH text-basis blocks.
- `/tmp/opencode/yn_{upstream,native}{,31,41,41t,41tt}.json` — mesh
  21³/31³/41³ × tol 1e-9/1e-11/1e-12 ladder (see SPEC §1 headline numbers).
