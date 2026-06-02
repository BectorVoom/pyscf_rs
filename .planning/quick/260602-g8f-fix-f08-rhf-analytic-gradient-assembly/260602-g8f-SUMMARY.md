---
quick_id: 260602-g8f
slug: fix-f08-rhf-analytic-gradient-assembly
date: 2026-06-02
status: complete
commit: 47298a0
---

# Quick Task 260602-g8f — F-08 RHF analytic gradient: SUMMARY

## Outcome

**RHF analytic nuclear-gradient assembly FIXED and finite-difference-certified
for s-shells.** Three real bugs in `pyscf-grad::rhf::grad_elec` were found and
fixed; the H2/STO-3G FD gate went from **0.695 → 2.6e-9 Ha/Bohr**. Verified
term-by-term against live PySCF 2.12.1 (`.upstream-venv`).

The remaining work is **NOT a pyscf_rs bug** — it is an external cintx p-shell
gradient-kernel bug (see below).

## The audit was doubly wrong about F-08

`AUDIT-FIX-2026-06-01.md` said the grad integrals were "MISSING from cintx" /
later "evaluate today, only multi-wave assembly remains." Ground-truthing:
1. The integrals evaluate, so the assembly runs — but produced **garbage**
   (broken molecular symmetry), because it was never FD-validated.
2. The remaining blocker is cintx returning **numerically wrong** grad integrals
   for p-shells — "evaluate ≠ correct" (cf. the DF-01 Rys-recurrence class).

## Three bugs fixed (`crates/pyscf-grad/src/rhf.rs`)

1. **Component layout.** cintx/pyscf-gto component integrals are F-order
   `(3,nao,nao)` with the **component axis FASTEST** —
   `out[comp + ncomp*(i + j*nao + …)]` (arity-2 and arity-4 alike; confirmed in
   `pyscf-gto/src/intor.rs` stitch). pyscf-grad read them component-*slowest*
   (`comp*nao*nao + …`) in `get_veff`, `hcore_deriv`, and `grad_elec`, scrambling
   x/y/z and breaking symmetry. (The `shape` metadata `[3,nao,nao]` is
   column-major, so "component-leading" by shape ≠ component-slowest in memory.)
2. **`hcore_deriv` transposed slice.** numpy `vrinv[:, p0:p1] += h1[:, p0:p1]`
   slices AXIS 1 (the bra/row index `i`). The loop restricted the ket/col `j`
   instead → under-counted the Hellmann-Feynman term (term1).
3. **`get_veff` K-contraction.** PySCF `('lk->s1ij','jk->s1il')`: exchange output
   is the integral's FOURTH axis `l` — `vk[x,i,l]=Σ_jk g(x,i,j,k,l)·D[j,k]`. The
   code output to `(i,j)` summing `(k,l)`; coincided only for 1-function-per-shell
   systems (why H2 masked it).

## Verification (in-sandbox, no libxc; pyscf-grad/scf dep graph = 0 libxc rows)

- **H2/STO-3G FD gate** (`rhf_verify_fd_numeric`, **un-gated**): max|fd−analytic|
  = 2.6e-9 ≤ 1e-6.
- **Term-by-term vs PySCF 2.12.1** on H2 and bent H2O: term1/term2/term3 match
  exactly wherever cintx integrals are correct.
- Full `pyscf-grad` suite green; `rustfmt --edition 2024 --check` clean; clippy
  no findings on changed code.

## Remaining blocker — EXTERNAL cintx p-shell gradient-kernel bug

Element-wise vs PySCF 2.12.1 on bent H2O/STO-3G (O carries a 2p shell):

| cintx integral            | max|Δ| vs upstream | verdict |
|---------------------------|--------------------|---------|
| `int1e_ovlp`, `int2e`     | ~7e-9              | correct |
| `int1e_ipovlp`, `int1e_ipkin` | ~7e-9         | correct |
| `int1e_iprinv` @ first atom | ~7e-9            | correct |
| `int1e_ipnuc`             | 3.7e-1            | **WRONG** |
| `int1e_iprinv` @ off-origin nucleus | 2.3e-1  | **WRONG** |
| `int2e_ip1`               | 1.1               | **WRONG** |

So `rhf_verify_fd_numeric_pshell` (H2O) stays `#[ignore]`'d with this reason.
Un-gate when cintx fixes its p-shell gradient kernels. (Recommend filing a cintx
bug: `int1e_ipnuc` / `int1e_iprinv` at off-origin centers + `int2e_ip1` are
wrong for l≥1.)

## Out of scope (follow-ups)

- MP2/CCSD `grad_elec` WR-01 (return bare RHF `de`) + their p-shell FD gates —
  also p-shell-blocked by the same cintx bug.
- UHF/RKS/UKS gradient variants.

## Notes

- Done inline (not via spawned executor) — a numerically-delicate port where the
  FD/oracle correctness matters, matching the AUDIT-FIX F-13/F-01 precedent.
- A mid-task ENOSPC (disk full from repeated builds) truncated `rhf.rs` to 0
  bytes; restored from git HEAD and re-applied all fixes cleanly (verified).
- Committed scoped by pathspec (`git commit -- <files>`) — a concurrent agent
  (`260602-b62`) had in-flight uncommitted changes in pyscf-kernels/scf/gto.
