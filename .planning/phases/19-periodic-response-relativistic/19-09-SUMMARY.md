# 19-09 SUMMARY — KS subclasses (rks/uks/krks/kuks)

**Shipped:** 2026-09-12. `krks` 8/8 green; **Gate A passes** for RKS, UKS,
PBE0-hybrid (1e-8 Ha) and KRKS (upstream's own 5dp eV, per-method
inheritance).

## Subclasses (thin, over 19-06/07/08 + `hyb_fraction`)

- `rks`/`uks` (gamma real), `krks`/`kuks` (k-point complex-Hermitian):
  HF part at the functional's exchange fraction + full XC-kernel matrix
  (no placement logic that could drop k-couplings).
- `hyb_fraction` table (PBE0 0.25, B3LYP 0.2, HF 1.0, pure → 0) with RSH
  refusal FIRST (a range-separated name must never resolve through the
  global table — caught in-test); `check_hybrid_kernel` refuses a pure
  kernel under a hybrid name; `require_rks_response` compiles 19-03's
  `gen_response` seam into the crate.

## Gates (`tests/krks.rs`, H2/cc-pVDZ + diamond fixtures, live 2.12.1)

- Exact hyb-linearity of the HF part (non-vacuous switch, coefficient-precise).
- RKS/PBE, UKS/PBE-triplet, RKS/PBE0 at 1e-8 Ha (fxc = upstream a − HF part).
- KRKS/PBE diamond shift-0 at 5dp eV; KUKS single-k consistency vs UKS.
- Two fixture lessons recorded: nohup/background runs resolved a STALE
  2.14.0 install (all fixtures now assert `2.12.1` + repo path up front);
  triplet-H2 SCF solutions vary run to run — the UKS section is saved
  atomically (blocks + spectra + roots from ONE run), never mixed.

## Deferred (named, not dropped)

The fxc GRID contraction (Becke/numint, Phases 4/12) feeding the fxc MATRIX.
