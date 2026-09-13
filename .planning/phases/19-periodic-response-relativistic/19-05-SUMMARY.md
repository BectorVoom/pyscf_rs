# 19-05 SUMMARY — periodic SCF stability

**Shipped:** 2026-09-12. `stability` 7/7 green.

## Drivers (`crates/pyscf-pbc-scf/src/stability.rs`)

- `rhf_internal` (Hessian `2·Re(hop)`, `stable = not (e < -1e-5)`, direction
  for `_rotate_mo` when unstable) and `rhf_external` (triplet-response hop,
  NO ×2 — bra+ket already combined in the hop) over one shared dense
  lowest-eigenpair core. Internal and external are DISTINCT types
  (`InternalStability` / `ExternalStability`), never one flag.
- `rotate_mo_real` (anti-Hermitian `dr` + small-matrix scaling-squaring expm
  with a rotation sanity check); complex orbitals refused, not approximated.

## Gate: verdict first (`tests/stability.rs`)

- Synthetic spectra pin the rule both sides of the threshold (the exact
  boundary is fp-noise in every implementation — probed at ±10%, stated).
- Live-hop fixtures from upstream 2.12.1 KRHF/cc-pVDZ Γ (H2 R=1.4 stable
  both ways; R=5.0 externally unstable at e=−0.273, hop matrices
  materialized from upstream's own closures): this port's drivers reproduce
  upstream's dense lowest eigenvalues (< 1e-8) AND both verdicts.
- Two scale bugs caught by the fixtures (both the same shape — the internal
  ×2 applied twice: once in the test's threshold probe, once baked into the
  first fixture generation; regenerated with the RAW hop).
- Rotation preserves orthonormality; 1-vs-8 determinism.
