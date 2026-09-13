# 19-01 SUMMARY — gate floors A–E (no Rust)

**Measured:** 2026-09-12, vendored PySCF **2.12.1**, assertion census over
`pyscf/pbc/{tdscf,gw,adc,x2c}/test/` + the Phase-16 solver-spread result.
`git diff --stat -- crates/` empty (no Rust written or edited).

- Census: tdscf 42 assertions (modal 5dp, tightest-energy 8dp — the 10dp six
  are `vind−A·z` operator identities, not energies); gw 40 (modal 4–5dp);
  adc 83 (58+ at 4dp, all IP/EA roots); x2c 34 (29 at 8dp).
- Solver spread: Phase-16 Davidson `nroots` spread 5.11e-7 bounds every
  Davidson-family gate from below; GW grid/contour spreads unreached ⇒ Gate C
  sits at the loosest modal assertion (4dp).
- AC-vs-CD split: same-cell run did not fit the budget ⇒ Gate C stated per
  route regardless (Phase-16 FFTDF/GDF 9.22e-4 Ha precedent).
- Gates written to `measurements/README.md`; `ROADMAP.md:465`,
  `PBC-MASTER-PLAN §7` Phase-19 row and `19-CONTEXT §2` restated identically:
  A1 1e-5 (HF roots, sorted, explicit nroots) · A2 1e-8 Ha (KS roots) ·
  B structural (one CPHF solver) · C 1e-4 per route · D 1e-4 · E 1e-8 Ha.
  `1e-15 eV` struck (∼1 ulp at 5 eV; best port result anywhere 221 ulp).
