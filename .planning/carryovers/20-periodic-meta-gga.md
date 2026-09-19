# Carryover — periodic meta-GGA (τ) NumInt absent

**Source:** `.planning/phases/20-pbc-python-bindings/measurements/example-gate-gap-scope.md` §1,
recorded by the 20-18 rollup (`20-VERIFICATION.md`). Blocks `examples/pbc/20-k_points_scf.py:41`
and `10-gamma_point_scf.py` (`xc='m06,m06'`).

## Measured

- `KRKS(xc='m06,m06').kernel()` fails after 126 s at **parse** time:
  `unknown XC functional token 'M06' (XC string: 'M06,M06')`. The parser
  (`crates/pyscf-dft/src/parser/libxc.rs:389-437`) tries only the inline `XC_CODES` for the
  family-prefix search; libxc_rs does register `XC_HYB_MGGA_X_M06` (449) and `XC_MGGA_C_M06` (235)
  (`libxc_rs/crates/libxc-core/src/registry/by_name.rs:423,519`). Effort **S**.
- After parsing it would hit `crates/pyscf-pbc-dft/src/xc.rs:103-112` (`Family::Mgga` refused).
  `XcType` is `Lda`/`Gga` only (`xc.rs:60-90`); the molecular NumInt also refuses MGGA
  (`pyscf-dft/src/numint.rs:899`). Effort **L**: τ build, `vtau` Fock contraction on uniform and
  Becke grids, KRKS + hybrid-K path, parity gate.
- Upstream suite: 8 tests refused on `WB97`/`HSE06` tokens (`upstream-pbc-suite-after-20-19.md` §5′) —
  range-separated tokens, same parser area.

## Risk

libxc_rs vendors a pre-7.0.0 master; 43 functionals differ from PySCF's 7.0.0 at source level
(memory `libxc-rs-vendored-master-is-pre-7.0.0`). M06 parity against PySCF 7.0.0 is unmeasured.

## Unblock

1. S: prefix search through the libxc_rs registry (M06 → 449/235); gate the parse only.
2. L: periodic MGGA NumInt; gate KRKS M06 (or SCAN) He/Si vs vendored 2.12.1 at a floor measured first.
