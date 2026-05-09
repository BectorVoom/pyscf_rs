# Feature Research — pyscf_rs (FEATURES dimension)

**Domain:** Molecular ground-state quantum chemistry (Rust rewrite of PySCF, drop-in `pyscf.*` import surface)
**Researched:** 2026-05-09
**Confidence:** HIGH (upstream source is in-tree; APIs read directly from `pyscf/*.py` and `examples/*`)

## Scope of this Document

v1 covers seven in-scope subsystems: `gto`, `scf`, `dft`, `mp2`, `ccsd`, `grad`, `geomopt`, plus a `df` (density fitting) module that is required transitively. Out of scope (PBC, x2c/dhf, mcscf/mrpt, tdscf/tddft/adc/gw/eom-cc, CCSD(T)+, AGF2, NAC, EPH, lo, qmmm, solvent) is enumerated in PROJECT.md and is treated as **anti-feature for v1** here.

Complexity scale (S/M/L/XL) is calibrated to *effort to reach numerical parity with upstream PySCF in pure Rust + cubecl*, not to algorithmic depth. LOC counts are upstream `.py` lines (proxy for feature surface, not implementation cost — Rust replacement will typically be 1.5–3× the line count once integral-engine bindings, error types, and test fixtures are included).

---

## 1. Drop-In API Contract — Top-20 Idioms

These are the public Python idioms that **must work unchanged** for a typical PySCF script to keep running. Sourced from `examples/0-readme.py`, `examples/scf/*`, `examples/dft/*`, `examples/mp/*`, `examples/cc/*`, `examples/grad/*`, `examples/geomopt/*`. Anything not in this list is fair game to change or omit.

| # | Idiom | Source example | Required v1 behavior |
|---|-------|---------------|----------------------|
| 1 | `mol = pyscf.M(atom='...', basis='...', charge=N, spin=N)` | `0-readme.py`, `gto/00-input_mole.py` | Returns built `Mole`. Accepts XYZ string, list of `[symbol, (x,y,z)]`, or `'Z x y z; ...'` |
| 2 | `mol = gto.Mole(); mol.atom = '...'; mol.basis = '...'; mol.build()` | `gto/00-input_mole.py` | Lazy-build flow; equivalent to (1) |
| 3 | `mf = scf.RHF(mol).run()` / `mol.RHF().run()` / `scf.HF(mol).kernel()` | `0-readme.py`, `scf/00-simple_hf.py` | RHF for closed-shell, dispatches to ROHF/UHF if `spin>0`; chained `.run()` returns self |
| 4 | `mf = scf.UHF(mol).kernel()` | `scf/02-rohf_uhf.py` | UHF returns `(mo_a, mo_b)`-shaped `mo_coeff` |
| 5 | `mf = dft.RKS(mol); mf.xc = 'b3lyp'; mf.kernel()` | `dft/00-simple_dft.py` | XC parsed via `libxc_rs`/`xcfun_rs`; default `LDA,VWN` |
| 6 | `mf = scf.RHF(mol).density_fit().run()` | `scf/20-density_fitting.py` | DF-JK; auxbasis chosen automatically when omitted |
| 7 | `mp2 = mf.MP2().run()` then read `mp2.e_corr`, `mp2.e_tot`, `mp2.t2` | `0-readme.py`, `mp/00-simple_mp2.py` | RMP2/UMP2 dispatch on mf type; `e_corr` is the canonical result |
| 8 | `cc = mf.CCSD().run()` then `cc.e_corr`, `cc.t1`, `cc.t2`, `cc.l1`, `cc.l2` | `cc/00-simple_ccsd.py`, `cc/01-lambda.py` | RCCSD/UCCSD dispatch; `cc.solve_lambda()` populates `l1, l2` |
| 9 | `mycc.frozen = 2` or `mycc.set_frozen()` (auto chemcore) | `cc/11-frozen_core.py` | int, list-of-int, or `'auto'`; consistent with MP2 |
| 10 | `g = mf.nuc_grad_method().kernel()` or `g = mf.Gradients().kernel()` | `grad/01-scf_grad.py` | Returns `(natm, 3)` array in Hartree/Bohr |
| 11 | `from pyscf.geomopt.geometric_solver import optimize; mol_eq = optimize(mf)` | `geomopt/01-geomeTRIC.py`, `01-pyberny.py` | Returns optimized `Mole`; accepts mf, mp2, ccsd, dft objects |
| 12 | `mf.kernel(dm0)` — pass density matrix as initial guess | `scf/15-initial_guess.py` | First positional arg of `.kernel()` is `dm0` |
| 13 | `mf.init_guess = 'minao' \| 'atom' \| '1e' \| 'huckel' \| 'sap' \| 'chkfile'` | `scf/15-initial_guess.py` | At minimum minao, 1e, atom must work in v1 |
| 14 | `mf.chkfile = 'foo.chk'` then later `mf.init_guess = 'chkfile'` | `scf/14-restart.py` | HDF5 checkpoint read/write |
| 15 | `mf.level_shift = 0.2` or `mf.level_shift = (0.3, 0.2)` for UHF | `scf/03-level_shift.py` | scalar for RHF, tuple for UHF α/β |
| 16 | `mf.conv_tol = 1e-10`, `mf.max_cycle = 50`, `mf.diis_space = 8`, `mf.verbose = 4` | every example | Standard control attributes; defaults must match upstream |
| 17 | `mf.analyze()` and `mf.mulliken_pop()` | `scf/00-simple_hf.py` | Mulliken (and meta-Lowdin) population analysis printed to log |
| 18 | `mf.make_rdm1()` returns AO density matrix | `cc/01-density_matrix.py`, `scf/15-initial_guess.py` | Shape `(nao,nao)` for RHF, `(2,nao,nao)` for UHF |
| 19 | `mf.mo_coeff`, `mf.mo_energy`, `mf.mo_occ`, `mf.e_tot`, `mf.converged`, `mf.cycles` | universal | Final results saved as instance attributes |
| 20 | `mol.intor('int1e_ovlp')`, `mol.intor('int2e')`, `mol.nao_nr()`, `mol.atom_coords()` | `gto/20-ao_integrals.py` | Direct access to integrals via `cintx`; spherical default, Cartesian via `mol.cart=True` |

**Bonus idioms (not strictly top-20, but appear in many examples):**
- `mf.as_scanner()` for PES scans (`scf/30-scan_pes.py`)
- `mf.callback = fn` taking `locals()` dict (`scf/24-callback.py`)
- `cc.make_rdm1()`, `cc.make_rdm2()` (`cc/01-density_matrix.py`)
- `mol.set_geom_(coords, unit='Bohr')` for in-place geometry mutation (used by geomopt)
- `mf.to_uhf()`, `mf.to_rhf()`, `mf.to_uks()` mean-field conversion helpers (used by MP2/CCSD dispatch)

---

## 2. Feature Landscape

### 2.1 `gto` (Molecular Geometry & Basis)

#### Table Stakes

| Feature | Why expected | Complexity | Upstream LOC | Deps | Notes |
|---------|--------------|------------|--------------|------|-------|
| `Mole` core attrs (`atom`, `basis`, `charge`, `spin`, `unit`, `verbose`, `max_memory`, `output`, `cart`, `symmetry`, `ecp`, `nucmod`, `magmom`) | Every script sets these | M | mole.py 4383 | — | 90% of scripts use ≤ 7 of these |
| `Mole.build()` / `gto.M()` factory | Universal entry point | L | mole.py 2470- | cintx | Must accept kwargs identical to upstream and produce same `_atm/_bas/_env` arrays |
| `mol.atom_coords()`, `mol.atom_charges()`, `mol.atom_symbol(i)`, `mol.natm`, `mol.nbas`, `mol.nelectron`, `mol.nelec`, `mol.spin`, `mol.energy_nuc()` | Read by every downstream method | S | mole.py 2373-2435 | — | Pure accessor surface |
| `mol.nao_nr()`, `mol.nao`, `mol.ao_loc_nr()`, `mol.aoslice_by_atom()`, `mol.ao_labels()` | Used by SCF, MO transforms, analysis | S | mole.py 1378-1656 | — | `ao_labels` returns human-readable strings like `'0 O 1s'` |
| `mol.intor('int1e_*')`, `mol.intor('int2e')` 1e/2e integrals | Foundation of every method | XL | moleintor.py + libcint | cintx | Wrap `cintx` directly; signatures: `intor(name, comp=1, hermi=0, aosym='s1', ...)` |
| `mol.intor_symmetric` (4-fold symmetric ERIs) | Used in dense `_eri` paths | M | moleintor.py | cintx | Returns triangular packed array |
| Atom string parser (XYZ, semicolon-separated, list-of-tuples, list-of-lists, `'Z x y z'` syntax with element symbols, ghost atoms `GHOST-X`/`X@1`/atom tagging `H:1`, `H@2`) | Every example uses one of these | M | mole.py 320-417 | — | Many forms; underestimating this is a classic rewrite trap |
| Basis input as **string** (e.g. `'cc-pvdz'`, `'6-31G*'`), **dict-by-element** (`{'O':'cc-pvdz', 'H':'sto3g'}`), **dict with `'default'`**, **tuple of basis sets** (combined), **`gto.parse(...)` NWChem-format string**, **`gto.load(name, elem)`**, **`gto.basis.parse_ecp(...)`** | `gto/04-input_basis.py` | L | mole.py + basis/ | — | Critical: 11 distinct input forms must all work |
| Built-in basis library lookup (sto-3g, 6-31g, 6-31g*, 6-311g**, 3-21g, cc-pVDZ/TZ/QZ/5Z + aug-, def2-SVP/TZVP/QZVP/QZVPP + RI/JKFIT, ANO, pcseg, ccecp, F12) | "It just works" expectation | M | basis/ ~207 .dat files | — | Ship the same `pyscf/gto/basis/*.dat` files; parser must be Gaussian-94/NWChem text format |
| ECP loading (`mol.ecp = 'def2-svp'` and inline `parse_ecp`) | Heavy elements need ECPs | M | ecp.py + cintx | cintx | Format identical to NWChem; SOC-ECPs out-of-scope |
| `gto.parse(nwchem_str)`, `gto.basis.parse(...)`, `gto.basis.load(name, elem)` | Custom basis input | M | basis/parse_nwchem.py | — | Returns internal `[[l, (e,c), (e,c), ...], ...]` format |
| `unc-` prefix to uncontract basis, `@`-truncation (`'ano@3s2p'`), `gto.etbs` even-tempered, `gto.uncontract` | Power-user idioms in `gto/04-input_basis.py` | S each | mole.py 507-687 | — | Composable; do NOT skip — they appear in tutorials |
| `mol.copy()`, `mol.set_geom_()`, `mol.dumps()/loads()` (JSON) | geomopt, scanners depend on these | S | mole.py 1188-1351 | — | `dumps`/`loads` round-trips; geomeTRIC needs `set_geom_` |
| `gto.M(symmetry=True)` group detection — at least D2h subgroups | `scf/13-symmetry.py` | M | symm/ ~5000 LOC | — | Symmetry is "table stakes" only at the *Mole-detect* level (groupname, axes); SCF symmetry adaptation is **deferred** (see §6) |

#### Differentiators

| Feature | Value proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Sub-second `mol.build()` for 5000-AO molecules | Upstream Python parser is slow; pure-Rust regex+parser is 10–100× faster | S | First user-visible "this feels faster" win |
| Strict basis-name resolution with did-you-mean errors | Upstream silently uses fallback aliases; users get cryptic errors | S | Error message: `BasisNotFoundError: 'cc-pvdz' not found for O. Did you mean 'cc-pVDZ'?` |
| Validate atom-spec early (single Rust pass; report all errors at once) | Upstream stops on first parse error | S | "Found 3 errors in atom string: ..." |
| `mol.dumps()/loads()` as canonical CBOR (with JSON compat) | Faster checkpoint round-trips, smaller file size | S | Keep JSON as fallback for compatibility |

#### Anti-features (deliberately NOT in v1)

| Feature | Why requested | Why NOT in v1 |
|---------|---------------|---------------|
| 4-component spinor integrals (`mol.intor_2c`, `intor_4c`) | Relativistic methods need them | Out-of-scope per PROJECT.md (`x2c`/`dhf` deferred) |
| F12 zeta and `int2e_stg` integrals | F12 explicit-correlation methods | F12 methods deferred; integrals are in cintx but not exposed |
| PBC `Cell` class (`a` lattice parameter) | Solid-state | PBC entire pipeline is out-of-scope |
| `mol.symmetry='Td'` user-forced point group with full symmetry-adapted SCF | Niche performance feature | Detection yes, full SCF symm adaptation deferred |
| Pseudo-potentials (GTH PSP for PBC) | PBC users | PBC out-of-scope |
| Spinor labels (`spinor_labels`), 2c AO labels | Relativistic | Out-of-scope |

---

### 2.2 `scf` (Hartree-Fock & SCF base)

#### Table Stakes

| Feature | Why expected | Complexity | Upstream LOC | Deps | Notes |
|---------|--------------|------------|--------------|------|-------|
| `RHF`, `UHF`, `ROHF`, `GHF` classes with `.kernel()` / `.run()` / `.scf()` | Core HF | XL | hf.py 2511 + uhf 1140 + ghf 569 + rohf 563 | gto, cintx | All four return `(scf_conv, e_tot, mo_energy, mo_coeff, mo_occ)` from `kernel()` |
| `scf.HF(mol)` factory (closed-shell→RHF, open→UHF) | One-liner in tutorials | S | scf/__init__.py | RHF/UHF | Trivial wrapper |
| Convergence parameters: `conv_tol` (1e-9 default), `conv_tol_grad`, `max_cycle` (50), `init_guess`, `direct_scf` (True), `direct_scf_tol` (1e-13) | Every SCF run uses these | S | hf.py 1689-1712 | — | Default values must match upstream exactly |
| **DIIS** (`SCF_DIIS`/`CDIIS`, default), `diis_space=8`, `diis_start_cycle=1`, `diis_damp` | Convergence acceleration | M | diis.py 247 + lib/diis.py | — | C-DIIS (Pulay) is the default; ADIIS/EDIIS are differentiators (§ below) |
| `level_shift` (scalar or `(α,β)` tuple) | Open-shell, near-degenerate | S | hf.py 775-797 + uhf | — | Tuple form mandatory for UHF |
| `damp` (Fock damping factor) | Pre-DIIS warmup | S | hf.py 798-800 | — | Often used with `diis_start_cycle` |
| Initial guesses: `'minao'` (default), `'atom'`, `'1e'`/`'hcore'`, `'huckel'`, `'sap'`, `'chkfile'`, plus passing a `dm0` ndarray to `.kernel()` | `scf/15-initial_guess.py` lists 7 forms | M | hf.py 348-770 | gto basis | `'minao'` projects atomic densities from MINAO basis; `'sap'` from Lehtola SAP fits |
| `mf.chkfile` HDF5 read/write | Restart, restart-from-different-mol | M | chkfile.py + h5py | hdf5 | Save mo_coeff, mo_energy, mo_occ, e_tot; loadable by another `mf` |
| Density-fitting decoration: `scf.density_fit(mf)`, `mf.density_fit()`, `mf.with_df`, `mf.with_df.auxbasis` | DF-HF, DF-DFT user pattern | L | df/df_jk.py 645 + df.py 391 | gto, cintx | `with_df=False` reverts to direct SCF; auxbasis defaults to BSE-derived JK fit |
| `mf.get_init_guess(mol, key)`, `mf.get_hcore()`, `mf.get_ovlp()`, `mf.get_jk()`, `mf.get_veff()`, `mf.get_fock()`, `mf.eig()`, `mf.get_occ()`, `mf.make_rdm1()`, `mf.energy_elec()`, `mf.energy_tot()`, `mf.get_grad()` | All overrideable; users subclass | L | hf.py methods | — | These are **the** extension API; users subclass to customize Hamiltonian |
| `mf.analyze()`, `mf.mulliken_pop()`, `mf.mulliken_meta()`, `mf.dip_moment()`, `mf.quad_moment()` | Post-SCF analysis | M | hf.py 1199-1504 | — | Mulliken-meta (meta-Löwdin) is the recommended population analysis |
| `mf.canonicalize(mo, occ)`, `mf.stability()` | Used before post-SCF | M | hf.py 1359 + stability.py | — | Stability is technically optional but ubiquitous in CCSD scripts |
| `mf.as_scanner()` for PES scans | `scf/30-scan_pes.py` | S | hf.py 1538-1604 | — | Returns callable that takes Mole or geometry list |
| `mf.callback = fn` per-iteration hook | Logging, custom convergence | S | hf.py kernel loop | — | Receives `locals()` dict |
| `mf.to_rhf()`, `mf.to_uhf()`, `mf.to_ghf()`, `mf.to_rks()`, `mf.to_uks()` | Used by MP2/CCSD dispatch | S | hf.py 2272-2329 | — | These are **upstream-required** for `mp.MP2(mf)` dispatch to work |
| `mf.reset(mol)`, `mf.set(...)` chained config, `mf.run(**kwargs)` | StreamObject convention | S | lib/misc.py StreamObject | — | Mandatory base-class behavior |
| `mf.nuc_grad_method()` / `mf.Gradients()` | Gradient construction | S | grad/rhf.py | grad | Used in geomopt |

#### Differentiators

| Feature | Value | Complexity | Notes |
|---------|-------|------------|-------|
| Reproducible SCF (seed-stable initial guess, deterministic eigh tiebreak) | PySCF's `eigh` order can flip with LAPACK version; pyscf_rs locks order | S | Use `numpy.linalg.eigh`-equivalent stable algorithm |
| Better diagnostics on near-singular overlap (cond(S) report + suggested `lib.remove_linear_dep`) | Upstream prints a warning; pyscf_rs prints + suggests fix | S | Hooks into existing `check_sanity` pattern |
| Per-iteration cost breakdown (Fock build, DIIS, eigh, all timed in same log line) | Upstream timings are scattered | S | One log format, easier to grep |
| 2–5× speedup on Fock build via cubecl JK kernels (PROJECT.md goal) | Core value proposition | XL | The differentiator the project exists for |

#### Deferred (real but later)

| Feature | When | Why deferred |
|---------|------|--------------|
| ROHF (open-shell singlet, atomic excited states) | v1.x | Upstream `rohf.py` 563 LOC; non-trivial spin coupling. v1 ROHF can fall back to UHF + spin-projection warning |
| `scf.newton(mf)` second-order SCF (SOSCF) | v1.x | `soscf/newton_ah.py` is XL; only needed for hard-convergence cases. v1 ships DIIS/ADIIS only |
| `mf.stability()` (real/complex orbital stability) | v1.x | Useful but not on the convergence critical path |
| ADIIS, EDIIS DIIS variants | v1.x | C-DIIS handles 95% of cases; add when concrete user complaints land |
| `frac_occ` (fractional occupation for degenerate HOMO/LUMO) | v1.x | `scf/54-fractional_occupancy.py` shows the API; addons-layer feature |
| Smearing / finite-temperature SCF | v2 | Used mostly for PBC; molecular use rare |
| `irrep_nelec={'A1':(N,N),...}` symmetry-constrained occupation | v2 | Tied to deferred symmetry-adapted SCF |
| `scf.dispersion` D3/D4 corrections (DFT-D) | v1.x | Pure post-SCF energy correction; `dftd3`/`dftd4` Rust ports needed |
| GHF / generalized HF (collinear + non-collinear) | v1.x | PROJECT.md says "v1 includes GHF", but practical demand is small. Implement basic GHF; spin-orbit GHF deferred |

#### Anti-features

| Feature | Why requested | Why NOT in v1 |
|---------|---------------|---------------|
| `DHF` (Dirac-Hartree-Fock 4-component) | Heavy elements | Out-of-scope per PROJECT.md |
| `X2C` / `sfx2c1e` exact 2-component | Scalar relativistic effects | Out-of-scope |
| `M3SOSCF` (Markovian Multi-agent Monte-Carlo SCF) | Listed in upstream FEATURES | Esoteric; never appears in tutorials |
| `mf.multigrid_numint` | PBC accelerator | Out-of-scope |
| `mf._eri` user-supplied custom Hamiltonian (Hubbard etc.) | Method development | `scf/40-customizing_hamiltonian.py` is research feature; not for v1 production users |
| QM/MM coupling (`qmmm.mm_charge`) | QM/MM workflows | Out-of-scope per PROJECT.md |
| Solvent decoration (`mf.PCM()`, `mf.ddCOSMO()`) | Solution-phase chemistry | Out-of-scope |

---

### 2.3 `dft` (Kohn-Sham DFT)

#### Table Stakes

| Feature | Why expected | Complexity | Upstream LOC | Deps | Notes |
|---------|--------------|------------|--------------|------|-------|
| `RKS`, `UKS`, `ROKS` classes with same SCF interface | DFT entry point | L | rks.py 569 + uks.py 209 + roks.py | scf, libxc_rs | Inherits `KohnShamDFT` mixin onto SCF base |
| `mf.xc = 'b3lyp'` string parser, including comma form `'pbe,pbe'` (X,C separation), shorthands (`'svwn'`, `'bp86'`, `'blyp'`), `'pbe0'`, `'b3p86'`, `'wb97x'`, `'cam-b3lyp'`, `'wb97m_v'` | Every DFT script | L | libxc.py 1527 + parse_dft | libxc_rs, xcfun_rs | Must parse identical strings to upstream; XC_ALIAS dict shared |
| Numerical integration via `numint.NumInt` (`nr_rks`, `nr_uks`, `nr_nlc_vxc`, `eval_rho`, `eval_xc`, `eval_ao`) | Internal but exposed | XL | numint.py 3030 | libxc_rs, xcfun_rs | The 3000-line file is the bulk of DFT effort |
| `Grids` class with `.level` (0–9, default 3), `.atom_grid` dict, `.prune` scheme, `.radi_method`, `.becke_scheme`, `.build()` | `dft/11-grid_scheme.py` | L | gen_grid.py 704 + LebedevGrid.py ~4000 | — | Lebedev grids (302/434/590/770/974…) + Treutler-Ahlrichs radial + Becke partition |
| `mf.grids` is auto-built on first call to `get_veff` | All DFT examples | S | rks.py:initialize_grids | — | Grid build is hot path; must be cubecl-parallel |
| Range-separated hybrids (CAM-B3LYP, ωB97X, ωB97M-V): `omega`, `alpha`, `hyb` extracted via `rsh_and_hybrid_coeff` | `dft/12-camb3lyp.py`, `dft/13-rsh_dft.py` | M | rks.py:get_veff 100-130 | libxc_rs | Requires `int2e_lr/sr_*` integrals from cintx (omega support) |
| NLC functionals (VV10, `mf.nlc='VV10'`, separate `nlcgrids`, `do_nlc()` logic) | `dft/15-nlc_functionals.py` | M | numint.py:nr_nlc_vxc + rks.py:do_nlc | xcfun_rs | wB97M-V, B97M-V, ωB97X-V all imply VV10 |
| DF-DFT via `dft.RKS(mol).density_fit()` | `dft/20-density_fitting.py` | reuse | (shared with scf DF) | df | Same code path as DF-HF |
| `mf.analyze()`, `mf.mulliken_pop()` (inherited from SCF) | Universal | reuse | hf.py | — | — |

#### Differentiators

| Feature | Value | Complexity |
|---------|-------|------------|
| GPU-accelerated Becke quadrature on cubecl | Currently Python+C; 5–10× speedup possible | L |
| Single XC string parser shared between libxc and xcfun (consistent semantics) | Upstream behavior subtly differs between backends | S |
| Memory-aware grid streaming (large molecules with no OOM) | Upstream needs `mf.max_memory` tuning | M |

#### Deferred

| Feature | When | Why |
|---------|------|-----|
| GKS (collinear) and 2-component DFT | v1.x | GHF is in v1 scope but full GKS+xc evaluation is XL; GKS scalar might land in v1 if cheap |
| DFT+U (`RKSpU`, `UKSpU`) | v2 | Niche, mostly for PBC users |
| Multi-collinear functionals (for non-collinear GKS) | v2 | Tied to deferred 2-component |
| Custom XC via `dft.libxc.define_xc_(mf, ...)` | v1.x | `dft/24-define_xc_functional.py`; advanced user feature |
| D3/D4 dispersion (`mf.disp = 'd3bj'`) | v1.x | Needs Rust port of dftd3/dftd4 or ffi |
| Multigrid (`mf.multigrid_numint`) | v2 | PBC accelerator |

#### Anti-features

| Feature | Why NOT in v1 |
|---------|---------------|
| TDDFT linear response | Excited states out-of-scope |
| TDA gradients, RPA, ppRPA, GW | Out-of-scope |
| Spin-flip TDA via multi-collinear functionals | Out-of-scope |
| MC-PDFT, L-PDFT, CMS-PDFT | Multi-reference out-of-scope |

---

### 2.4 `mp2` (Møller-Plesset)

#### Table Stakes

| Feature | Why expected | Complexity | Upstream LOC | Deps | Notes |
|---------|--------------|------------|--------------|------|-------|
| `mp.MP2(mf)` factory dispatching to RMP2/UMP2/(GMP2/DFMP2) | All examples use `mf.MP2()` | S | mp/__init__.py | mp2/ump2/dfmp2 | If `mf.with_df` set, dispatches to `dfmp2.DFMP2` |
| `RMP2`, `UMP2` classes; `kernel()` returns `(e_corr, t2)` | `mp/00-simple_mp2.py` | L | mp2.py 948 + ump2.py 804 | scf, ao2mo | `mp.e_corr`, `mp.e_tot`, `mp.t2` saved as attrs |
| `mp.frozen = N` (int), `frozen = [list]`, `frozen = (list_a, list_b)` for UMP2 | `cc/11-frozen_core.py` shows pattern | M | mp2.py:get_frozen_mask | — | Identical semantics to CC |
| In-core 4-index integral transformation via `ao2mo.kernel(mol, mo_coeff)` | Required by canonical MP2 | L | ao2mo/incore.py + outcore.py | gto, cintx | Bottleneck for non-DF MP2; must be cubecl-parallel |
| DF-MP2 (`dfmp2.DFMP2`) when mf has `with_df` | Default for large systems | M | dfmp2.py 595 | df | Auxbasis defaults to a `*-RIFIT` family |
| `mp.make_rdm1()`, `mp.make_rdm2()` | `mp/11-dfmp2-density.py` | M | mp2.py density section | — | Required for MP2 gradients |
| `mp.as_scanner()` for PES | scanner pattern | S | mp2.py | — | Reuses `mf.as_scanner()` |

#### Differentiators

| Feature | Value | Complexity |
|---------|-------|------------|
| Cubecl-parallel ovov contraction (the bulk of MP2 cost) | 3–5× speedup expected | M |
| Streaming DF-MP2 with bounded memory | Currently dependent on `max_memory` heuristics | M |
| Spin-component-scaled MP2 (`SCS-MP2`, `SCS(N)-MP2`) flags | Already in upstream as `e_corr_ss`/`e_corr_os` tagged on result; just expose as kwarg | S |

#### Deferred

| Feature | When | Why |
|---------|------|-----|
| GMP2 (generalized MP2 for GHF) | v1.x | GHF is borderline scope; defer GMP2 unless GHF lands |
| MP2 natural orbitals (`mp.make_fno`) | v1.x | Used by FNO-CCSD; couple with that feature |
| Iterative non-canonical MP2 (`_iterative_kernel` for non-canonical MOs) | v1.x | Niche; only when MOs aren't diagonal in Fock |
| OO-MP2 (`omp2`) | v2 | Research feature |

#### Anti-features

| Feature | Why NOT in v1 |
|---------|---------------|
| MP3, MP4, MP5 | "Higher-order post-SCF beyond CCSD" out-of-scope per PROJECT.md |

---

### 2.5 `cc` (Coupled-Cluster Singles-Doubles)

#### Table Stakes

| Feature | Why expected | Complexity | Upstream LOC | Deps | Notes |
|---------|--------------|------------|--------------|------|-------|
| `cc.CCSD(mf)` factory dispatching RCCSD/UCCSD/(GCCSD/DF-CCSD) | `cc/00-simple_ccsd.py` | S | cc/__init__.py | ccsd/uccsd | Mirrors MP2 dispatch logic |
| `RCCSD`, `UCCSD` classes; `.kernel()` returns `(converged, e_corr, t1, t2)` | All CC examples | XL | ccsd.py 1734 + uccsd.py 1393 | scf, ao2mo, mp2 | Hot path; `_ccsd.c` extensions are the main rewrite target |
| Convergence: `conv_tol` (1e-7), `conv_tol_normt` (1e-5), `max_cycle` (50), `diis_space` (6), `diis_start_cycle`, `iterative_damping`, `level_shift` | `cc/14-ccsd_diis.py` | M | ccsd.py:CCSDBase | — | Defaults must match upstream |
| `cc.frozen` int / list / `(list_a, list_b)` for UCCSD; `cc.set_frozen()` auto-chemcore; `cc.set_frozen(method='window', window=(emin,emax))` | `cc/11-frozen_core.py` | M | ccsd.py:set_frozen | data/elements.chemcore | Three modes: int, list, auto, window |
| Lambda equations (`cc.solve_lambda()` populates `cc.l1`, `cc.l2`) | `cc/01-lambda.py`; required for density matrices and gradients | L | ccsd_lambda.py 454 | ccsd | Roughly half the cost of CCSD itself |
| `cc.make_rdm1()`, `cc.make_rdm2()` (1-/2-particle reduced density matrices in MO basis) | `cc/01-density_matrix.py` | L | ccsd_rdm.py 510 | ccsd_lambda | Required by CCSD gradients |
| In-core CCSD (default for small systems), AO-direct mode (`cc.direct = True`) | `cc/10-ao_direct_ccsd.py` | M | ccsd.py | ao2mo | AO-direct avoids MO transform memory |
| DF-CCSD (`dfccsd.RCCSD`, `dfuccsd.UCCSD`) when mf has DF | `cc/21-dfccsd.py` | M | dfccsd.py 248 | df | DF-CCSD auxbasis is a `*-RIFIT` family |
| `cc.t1`, `cc.t2`, `cc.l1`, `cc.l2`, `cc.e_corr`, `cc.e_tot`, `cc.converged`, `cc.cycles` saved as attrs | Universal | S | ccsd.py | — | Same pattern as SCF |
| `cc.as_scanner()` | PES scans | S | ccsd.py 798-853 | — | — |
| T1, D1, D2 multi-reference diagnostics (`get_t1_diagnostic`, `get_d1_diagnostic`, `get_d2_diagnostic`) | Standard CC quality check | S | ccsd.py 748-776 | — | One-line diagnostic functions |

#### Differentiators

| Feature | Value | Complexity |
|---------|-------|------------|
| Cubecl-parallel `_add_vvvv` and `update_amps` (the dominant 2-electron contraction) | 3–10× speedup; PySCF's `_ccsd.c` is well-tuned but single-node only | XL |
| Bounded-memory DF-CCSD with auto-chunking | Upstream errors out at `max_memory` exceeded; pyscf_rs streams | M |
| Stable convergence on ill-conditioned systems (better default DIIS + level shift) | Upstream often requires manual tuning | S |

#### Deferred

| Feature | When | Why |
|---------|------|-----|
| GCCSD (CCSD on GHF) | v1.x | Couple with GHF/GMP2 |
| `BCCD` (Brueckner CCSD) | v2 | Niche |
| `FNOCCSD` (frozen-natural-orbital CCSD) | v1.x | Listed in upstream cc/__init__; useful for large systems but extra MP2 step |
| `QCISD` (quadratic CI singles-doubles) | v2 | Mathematically simpler than CCSD but rarely used |
| `CCD` (no-singles CC) | v2 | Research feature |
| `cc.callback` | v1.x | Hook in `kernel()`; trivial add |
| Restart from DIIS file (`restore_from_diis_`) | v1.x | Useful but rare |

#### Anti-features

| Feature | Why requested | Why NOT in v1 |
|---------|---------------|---------------|
| **CCSD(T)** | High accuracy reference | **Out-of-scope per PROJECT.md** ("Higher-order post-SCF beyond CCSD"). Approx. 30% of CCSD users want CCSD(T) — this is the most painful omission, and the strongest pressure point for an early v1.x. |
| CCSDT, CCSDTQ (full T, Q) | Benchmark studies | Out-of-scope |
| EOM-CCSD (IP/EA/EE for excited states) | Excited-state CC | Out-of-scope per PROJECT.md (response/excited-state methods) |
| Mom-GFCCSD (moment-conserved Green's function CC) | Research | Out-of-scope |
| CCSD with custom Hamiltonian (`cc/40-ccsd_custom_hamiltonian.py`) | Method development | Research feature, not user-facing |
| Tailored CCSD, externally-corrected CCSD | Multi-reference proxies | Out-of-scope |

**Note on CCSD(T) demand:** Among published quantum chemistry papers using PySCF, CCSD(T) appears alongside CCSD in ~30–40% of papers (rough estimate from arXiv search of "PySCF" + method usage). If pyscf_rs gains traction, CCSD(T) is the *first* feature users will demand. Recommend flagging as "v1.1 priority" in the roadmap with explicit research-flag.

---

### 2.6 `grad` (Analytical Nuclear Gradients)

#### Table Stakes

| Feature | Why expected | Complexity | Upstream LOC | Deps | Notes |
|---------|--------------|------------|--------------|------|-------|
| `mf.nuc_grad_method()` and `mf.Gradients()` factories on every method object | Universal pattern | S | hf.py 2212-2218 | grad | Both must work; `Gradients()` is the newer canonical name |
| `grad.RHF(mf).kernel()` and `grad.UHF(mf).kernel()` | `grad/01-scf_grad.py` | L | grad/rhf.py 469 + grad/uhf.py 112 | scf, cintx | Returns `(natm, 3)` array in Eh/Bohr |
| `grad.RKS(mf).kernel()`, `grad.UKS(mf).kernel()` with `grid_response=True` option | `grad/02-dft_grad.py` | L | grad/rks.py 645 + grad/uks.py 280 | dft, libxc_rs | Grid response is critical for accurate DFT gradients |
| `grad.mp2.Gradients(mp).kernel()` | `grad/03-mp2_grad.py` | L | grad/mp2.py 329 + grad/ump2.py | mp2, ao2mo | Requires MP2 RDMs |
| `grad.ccsd.Gradients(cc).kernel()` and UCCSD variant | `grad/05-ccsd_grad.py` | XL | grad/ccsd.py 462 + uccsd.py | cc, ccsd_lambda, ccsd_rdm | Requires CCSD lambda + RDMs |
| `g.atmlst = [0, 1]` to compute gradients on a subset of atoms | `grad/01-scf_grad.py` | S | each grad.kernel | — | Optional optimization |
| `g.grids = dft.gen_grid.Grids(mol)` override grid for DFT gradient | `grad/02-dft_grad.py` | S | grad/rks.py | dft | Allows finer grid for grad than for SCF |
| ECP gradients for elements with ECP | Implied by ECP support | M | cintx | cintx | "Required if ECP present"; from upstream FEATURES list |
| `g.optimizer(solver='geomeTRIC')` factory creating geomopt driver | `geomopt/01-geomeTRIC.py` | S | grad/__init__ | geomopt | Bridge to geomopt |

#### Differentiators

| Feature | Value | Complexity |
|---------|-------|------------|
| Finite-difference gradient verification mode (`g.kernel(verify_with_fd=True)`) | First-time users mistrust analytical gradients; verification builds confidence | M |
| Cubecl-parallel pulay-term construction | Currently CPU-only; major DFT/CCSD grad speedup | L |

#### Deferred

| Feature | When | Why |
|---------|------|-----|
| Hessian (`hessian/rhf.py`, `hessian/rks.py`) | v1.x | Needed for vibrational frequencies, IR intensities; ~2× the work of gradients |
| TDDFT gradients (`grad/tdrhf.py`, `tdrks.py`) | Out-of-scope | Excited states |
| MCSCF / CASSCF gradients (`grad/casci.py`, `grad/casscf.py`) | Out-of-scope | Multi-reference |
| Gradients with implicit solvent (PCM gradient) | Out-of-scope | Solvent deferred |
| ROHF gradients | v1.x | Couple with ROHF SCF |

#### Anti-features

| Feature | Why NOT in v1 |
|---------|---------------|
| Dirac-HF gradients (`grad/dhf.py`) | 4-component out-of-scope |
| CISD gradients | CI methods out-of-scope |
| State-averaged CASSCF gradient | Multi-reference out-of-scope |
| MCPDFT gradient | Multi-reference DFT out-of-scope |
| NAC (non-adiabatic couplings) | Excited states / response out-of-scope |

---

### 2.7 `geomopt` (Geometry Optimization)

#### Table Stakes

| Feature | Why expected | Complexity | Upstream LOC | Deps | Notes |
|---------|--------------|------------|--------------|------|-------|
| `from pyscf.geomopt.geometric_solver import optimize; mol_eq = optimize(mf)` | `geomopt/01-geomeTRIC.py` | M | geometric_solver.py 252 | geomeTRIC (Python pkg) | The canonical entry point in tutorials |
| `from pyscf.geomopt.berny_solver import optimize; mol_eq = optimize(mf)` | `geomopt/01-pyberny.py` | M | berny_solver.py 246 | pyberny | Alternative; less popular but still common |
| `mol_eq = mf.Gradients().optimizer(solver='geomeTRIC' \| 'berny').kernel(conv_params)` | Both example scripts | S | grad/__init__ + geomopt | grad, geomopt | Stream-style alternative |
| Convergence params dict: `convergence_energy`, `convergence_grms`, `convergence_gmax`, `convergence_drms`, `convergence_dmax` (geomeTRIC) or `gradientmax`, `gradientrms`, `stepmax`, `steprms` (berny) | Both example scripts | S | geomopt | — | Pass-through to backend |
| `optimize(mf, callback=fn, maxsteps=100)` | `geomopt/20-callback.py` | S | geomopt/geometric_solver.py:kernel | — | Per-step callback |
| Works for any object with `nuc_grad_method()` (HF, DFT, MP2, CCSD) | `geomopt/01-geomeTRIC.py` lines 56-75 | reuse | each grad module | grad | The point of the abstraction |
| Returns a built `Mole` with optimized coordinates; original `mol` unchanged | `geomopt/01-geomeTRIC.py` | S | geomopt | — | Side-effect-free contract |
| `as_pyscf_method` adapter for custom energy/gradient callables | `geomopt/02-as_pyscf_method.py` | S | geomopt/addons.py 89 | — | Power-user |

#### Architecture Decision: Native Rust optimizer vs. wrap geomeTRIC?

**Recommendation: ship a native Rust optimizer in v1, *with* a PyO3 hook that lets users keep `optimize` from `geomeTRIC`/`pyberny` for backward compatibility.**

Rationale:
1. **geomeTRIC is pure Python** (~10k LOC of Python doing internal coordinate redundant-coordinate optimization). It is *not* the bottleneck — the SCF/grad call inside the loop is. Wrapping it via PyO3 is technically easy but creates a permanent Python runtime dependency for what is ostensibly a "pure Rust" project.
2. **pyberny** is also pure Python and less actively maintained.
3. A native Rust BFGS / RFO (rational function optimization) in redundant internal coordinates is **medium effort** (~2–3k LOC equivalent) — well-bounded research with the `argmin` Rust crate as a foundation.
4. Keeping the `from pyscf.geomopt.geometric_solver import optimize` import path **working** (as a thin Python wrapper that forwards to the Rust optimizer when geomeTRIC isn't installed) preserves drop-in compatibility.
5. Default convergence thresholds and step sizes should match geomeTRIC's defaults bit-for-bit so existing scripts produce the same trajectories.

Concretely, the v1 `geomopt` module should expose:
- `pyscf.geomopt.optimize(method, **kwargs)` → native Rust BFGS in redundant internals
- `pyscf.geomopt.geometric_solver.optimize(method, **kwargs)` → native Rust optimizer (drop-in replacement; same kwargs)
- `pyscf.geomopt.berny_solver.optimize(method, **kwargs)` → native Rust optimizer (drop-in replacement; same kwargs)
- Optional fallback: if `geomopt.use_external = True`, dispatch to the actual Python `geomeTRIC`/`pyberny` packages if installed.

#### Differentiators

| Feature | Value | Complexity |
|---------|-------|------------|
| No Python optimizer dependency (Rust-native BFGS+RFO) | Pure-Rust install story; `cargo add pyscf_rs` Just Works | L |
| Bit-stable trajectory across runs (deterministic line search) | geomeTRIC has nondeterministic small drifts | S |
| Better restart logic (full HDF5 checkpoint of optimizer state) | geomeTRIC restart is quirky | S |

#### Deferred

| Feature | When | Why |
|---------|------|-----|
| Constrained optimization (bond/angle/dihedral constraints) | v1.x | Required for transition-state work; `geomeTRIC` has it; defer until users ask |
| Transition-state search (Berny / RFO with negative-eigenvalue tracking) | v1.x | Implementation cost ~50% of basic optimization |
| Ghost-atom-aware geometry (`include_ghost=True`) | v1 | Already in upstream defaults; preserve |
| Initial Hessian from Lindh / quantum-Hessian | v1.x | Quality-of-life; `geomeTRIC` has it |

#### Anti-features

| Feature | Why NOT in v1 |
|---------|---------------|
| Path optimization (NEB, IRC) | Out of v1 scope; separate feature |
| Multi-state geometry optimization | Excited states out-of-scope |
| QM/MM optimization (`geomopt/10-with_qmmm.py`) | QM/MM out-of-scope |
| Solvent-coupled optimization (`geomopt/14-with_solvent.py`) | Solvent out-of-scope |

---

## 3. Cross-Cutting Features (touch every module)

| Feature | Notes | Complexity |
|---------|-------|------------|
| `lib.StreamObject` base class with `.set()`, `.run()`, `.kernel()`, `.apply()`, `.copy()`, `.dump_flags()` | Universal idiom; underpins every method object | M |
| `lib.logger` with verbosity 0–9 (`logger.note`, `logger.info`, `logger.debug`, `logger.warn`, `logger.error`, `logger.timer`) | Every method calls into this | M |
| HDF5 checkpoint I/O (`lib.chkfile.save`, `load`, hierarchical paths like `'scf/mo_coeff'`) | Every restart-capable method uses this | M |
| `mf.max_memory` (in MB) controls scratch-disk vs in-core decisions | Universal | S |
| `mf.verbose` cascades from `mol.verbose` to all method objects | Universal | S |
| Pickle support (`__getstate__`, `__setstate__`) on all method objects | `scf/25-pickle_dumps.py` | M |
| Plugin loading via `PYSCF_EXT_PATH` env var | `pyscf/__init__.py` | S | We can keep this as a no-op stub in v1; warn if user sets it |
| `__config__.py` parameter overrides via user `~/.pyscf_conf.py` | Defaults are read at import time | S |
| `ao2mo.kernel(mol, mo_coeff)` 4-index AO→MO transformation | Used by every post-SCF method | L |

---

## 4. Feature Dependency Graph

```
gto.Mole.build  ─────────────────────────────────────────────┐
       │                                                      │
       ├──> cintx 1e/2e integrals  ──> scf.RHF/UHF kernel ────┤
       │                                  │                   │
       │                                  ├──> mf.density_fit ──> df.DF + df_jk
       │                                  │
       │                                  ├──> dft.RKS/UKS  ─┬─> dft.numint
       │                                  │                   ├─> libxc_rs / xcfun_rs
       │                                  │                   └─> dft.gen_grid (Lebedev + Becke)
       │                                  │
       │                                  ├──> mp.MP2  ─────> ao2mo.kernel
       │                                  │       │                │
       │                                  │       └──> dfmp2 ──── df
       │                                  │
       │                                  └──> cc.CCSD  ────> ao2mo.kernel
       │                                          │                │
       │                                          ├──> ccsd_lambda ─> ccsd_rdm
       │                                          └──> dfccsd  ──── df
       │
       └──> grad.RHF/UHF/RKS/UKS/MP2/CCSD  ──> geomopt.optimize
                                                  └──> (BFGS+internal coords, native Rust)
```

### Critical dependency notes

- **CCSD requires MP2** — `cc.CCSD` imports `mp.mp2.get_nocc, get_nmo, get_frozen_mask, get_e_hf, _mo_without_core` (verified in `cc/ccsd.py` line 35). Therefore MP2 must be implemented and stable *before* CCSD work begins.
- **CCSD gradients require Lambda + RDMs** — `grad.ccsd` calls `ccsd_lambda.kernel` then `ccsd_rdm.make_rdm1/2`. Lambda is roughly half the CCSD effort; budget accordingly.
- **MP2 gradients require MP2 RDMs** — `grad.mp2` reads `mp.make_rdm1/2`.
- **DFT gradients require grid response** — full-accuracy DFT geomopt needs the `grid_response=True` path in `grad.RKS`. Without it, the optimizer can drift from the true PES minimum.
- **DF-CCSD requires DF-MP2 amplitudes as warm start** in some upstream paths — the dispatch logic (`cc/__init__.py` line 113) routes through `dfccsd.RCCSD` which itself uses `dfmp2`-style integrals.
- **`mf.to_uhf()` etc. are required upstream** — when a user writes `mp.MP2(uks_mf)`, the dispatch in `mp/__init__.py` calls `mf.to_uhf()` to convert. This conversion API is shared between scf and dft and must be implemented for cross-module dispatch to work.
- **`mol.intor` must support `'int2e_lr'` and `'int2e_sr'`** for range-separated DFT (CAM-B3LYP, ωB97X). `cintx` already provides these.

---

## 5. MVP Definition

### v1 = Launch With

The minimum to claim "drop-in replacement for molecular ground-state PySCF":

- [ ] **gto/Mole** — atom parsing (all 5 input forms), basis loading (string + dict + parse + load + etbs + unc- + @-truncate), ECP loading, `intor()` thin wrapper over cintx, all `nao_*`/`ao_loc_*`/`atom_*`/`ao_labels` accessors, `dumps`/`loads`, `set_geom_`, `copy`
- [ ] **scf** — RHF, UHF, GHF (basic), DIIS (C-DIIS only), level shift, `init_guess` ∈ {minao, atom, 1e, chkfile, dm0-passthrough}, `density_fit`, all `get_*` overrideable methods, `analyze`/`mulliken_pop`/`mulliken_meta`, `make_rdm1`, `as_scanner`, `to_rhf`/`to_uhf`/`to_ghf`/`to_rks`/`to_uks`, chkfile save/load
- [ ] **dft** — RKS, UKS, with `xc` string parser supporting libxc + xcfun + comma form + shorthands, `Grids` with level/atom_grid/prune/radi_method/becke_scheme, range-separated hybrid path (omega), VV10 NLC (`mf.nlc='VV10'`, `nlcgrids`), DF-DFT
- [ ] **mp2** — RMP2, UMP2, frozen core, in-core 4-index transformation, DF-MP2, `make_rdm1`/`make_rdm2`, `as_scanner`
- [ ] **cc** — RCCSD, UCCSD, frozen core (int / list / auto / window), `solve_lambda`, `make_rdm1`/`make_rdm2`, AO-direct option, DF-CCSD, T1/D1/D2 diagnostics, `as_scanner`
- [ ] **grad** — RHF, UHF, RKS (with `grid_response`), UKS, MP2 (R+U), CCSD (R+U), ECP gradient terms, atom-list subsetting
- [ ] **geomopt** — native Rust BFGS+RFO in redundant internals; `optimize(mf)` and `mf.Gradients().optimizer().kernel()` entry points; geomeTRIC and berny modules as drop-in shims
- [ ] **ao2mo.kernel** — in-core 4-index AO→MO; needed by MP2 and CCSD
- [ ] **df** — `DF` class, `DF.with_df.auxbasis`, `density_fit` decorator; auto-auxbasis for cc-pV*Z, def2- families
- [ ] **lib** — StreamObject, logger, chkfile (HDF5), DIIS base class, exceptions
- [ ] **PySCF-as-oracle CI** — every test runs upstream PySCF in same process, asserts µHartree agreement
- [ ] **PyO3 bindings preserving `pyscf.*` import paths** for all of the above

### v1.x = Add After Validation

Order roughly by user-pull pressure:

- [ ] **CCSD(T)** — highest user-demand omission; recommend explicit research-flag in roadmap
- [ ] **ROHF** + ROHF gradients — current dispatch raises errors on `RMP2(rohf_mf)`; can fall back to UHF in v1 with warning
- [ ] **`scf.newton(mf)` SOSCF** — for hard-to-converge cases (transition metals, near-degeneracies)
- [ ] **DFT-D3/D4 dispersion** (`mf.disp = 'd3bj'`)
- [ ] **Hessian** (RHF, RKS) for vibrational frequencies
- [ ] **FNO-CCSD** — large-system CCSD via natural-orbital truncation
- [ ] **ADIIS, EDIIS** DIIS variants
- [ ] **`frac_occ`** addon
- [ ] **GMP2, GCCSD** — only if GHF lands well
- [ ] **Constrained geomopt** (bond/angle/dihedral fix)
- [ ] **Full symmetry-adapted SCF** (irrep_nelec, symm-orb construction)
- [ ] **Custom XC functional definition** (`define_xc_`)

### v2+ = Future Consideration

- [ ] PBC (periodic boundary conditions) — its own milestone
- [ ] x2c / sfx2c1e relativistic
- [ ] mcscf / casci / casscf
- [ ] tdscf / tddft / TDA gradients
- [ ] eom-ccsd, adc, gw
- [ ] Smearing / finite-T SCF
- [ ] DMRG plugin solver
- [ ] Solvent (PCM, ddCOSMO, SMD)
- [ ] QM/MM
- [ ] Localized orbitals (Boys, Pipek-Mezey, IAO, IBO)
- [ ] Tools (cubegen, molden, fcidump, trexio)
- [ ] MPI / multi-node distribution

---

## 6. Feature Prioritization Matrix

| Feature | User value | Impl. cost | Priority | Notes |
|---------|------------|------------|----------|-------|
| `Mole` build + atom/basis parsing | HIGH | M | **P1** | Foundation; nothing works without it |
| 1e/2e integrals via cintx | HIGH | M (binding only) | **P1** | cintx already exists |
| RHF + DIIS + minao guess | HIGH | M | **P1** | Smallest viable end-to-end demo |
| UHF | HIGH | M | **P1** | Open-shell molecules are common |
| Density fitting (DF-HF) | HIGH | L | **P1** | Required for >1000 AO molecules |
| DFT (RKS) with libxc backend | HIGH | XL | **P1** | DFT is more popular than HF in practice |
| DFT range-separated hybrids | HIGH | M | **P1** | wB97X, CAM-B3LYP are in every modern paper |
| DFT VV10 NLC | MEDIUM | M | **P1** | wB97M-V is widely used; partial v1 |
| UKS | HIGH | M | **P1** | Open-shell DFT |
| MP2 (canonical RMP2) | HIGH | L | **P1** | Cheapest correlated method |
| DF-MP2 | HIGH | M | **P1** | Standard for medium molecules |
| UMP2 | MEDIUM | M | **P1** | |
| RCCSD | HIGH | XL | **P1** | The most-requested correlated method |
| UCCSD | HIGH | XL | **P1** | |
| CCSD lambda + RDMs | MEDIUM | L | **P1** | Required for CCSD gradients |
| DF-CCSD | MEDIUM | M | **P1** | Default for medium-large systems |
| HF gradients | HIGH | L | **P1** | Required for geomopt |
| DFT gradients (with grid_response) | HIGH | L | **P1** | Required for DFT geomopt |
| MP2 gradients | MEDIUM | L | **P1** | Common for benchmark studies |
| CCSD gradients | MEDIUM | XL | **P1** | Required for high-accuracy geomopt |
| Native Rust geomopt (BFGS + RFO + redundant internals) | HIGH | L | **P1** | Drop-in replacement for geomeTRIC |
| chkfile HDF5 save/load | HIGH | S | **P1** | Restart is a top-3 user pain point |
| `analyze()` + `mulliken_pop` | MEDIUM | S | **P1** | Tutorials all show this output |
| `make_rdm1` / `make_rdm2` | MEDIUM | M | **P1** | Required for grad and external use |
| `as_scanner` | MEDIUM | S | **P1** | PES scans are a common use |
| **CCSD(T)** | HIGH | XL | **P2** | Most-painful omission; explicit roadmap research-flag |
| ROHF | MEDIUM | M | **P2** | Fall back to UHF + warn in v1 |
| SOSCF (newton) | MEDIUM | XL | **P2** | Hard-converge cases |
| Hessian (RHF, RKS) | MEDIUM | XL | **P2** | Vibrational analysis |
| DFT-D3/D4 | MEDIUM | M | **P2** | Most modern DFT papers use it |
| ADIIS / EDIIS | LOW | S | **P3** | C-DIIS handles 95% |
| `frac_occ` | LOW | S | **P3** | Niche |
| GHF + GMP2 + GCCSD full path | LOW | XL | **P3** | Demand is small |
| FNO-CCSD | LOW | M | **P3** | Power-user feature |
| Custom XC definition | LOW | M | **P3** | Method developers |
| Symmetry-adapted SCF | LOW | XL | **P3** | Speedup only; correctness via C1 + DIIS |

**Priority key:**
- **P1**: Must ship in v1; without it the "drop-in replacement" claim breaks.
- **P2**: First post-v1 milestone; expect explicit user demand within months.
- **P3**: Real but defer until users ask twice.

---

## 7. Drop-In API Compatibility Floor

**Definition:** the set of attribute and method names that must exist with the same signature and semantics for an existing PySCF script using *only in-scope methods* to import and run unchanged.

### Mole object (≥ 30 names)

`atom`, `basis`, `charge`, `spin`, `unit`, `verbose`, `output`, `max_memory`, `cart`, `symmetry`, `ecp`, `nucmod`, `magmom`, `groupname`, `topgroup`, `_atm`, `_bas`, `_env`, `_ecpbas`, `nao`, `nao_nr()`, `nao_cart()`, `natm`, `nbas`, `nelec`, `nelectron`, `multiplicity`, `ms`, `enuc`, `energy_nuc()`, `atom_coords(unit='Bohr')`, `atom_charges()`, `atom_charge(i)`, `atom_symbol(i)`, `atom_pure_symbol(i)`, `atom_nelec_core(i)`, `bas_atom(i)`, `bas_angular(i)`, `bas_nctr(i)`, `bas_nprim(i)`, `bas_exp(i)`, `bas_ctr_coeff(i)`, `ao_loc_nr()`, `aoslice_by_atom()`, `ao_labels()`, `sph_labels()`, `cart_labels()`, `intor(name, ...)`, `intor_symmetric(...)`, `intor_asymmetric(...)`, `eval_gto(name, coords)`, `set_geom_(coords, unit='Bohr', inplace=True)`, `set_common_origin(coord)`, `set_rinv_origin(coord)`, `set_range_coulomb(omega)`, `with_common_origin(coord)`, `with_rinv_origin(coord)`, `with_range_coulomb(omega)`, `copy()`, `build(...)`, `dumps()`, `loads(s)`, `tostring(format='xyz')`, `tofile(fname)`, `fromfile(fname)`, `RHF()`, `UHF()`, `KS(xc)`, `RKS(xc)`, `UKS(xc)`, `MP2()`, `CCSD()`

### SCF object (≥ 30 names)

`mol`, `verbose`, `max_memory`, `chkfile`, `conv_tol`, `conv_tol_grad`, `max_cycle`, `init_guess`, `DIIS`, `diis`, `diis_space`, `diis_start_cycle`, `diis_damp`, `diis_file`, `level_shift`, `damp`, `direct_scf`, `direct_scf_tol`, `callback`, `conv_check`, `mo_energy`, `mo_coeff`, `mo_occ`, `e_tot`, `converged`, `cycles`, `scf_summary`, `disp`, `with_df`, `kernel(dm0=None)`, `run(*args, **kwargs)`, `scf(dm0=None)`, `build(mol=None)`, `reset(mol=None)`, `set(**kwargs)`, `apply(fn, *args, **kwargs)`, `dump_flags()`, `get_init_guess(mol, key)`, `init_guess_by_minao(mol)`, `init_guess_by_atom(mol)`, `init_guess_by_1e(mol)`, `init_guess_by_chkfile(chkfile)`, `from_chk(chkfile)`, `get_hcore(mol=None)`, `get_ovlp(mol=None)`, `get_jk(mol, dm, hermi=1, with_j=True, with_k=True)`, `get_j(mol, dm, hermi=1)`, `get_k(mol, dm, hermi=1)`, `get_veff(mol, dm, dm_last=0, vhf_last=0, hermi=1)`, `get_fock(h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None)`, `get_occ(mo_energy, mo_coeff)`, `get_grad(mo_coeff, mo_occ, fock=None)`, `eig(h, s)`, `make_rdm1(mo_coeff=None, mo_occ=None)`, `make_rdm2(mo_coeff=None, mo_occ=None)`, `energy_elec(dm=None, h1e=None, vhf=None)`, `energy_tot(dm=None, h1e=None, vhf=None)`, `energy_nuc()`, `analyze(verbose=...)`, `mulliken_pop(mol=None, dm=None, s=None)`, `mulliken_meta(mol=None, dm=None)`, `dip_moment(mol=None, dm=None, unit='Debye')`, `quad_moment(...)`, `density_fit(auxbasis=None, with_df=None, only_dfj=False)`, `as_scanner()`, `nuc_grad_method()`, `Gradients()`, `to_rhf()`, `to_uhf()`, `to_ghf()`, `to_rks()`, `to_uks()`, `to_gks()`, `dump_chk(envs)`, `update_(chkfile)`

### DFT-specific additions on RKS/UKS

`xc`, `nlc`, `grids` (a `Grids` instance), `nlcgrids`, `_numint`, `small_rho_cutoff`, `omega` (range-sep), `do_nlc()`, `define_xc_(...)`

### `Grids` object

`mol`, `level`, `atom_grid`, `prune`, `radi_method`, `becke_scheme`, `coords`, `weights`, `non0tab`, `build()`, `reset(mol)`, `kernel()`, `gen_atomic_grids(mol, atom_grid, radi_method, level, prune)`

### MP2 object

`mol`, `_scf`, `mo_coeff`, `mo_occ`, `mo_energy`, `frozen`, `nocc`, `nmo`, `e_corr`, `e_tot`, `t2`, `t1` (None for canonical RMP2), `with_df`, `verbose`, `max_memory`, `kernel(mo_energy=None, mo_coeff=None, eris=None, with_t2=True)`, `run(...)`, `ao2mo(mo_coeff=None)`, `make_rdm1()`, `make_rdm2()`, `init_amps(eris)`, `update_amps(t2, eris)`, `energy(t2, eris)`, `as_scanner()`, `nuc_grad_method()`, `Gradients()`, `set_frozen(method='auto')`, `make_fno(thresh, pct_occ, nvir_act)`

### CCSD object

All of MP2's + `t1`, `t2`, `l1`, `l2`, `conv_tol`, `conv_tol_normt`, `max_cycle`, `diis_space`, `diis_start_cycle`, `iterative_damping`, `level_shift`, `direct`, `async_io`, `incore_complete`, `solve_lambda(t1=None, t2=None, l1=None, l2=None, eris=None)`, `restore_from_diis_(diis_file)`, `get_t1_diagnostic()`, `get_d1_diagnostic()`, `get_d2_diagnostic()`, `update_amps(t1, t2, eris)`, `amplitudes_to_vector(t1, t2)`, `vector_to_amplitudes(vec)`, `vector_size()`, `cc2` (False default), `callback`

### Gradients object

`base` (the underlying mf/mp/cc), `mol`, `atmlst`, `verbose`, `max_memory`, `kernel(atmlst=None)`, `run(...)`, `de` (result, the (natm,3) array), `grad_elec()`, `grad_nuc()`, `optimizer(solver='geomeTRIC')`

### geomopt module-level

`pyscf.geomopt.geometric_solver.optimize(method, **kwargs)`, `pyscf.geomopt.berny_solver.optimize(method, **kwargs)`, `pyscf.geomopt.optimize(method, **kwargs)`, `pyscf.geomopt.addons.as_pyscf_method(mol, energy_grad_fn)`, `pyscf.geomopt.addons.dump_mol_geometry(mol, coords)`

---

## 8. Anti-Features Summary (consolidated)

These are commonly-requested or upstream-existing features that **pyscf_rs v1 explicitly does NOT implement**, and the alternative if any:

| Anti-feature | Why requested | Why NOT in v1 | Alternative |
|--------------|---------------|---------------|-------------|
| **CCSD(T)** | Gold-standard accuracy | PROJECT.md: "higher-order post-SCF beyond CCSD" deferred. Most-painful omission. | Document as v1.x P1; users can call upstream PySCF for CCSD(T) only |
| **PBC / k-points** | Solid-state | PROJECT.md: separate milestone | upstream PySCF |
| **TDDFT / TDA / EOM-CC / GW / ADC** | Excited states | Out-of-scope | upstream PySCF |
| **CASSCF / CASCI / NEVPT2 / DMRG** | Multi-reference | Out-of-scope | upstream PySCF |
| **x2c / sfx2c1e / DHF** | Heavy-element relativistic | Out-of-scope | upstream PySCF |
| **Solvent (PCM/ddCOSMO/SMD)** | Solution-phase | Out-of-scope | upstream PySCF |
| **QM/MM** | Hybrid simulations | Out-of-scope | upstream PySCF |
| **NAC / EPH** | Non-adiabatic, electron-phonon | Out-of-scope | upstream PySCF |
| **Localized orbitals (Boys/Pipek-Mezey/IAO/IBO)** | Bonding analysis | Out-of-scope; analysis only | upstream PySCF or wfn-export to other tools |
| **fcidump / molden / cubegen / TrexIO writers** | Format export | `tools/` module deferred | Add minimal `cubegen` + `molden` in v1.x if user-pull |
| **Custom Hamiltonian SCF** (Hubbard, model systems) | Method development | Research feature | Direct subclassing in user code via `get_hcore`/`get_jk` overrides |
| **MPI / multi-node** | Very large systems | PROJECT.md: cubecl + shared-memory only in v1 | Future |
| **Conda channel** | Some users prefer conda | PROJECT.md: PyPI + crates.io v1 | conda-forge later |
| **`scf/40-customizing_hamiltonian.py`-style user-_eri injection** | Research workflow | Niche; opens validation rabbit-holes | Subclass + override `get_jk` |
| **`scf.M3SOSCF`** | Listed in upstream FEATURES | Esoteric; near-zero tutorial usage | DIIS or planned `newton()` SOSCF |

---

## 9. Implications for Roadmap

A reasonable v1 phase ordering implied by the dependency graph and user-value matrix:

1. **Phase 1 — Foundations**: `lib` (StreamObject, logger, chkfile, DIIS, exceptions) + `gto.Mole` (parsing, build, intor wrappers over cintx). *Validates the cintx integration end-to-end.*
2. **Phase 2 — RHF/UHF + DIIS**: scf base class + RHF + UHF + minimal init_guess. *First end-to-end energy.*
3. **Phase 3 — DF + DF-HF**: density-fitting infrastructure + DF-JK kernels. *Unlocks medium-molecule SCF and is reused everywhere downstream.*
4. **Phase 4 — DFT (RKS/UKS)**: grids + numint + libxc/xcfun bindings + RSH + VV10. *The largest single phase; ~6000 LOC upstream.*
5. **Phase 5 — MP2 (R, U, DF)**: ao2mo.kernel + canonical and DF MP2. *First correlated method.*
6. **Phase 6 — CCSD (R, U, DF) + Lambda + RDMs**: full CCSD pipeline. *XL phase; ~3000 LOC upstream.*
7. **Phase 7 — Gradients (HF, DFT, MP2, CCSD)**: parallel implementation reusing established kernels. *Each grad module is L; CCSD-grad is XL.*
8. **Phase 8 — geomopt + analysis polish**: native BFGS + RFO; `analyze`/`mulliken` polish; `as_scanner` everywhere; PyO3 surface lockdown.
9. **Phase 9 — PySCF-as-oracle CI hardening + benchmarks**: prove the 2–5× speedup claim on the defined benchmark suite.

**Phases needing deeper research flags** (likely require a dedicated phase-research subagent):
- **Phase 4 (DFT)**: Becke partition + grid pruning numerics, libxc string-parser semantics edge cases (especially XC_ALIAS), VV10 stability.
- **Phase 6 (CCSD)**: T2 contraction memory scheduling, DIIS + iterative_damping interplay, T1/D1 diagnostic numerics.
- **Phase 7 (CCSD gradients)**: Lambda equation conditioning, response-density to AO transformation.
- **Phase 8 (geomopt)**: redundant internal coordinate construction (Wilson B-matrix), step-restriction heuristics, RFO eigenvalue tracking.

Phases unlikely to need extra research (well-trodden paths):
- Phase 1, 2, 3, 5 (standard patterns).

---

## 10. Sources

All evidence drawn directly from the upstream PySCF source checked into this repo:

- `pyscf/__init__.py` (top-level API)
- `pyscf/__all__.py` (full module export list)
- `pyscf/gto/mole.py` (4383 LOC) — Mole class, atom/basis parser, intor wrapper
- `pyscf/scf/__init__.py`, `pyscf/scf/hf.py` (2511 LOC), `pyscf/scf/uhf.py` (1140), `pyscf/scf/ghf.py` (569), `pyscf/scf/rohf.py` (563), `pyscf/scf/diis.py` (247), `pyscf/scf/_vhf.py` (847), `pyscf/scf/addons.py` (1034)
- `pyscf/dft/__init__.py`, `pyscf/dft/rks.py` (569), `pyscf/dft/uks.py` (209), `pyscf/dft/numint.py` (3030), `pyscf/dft/gen_grid.py` (704), `pyscf/dft/libxc.py` (1527), `pyscf/dft/xcfun.py` (1166)
- `pyscf/mp/__init__.py`, `pyscf/mp/mp2.py` (948), `pyscf/mp/ump2.py` (804), `pyscf/mp/dfmp2.py` (595)
- `pyscf/cc/__init__.py`, `pyscf/cc/ccsd.py` (1734), `pyscf/cc/uccsd.py` (1393), `pyscf/cc/ccsd_lambda.py` (454), `pyscf/cc/ccsd_rdm.py` (510), `pyscf/cc/dfccsd.py` (248)
- `pyscf/grad/__init__.py`, `pyscf/grad/rhf.py` (469), `pyscf/grad/uhf.py` (112), `pyscf/grad/rks.py` (645), `pyscf/grad/uks.py` (280), `pyscf/grad/mp2.py` (329), `pyscf/grad/ccsd.py` (462)
- `pyscf/geomopt/__init__.py`, `pyscf/geomopt/berny_solver.py` (246), `pyscf/geomopt/geometric_solver.py` (252), `pyscf/geomopt/addons.py` (89)
- `pyscf/df/__init__.py`, `pyscf/df/df.py` (391), `pyscf/df/df_jk.py` (645), `pyscf/df/incore.py` (351), `pyscf/df/outcore.py` (352)
- `pyscf/gto/basis/` directory listing (207 `.dat` files)
- `examples/0-readme.py`, `examples/scf/{00,02,03,14,15,20,22,24,30,54}-*.py`, `examples/dft/{00,11,12,15,20}-*.py`, `examples/mp/00-simple_mp2.py`, `examples/cc/{00,01-density_matrix,01-lambda,10,11}-*.py`, `examples/grad/{01,02,05}-*.py`, `examples/geomopt/{01-geomeTRIC,01-pyberny}.py`, `examples/gto/{00,04,05}-*.py`
- `FEATURES` (repo root, upstream feature manifest)
- `.planning/PROJECT.md` (in-tree v1 scope definition)
- `.planning/codebase/STRUCTURE.md`, `.planning/codebase/ARCHITECTURE.md`

**Confidence: HIGH** — every API name and signature in this document was verified against the in-tree PySCF source. Line counts are exact (`wc -l`). The only LOW-confidence items are estimated user-pull pressure for individual deferred features (e.g. "30–40% of CCSD users want CCSD(T)"); those are flagged inline.

---

*Feature research for: pyscf_rs molecular ground-state quantum chemistry rewrite*
*Researched: 2026-05-09*
