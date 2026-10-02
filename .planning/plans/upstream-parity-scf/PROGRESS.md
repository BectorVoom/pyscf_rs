# Progress — upstream-parity-scf

- T01 done — 10 test-result lines (8 TEST targets + 2 ORACLE targets), no failures. NOTE: baseline was captured with the T02/T03 working-tree changes already present (level-shift fix); the BEFORE binary copy may also post-date them.
- T02 done — `level_shift_formula` red test found already in tree (was green: fix pre-applied); formula asserted as upstream `F + (S − S·D·S)·factor`.
- T03 done — `level_shift` follows upstream (no ½), `level_shift_dms` default + ROHF `(dma+dmb)·½` override + all 7 forwarding wrappers; `level_shift_formula` 2 passed; `kscf`/`kscf_resume`/`grad_fc_hoist` as baseline; `krks_dzvp_small_cell` passes un-retuned; `cargo check -p pyscf-py` OK.
- T04 done — `S·D`, `(S·D)·S` via device `zgemm_dense` (client once per call); `level_shift_formula` 2 passed at unchanged tolerances; `kscf` 9 passed + `kscf_resume` bit-identical OK.
- T05 done — `kscf_conv_check_oracle` passes (Γ mo_energy max|Δ| 2.7e-8 vs stock; e_tot |Δ| 2.1e-13). NOTE: passed on first run because the T06 final-diagonalisation was already present in the working tree (pre-applied, uncommitted); upstream probe confirms conv_check=True vs False differ by exactly the 0.2 shift, so the test is a valid gate.
- REVIEW AGENT UNAVAILABLE: `opencode/claude-opus-5-5` subagent failed with "Insufficient account funds". Proceeding with self-review against task specs instead.
- T06 done (code pre-applied in tree; verified) — ORACLE `kscf_conv_check_oracle` passes; `kscf`/`kscf_resume`/`grad_fc_hoist` as baseline; ORACLE `scf_ovlp_smearing_oracle` passes (re-run post-T04: e_tot |Δ| 1.1e-9).
- T07 BLOCKED (STOP rule, see tasks/REPORT-T07.md) — pipeline builds; `YTA_CONV_CHECK`/`conv_check`/identity/`"final"` history wired; but KTaO3 SZV recipe does not converge in 80 (or 150) cycles so the final step never runs (both greps 0, want 1). Upstream 2.12.1 on the identical recipe also fails (diverges, |g|→0.11): recipe stiffness, not a port regression. Resume mechanics OK (second run resumes at cycle 80, no identity churn).
T06 done — conv_check oracle passes (mo_energy max|Δ| 2.728e-8); kscf suite same as baseline; smearing oracle passes
T08 done — coarse-mesh diamond get_pp FAILS at 1.8456e-3; KTaO3 mesh-21 get_pp FAILS at 1.3960e-2 (both red as expected)
T07 partial — pipeline wiring builds + history carries final flag; verify recipe does not converge in 80/150 cycles for BEFORE binary, port, OR upstream 2.12.1 itself (see REPORT-T07.md); T03 identity holds to 3e-14; T06 oracle proves the mechanism
T09 done — projector table 1.0447e-14 (<1e-12); host ft_ao 1.0658e-14 SZV / 1.5099e-14 DZVP (<1e-11); get_pp coarse still FAILS as expected
T10 done — host G-space route default; KTaO3 get_pp 3.9e-13, diamond coarse 1.3e-13; krks_dzvp_small_cell re-derived from upstream (see REPORT-T10.md); all other gates pass
T11 done — pbc_pp_fold 2 passed (host 1e-12, bit-identity across blocks)
T12 done — pbc_pp_proj 2 passed (1e-12 formula, to_bits offset); pbc_pp_fold still green
T13 done — ftao_block 1.3145e-16 SZV+DZVP (<1e-11)
T14 done — device_projection 4 passed (<=2.6e-14, bit-identical blocks); oracles pass on device/host/tiny-budget; KTaO3 DZVP h1e: reciprocal-device 16.6s vs analytic-realspace 20.7s

## Review agent (opus 5.5) status
- 2026-10-02: spawn failed: "Insufficient account funds". Will retry after T15/T16.
T15 done — example builds; ktrun unittest OK (15); old checkpoint refused default (exit 6), accepted realspace (exit 0)
Review agent retry failed again: Insufficient account funds. Will retry once more after T16.
T16 done — exit 0; s2 d_e_tot 4.98e-10 (<1e-7), converged T/T; bands 3.37e-6 (<1e-5); d_gap 8.60e-6 eV (<1e-4); pre/s1 cycles 12/13, 13/14
T17 done — REPORT-T17.md written (runner sha a65d0ce4, config ktao3-t4.json, owner commands); OWNER ACTION pending
Review agent retry #3 failed again: Insufficient account funds.

## Review (2026-10-03)

- Restored three files that had only been reformatted (`pyscf-kernels/src/eval_gto.rs`, `pbc/band_vmat.rs`, `pyscf-pbc-df/src/fft_jk.rs`); trimmed reformat-only hunks from `fftdf.rs`, both `lib.rs`, `kscf_resume.rs`.
- NEW FIX (D4, found via REPORT-T10's "gradient exactly 0.0" note): smeared KRKS/KUKS/KUHF used the occupied-virtual gradient; upstream uses the full lower triangle (`pbc/scf/smearing.py:152-164`). `KOverrideHooks::smeared()` + default `get_grad` now select it; `Krhf`'s private copy removed. Effect: cycle counts equal upstream's (TaO SZV 21, DZVP 22; KTaO3 13/14/3), and the SZV `converged` flag now matches upstream (false).
- `krks_dzvp_small_cell.rs`: both flags asserted (false, false); the `pre` SCF of test 2 runs with `conv_check: false`, like the pipeline's `pre` stage.
- `pp_gspace::project_host`: parallel over rows, one accumulator per element across blocks (bit-identical at any block size, new test), no work for a cell without projectors.
- T07 closed with the T16 recipe (see REPORT-T07.md "Resolution"); the task file's recipe was wrong.
- T16 limits were too tight for a 1e-6 SCF: bands are reproducible to ~2e-5 Ha there (convergence noise in BOTH codes). Added a tight (1e-10) run as the real parity gate: s2 2.8e-14 Ha, bands 6.1e-8 Ha, gap 2.1e-6 eV, identical cycle counts.
- Suites after the fixes: pyscf-kernels (pbc_pp_fold, pbc_pp_proj, pbc_gv, pbc_band_vmat), pyscf-pbc-scf (level_shift_formula, kscf, kscf_resume, grad_fc_hoist), pyscf-pbc-df (fftdf, pp_gspace_device + oracles fftdf, pp_gspace_oracle), pyscf-pbc-dft (krks_dzvp_small_cell, numint_ao_budget + oracles kscf_conv_check_oracle, scf_ovlp_smearing_oracle) — all pass.

## Follow-up (2026-10-03): MINAO guess and DIIS basis — full trajectory parity

- D6 MINAO: `pyscf-scf::init_guess_by_minao` had only the all-electron path; ported upstream's ECP/pseudopotential branch (`hf.py:366-437`: `atom_nelec_core`, `core_configuration`, input-basis valence occupation when `|det s12| > 0.1`). Guess density vs upstream: KTaO3 SZV 1.3e-15, DZVP 2.7e-14, diamond 6.7e-16 (`pyscf-pbc-scf/tests/init_guess_minao_pseudo_oracle.rs`). All-electron path unchanged (pyscf-scf suite green).
- D5 DIIS: the error vector is now `C^H (FDS − SDF) C` in the orthonormal basis of the initial Fock eigenvectors (`hf.py:152-157`, `diis.py:89-110`); products on the device GEMM above 64 AOs.
- KTaO3 pipeline at the production tolerance (1e-6) vs STOCK PySCF: EVERY cycle of pre/s1/s2 within 7.4e-13 Ha, cycle counts 13/14/3 identical, bands 4.3e-11 Ha, gap 6.5e-13 eV. TaO test energies now equal upstream's printed digits (SZV −74.12862803).
- T4 (CUDA, boomvector, runner 4e9ab7aa — before D5/D6): identical to the local CPU run (stage energies to 12 digits, bands 7e-12 Ha), 0.12 h, 2.8 GB GPU. Rerun with the D5/D6 runner: see the T17 line below.
- T17 done on CUDA (T4, boomvector, runner 2c9e215b): every cycle within 7.4e-13 Ha of stock PySCF, bands 4.3e-11 Ha, gap 6.5e-13 eV.
