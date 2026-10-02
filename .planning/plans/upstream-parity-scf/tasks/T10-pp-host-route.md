# T10 — G-space route on the host, and the switch

**Goal.** `FFTDF.get_pp` and the band path use upstream's reciprocal-space
non-local pseudopotential by default, computed on the host in G-blocks.
T08's two oracle tests turn green. (Kernels replace the host inner loops in
T11–T14; the host route stays as the reference and fallback.)

## The mathematics (from `pyscf/pbc/df/fft.py:126-170`), per k-point

```
A[g, q]   = ft_ao(cell, Gv, kpt)[g, q] / sqrt(vol)          // (ngrids, nao)
B[row, q] = Σ_g S[row, g] · A[g, q]                          // (nrow, nao)
V[p, q]   = (1/vol) · Σ_channels Σ_{i,j < nproj} Σ_{m < 2l+1}
              conj(B[row0 + i·(2l+1) + m, p]) · h[i·nproj + j] · B[row0 + j·(2l+1) + m, q]
```

`V` is added to the local part. At Γ the imaginary plane of the SUM is set
to zero (the existing code in `vpp_add_nonlocal` already does that).

## Read first

- `crates/pyscf-pbc-df/src/fftdf.rs` lines 860–945 (`get_pp`,
  `pp_local_potential_r`, `vpp_add_nonlocal`, `get_hcore_nonlocal`).
- `crates/pyscf-pbc-dft/src/krks.rs` lines 386–394 (the one caller of
  `get_hcore_nonlocal`; `g.mesh` is the mesh in scope there).

## Do

1. In `crates/pyscf-pbc-df/src/pp_gspace.rs` add:

```rust
/// Which evaluation of the non-local pseudopotential FFTDF uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PpNonlocal {
    /// Upstream's: reciprocal space on the FFT mesh (`fft.py:114-176`).
    Reciprocal,
    /// Analytic, mesh-independent (`pyscf_pbc_gto::pseudo::get_pp_nl`).
    RealSpace,
}

/// `PYSCF_PBC_FFTDF_PP_NL=realspace` selects [`PpNonlocal::RealSpace`];
/// anything else (or unset) is upstream's [`PpNonlocal::Reciprocal`].
/// Read on every call — not cached — so one process can test both.
pub fn pp_nonlocal_route() -> PpNonlocal {
    match std::env::var("PYSCF_PBC_FFTDF_PP_NL") {
        Ok(v) if v.trim().eq_ignore_ascii_case("realspace") => PpNonlocal::RealSpace,
        _ => PpNonlocal::Reciprocal,
    }
}

/// G-points per block for a memory budget in MB (`A` is `nb × nao` and `S`
/// is `nrow × nb`, 16 bytes per complex value). At least 1024 points.
pub fn block_points(budget_mb: f64, nao: usize, nrow: usize, ngrids: usize) -> usize {
    let per_point = 16.0 * (nao + nrow) as f64;
    let nb = (budget_mb * 1024.0 * 1024.0 / per_point) as usize;
    nb.max(1024).min(ngrids.max(1))
}

/// `PYSCF_PBC_PP_NL_BUDGET_MB`, default 1024. Read on every call.
pub fn budget_mb() -> f64 {
    std::env::var("PYSCF_PBC_PP_NL_BUDGET_MB")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or(1024.0)
}

/// `B[row, q]` on the host, G walked in blocks of `block` points.
/// Planar, row-major `(nrow, nao)`.
pub fn project_host(
    cell: &Cell,
    t: &ProjTables,
    gv: &[[f64; 3]],
    kpt: [f64; 3],
    block: usize,
) -> Result<(Vec<f64>, Vec<f64>), PbcDfError> { /* loop g0 in (0..ngrids).step_by(block) */ }

/// `V[p, q]` from `B` — the sum over channels in the formula above.
/// Row-major `(nao, nao)`.
pub fn sandwich(t: &ProjTables, b_re: &[f64], b_im: &[f64], nao: usize, vol: f64) -> pyscf_algebra::CTensor { /* … */ }

/// The non-local pseudopotential at every k-point, reciprocal-space route.
pub fn get_pp_nl_gspace(
    cell: &Cell,
    mesh: [usize; 3],
    kpts: &[[f64; 3]],
) -> Result<Vec<pyscf_algebra::CTensor>, PbcDfError> {
    let gv = pyscf_pbc_gto::get_gv(cell, Some(mesh))?;
    let t = proj_tables(cell)?;
    let nao = cell.mol.nao_nr;
    let block = block_points(budget_mb(), nao, t.rows.len(), gv.len());
    let vol = cell.vol();
    let mut out = Vec::with_capacity(kpts.len());
    for kpt in kpts {
        let (b_re, b_im) = project_host(cell, &t, &gv, *kpt, block)?;
        out.push(sandwich(&t, &b_re, &b_im, nao, vol));
    }
    Ok(out)
}
```

   Inside `project_host`, per block: `ft_ao_kpt(&cell.mol, &gv[g0..g1], kpt)`
   returns planar `(nb, nao)` with index `g * nao + q`;
   `proj_values_host(cell, t, &gv[g0..g1], kpt)` returns `(nrow, nb)` with
   index `row * nb + g`. Accumulate
   `B[row * nao + q] += S[row, g] · A[g, q] / sqrt(vol)` as a complex product,
   with `g` ascending in the innermost loop.
   If `t.rows` is empty, `get_pp_nl_gspace` returns zero matrices.
2. `crates/pyscf-pbc-df/src/fftdf.rs`:
   - `vpp_add_nonlocal` gains a parameter `mesh: [usize; 3]`. Its first line
     becomes:

```rust
let vnl_c: Vec<CTensor> = match crate::pp_gspace::pp_nonlocal_route() {
    crate::pp_gspace::PpNonlocal::Reciprocal => crate::pp_gspace::get_pp_nl_gspace(cell, mesh, kpts)?,
    crate::pp_gspace::PpNonlocal::RealSpace => pyscf_pbc_gto::pseudo::get_pp_nl(cell, kpts)?
        .iter()
        .map(|m| forder_to_c(m, cell.mol.nao_nr, cell.mol.nao_nr))
        .collect(),
};
```
     and the loop body uses `zadd_assign(v, &vnl_c[k]);` (the `forder_to_c`
     call that was inside the loop is now in the `RealSpace` arm). Keep the
     Γ block that zeroes `v.im` exactly as it is.
   - `get_pp`: pass `df.mesh`.
   - `get_hcore_nonlocal(cell, kpts)` becomes
     `get_hcore_nonlocal(cell, mesh, kpts)` and passes `mesh` on.
3. `crates/pyscf-pbc-dft/src/krks.rs` line ~391: call
   `get_hcore_nonlocal(cell, g.mesh, kpts_band)`.
4. Replace the section "Deviation from upstream's `get_pp`" in the module
   header of `fftdf.rs` (lines 6–24) by:

```rust
//! # The non-local half of `get_pp`
//!
//! Default: upstream's reciprocal-space evaluation on the FFT mesh
//! (`fft.py:114-176`, [`crate::pp_gspace`]). `PYSCF_PBC_FFTDF_PP_NL=realspace`
//! selects the analytic [`pyscf_pbc_gto::pseudo::get_pp_nl`] instead, which
//! is exact in the basis and differs from upstream below a converged mesh.
```
5. In `crates/pyscf-pbc-df/tests/fftdf.rs` header (lines 17–25, "Why
   `get_pp` is gated at mesh >= 31") replace the paragraph by one sentence:
   `//! `get_pp` is gated against upstream at mesh 31 and at the coarse mesh 11.`

## Verify

1. ORACLE `-p pyscf-pbc-df --test pp_gspace_oracle` → 3 passed.
2. ORACLE `-p pyscf-pbc-df --test fftdf` → all pass, including
   `…_at_a_coarse_mesh`. Write the coarse-mesh deviation to `PROGRESS.md`.
3. TEST `-p pyscf-pbc-df --test fftdf` → same as baseline.
4. TEST `-p pyscf-pbc-dft --test krks_dzvp_small_cell --test numint_ao_budget`
   → same as baseline.
5. ORACLE `-p pyscf-pbc-dft --test kscf_conv_check_oracle --test scf_ovlp_smearing_oracle`
   → pass.

## If it fails

- Deviation exactly a factor `vol` or `sqrt(vol)` off: check the two
  `1/sqrt(vol)` (in `A`) and the one `1/vol` (in `V`).
- Deviation only at k-points ≠ Γ: `ft_ao_kpt` must receive `Gv` and `kpt`
  (it adds them itself); the projector phase uses `Gv`.
- A test OUTSIDE this task's list that compares FFTDF with another density
  fitting (GDF, AFTDF, RSDF, multigrid) at a coarse mesh starts failing:
  STOP and report its name and numbers. Do not change it.
- Verify 5 `scf_ovlp_smearing_oracle` gets WORSE than its tolerance: STOP
  and report.
