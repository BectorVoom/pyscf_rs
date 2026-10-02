# T14 — Device driver and memory budget

**Goal.** One device call per `get_pp` runs K-PP1 → K-PP2 → K-PP3 block by
block and returns only the small matrix `B` per k-point. It becomes the
default; the host route (T10) stays as fallback and reference.

```
upload once:  gv, all tables
allocate once: A (nao × nbs), S (nrow × nbs)          nbs = points per block
for each k-point:
    B = zeros (nrow × nao)
    for g0 in 0, nbs, 2·nbs, … :   nb = min(nbs, ngrids − g0)
        K-PP1 → A      K-PP2 → S      K-PP3: B += S·A
    read B back                                       the only read-back
```

Device memory = `16·nbs·(nao + nrow)` bytes + tables. For YTa7O19 DZVP
(nao 910, nrow 340, 249 615 points) the default 1024 MB budget gives
`nbs = 53 687` → 5 blocks per k-point. A T4 (15 GB) has room for it.

## Read first (required)

CubeCL manual: `11_launch_overhead_and_transfers.md` §2, §3, §5;
`13_memory_preallocation.md`.

Repo: `crates/pyscf-kernels/src/pbc/ft_aopair.rs` lines 268–300 (`launch`:
upload all tables into a `Vec<Handle>`, launch on handles, one batched
read); `crates/pyscf-pbc-df/src/ft_ao/mod.rs` lines 660–675 (how this crate
obtains a client and maps a kernel error to `PbcDfError::Backend`).

## Do

1. `crates/pyscf-kernels/src/pbc/pp_gspace.rs`:

```rust
/// Everything one `pp_gspace_project` call uploads.
#[derive(Debug, Clone, Default)]
pub struct PpGspaceTables {
    /// `(ngrids, 3)` mesh G-vectors, flattened (NOT shifted by k).
    pub gv: Vec<f64>,
    pub proj: PpProjTables,
    pub ftao: PpFtAoTables,
}

/// `B[row, q]` for every k-point: planar, row-major `(nrow, nao)`.
/// `block` = G-points per block (clamped to `1..=ngrids`).
pub fn pp_gspace_project(
    client: &AlgebraClient,
    t: &PpGspaceTables,
    kpts: &[[f64; 3]],
    block: usize,
) -> Result<Vec<(Vec<f64>, Vec<f64>)>, AlgebraError>
```
   Inner `fn run<R: Runtime>(client: &ComputeClient<R>, …)`:
   - validate with the same checks as `pp_proj_block` and `pp_ftao_block`;
   - upload `gv` and every table ONCE, before the k loop;
   - `nbs = block.clamp(1, ngrids)`; allocate `a_re, a_im` with
     `client.empty(nao * nbs * 8)` and `s_re, s_im` with
     `client.empty(nrow * nbs * 8)` ONCE;
   - per k-point: upload `kpt` (3 values) and a zero `B`
     (`upload::<R, f64>(client, &vec![0.0; nrow * nao])` twice); loop the
     blocks calling the three `launch_*_on_handles` functions of T11–T13
     (`launch_pp_fold_on_handles::<R, f64>`); then
     `client.read(vec![b_re, b_im])` — one call per k-point.
   - `nrow == 0` → return zero-length-row results without launching.
2. `crates/pyscf-pbc-df/src/pp_gspace.rs`:
   - `pub fn proj_kernel_tables(cell: &Cell, t: &ProjTables) ->
     Result<pyscf_kernels::pbc::pp_gspace::PpProjTables, PbcDfError>`:
     per row, `row_r` = atom position; `row_par` = `[0.5·rl², rl^(l+1.5)·
     π^1.25·common_fac_sp(l), c0, c1, c2]` (`qli_coeffs(l, i)`); terms = the
     non-zero entries of row `m` of `cart2sph_l_matrix(l)` with
     `cart_powers(l)`.
   - `pub fn project_device(cell, t: &ProjTables, gv: &[[f64; 3]], kpts,
     block) -> Result<Option<Vec<(Vec<f64>, Vec<f64>)>>, PbcDfError>`:
     `None` when `ftao_tables(cell)?` is `None`; otherwise build
     `PpGspaceTables`, get the client as `ft_ao/mod.rs:667` does, call
     `pp_gspace_project`, map the error to `PbcDfError::Backend`.
   - In `get_pp_nl_gspace` (T10): use the host route when the variable
     `PYSCF_PBC_PP_NL_HOST` is `1` or `project_device` returned `None`;
     otherwise use the device result. `sandwich` is unchanged and stays on
     the host: `B` is `nrow × nao` (5 MB for YTa7O19), smaller than the
     `nao × nao` answer.
   - Log once per call:
     `tracing::info!(ngrids, nao, nrow, block, device, "pp_gspace: non-local pseudopotential")`.
3. Append to `crates/pyscf-pbc-df/tests/pp_gspace_device.rs`:
   - `device_projection_matches_the_host` — KTaO3 SZV, mesh `[9,9,9]`, all
     9 k-points of `[3,3,1]`: `project_device(..., block = 729)` against
     `project_host(..., block = 729)` per k-point, `1e-10·(1 + |v|)`.
   - `device_projection_is_bit_identical_across_block_sizes` —
     `block = 729` against `block = 100` (8 blocks, last one ragged):
     `to_bits()` equal for every element, both planes, every k-point.
   - `block_points_respects_the_budget` — plain function:
     `block_points(1024.0, 910, 340, 249_615) == 53_687`,
     `block_points(1.0, 910, 340, 249_615) == 1024`,
     `block_points(1024.0, 27, 30, 729) == 729`.

## Verify

1. TEST `-p pyscf-pbc-df --test pp_gspace_device` → all pass.
2. ORACLE `-p pyscf-pbc-df --test pp_gspace_oracle --test fftdf` → all pass
   (device route).
3. The same ORACLE command with `PYSCF_PBC_PP_NL_HOST=1` in front → all
   pass (host route).
4. The same ORACLE command with `PYSCF_PBC_PP_NL_BUDGET_MB=0.01` in front →
   all pass (many small blocks).
5. Time one call. Run the pipeline to its one-electron stage on KTaO3 DZVP,
   device route then analytic route, and write both `h1e.bin: computed in …`
   times to `PROGRESS.md`:

```bash
cargo build --release -p pyscf-pbc-dft --example yta7o19_bands -j12
U=/home/user/Documents/workspace/.yta_bundle/upcheck
for r in reciprocal realspace; do rm -rf $U/t14_$r; mkdir -p $U/t14_$r
  env PYSCF_PBC_FFTDF_PP_NL=$r YTA_CELL=$U/ktao3.json YTA_CKPT_DIR=$U/t14_$r \
      YTA_BASIS=gth-dzvp-molopt-sr YTA_KE=60 YTA_KMESH=3,3,3 YTA_STOP_AFTER=1e \
      target/release/examples/yta7o19_bands 2>&1 | grep "h1e.bin"; done
```

## If it fails

- Verify 1 bit-identity fails but the 1e-10 test passes: K-PP3 was launched
  with `nb` where `nbs` belongs (or the reverse) on the ragged last block.
  `nbs` is the buffer stride and never changes; `nb` is the valid count.
- Device result is zero everywhere: `B` was created with `client.empty`
  instead of an uploaded zero vector, or the read-back used the wrong
  handles.
- Out-of-memory on the CPU runtime at Verify 5: lower the budget
  (`PYSCF_PBC_PP_NL_BUDGET_MB=256`) and report the number that worked.
