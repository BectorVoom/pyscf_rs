# T13 — Kernel K-PP1: the AO Fourier transform block `A[q, g]`

**Goal.** A CubeCL kernel that writes
`A[q, g] = ft_ao(cell, Gv, kpt)[g, q] / sqrt(vol)` for one block of
G-points, AO-major, equal to the host `ft_ao_kpt` (validated against
upstream in T09).

## Read first (required before writing the kernel)

CubeCL manual: the same three pages as T12, plus
`11_launch_overhead_and_transfers.md` §5.

Repo:
- `crates/pyscf-pbc-df/src/ft_ao/single.rs` lines 46–157 — the HOST
  algorithm this kernel reproduces. Read it line by line.
- `crates/pyscf-pbc-df/src/ft_ao/mcmurchie.rs` lines 56–112 — `e_index`,
  `e_coefficients`.
- `crates/pyscf-kernels/src/pbc/ft_aopair.rs` lines 150–200 — the same
  polynomial loop already written as a kernel. Copy its structure.

## The value a lane writes

Lane = `(q, g)` with `q` a SPHERICAL AO index. With `Gk = Gv[g0+g] + kpt`,
`g2 = |Gk|²`, AO centre `R`, angular momentum `l`, `nt = l + 1`:

```
acc = 0 (complex)
for each primitive r of the AO's contraction:
    w = weight[r] · exp(−g2 / (4·alpha[r]))
    for each term t of the AO (non-zero cart→sph entry, powers ix,iy,iz):
        poly = Σ_{tt ≤ ix} Σ_{uu ≤ iy} Σ_{vv ≤ iz}
                 E[ix][tt]·E[iy][uu]·E[iz][vv] · Gk_x^tt · Gk_y^uu · Gk_z^vv · (−i)^(tt+uu+vv)
        acc += term_c[t] · w · poly
A[q, g] = acc · (cos θ + i·sin θ),   θ = −(Gk · R)
```

- `E[i][t]` for primitive `r` = `etab[prim_eoff[r] + i·nt + t]`, the data of
  `e_coefficients(l, 0, alpha, 0.0, 0.0, 1.0)` (its index is
  `e_index(0, l, i, 0, t) = i·(l+1) + t`).
- `(−i)^n`: `n % 4 = 0, 1, 2, 3` → `+re`, `−im`, `−re`, `+im`
  (`ft_aopair.rs:153-175`).
- `weight[r] = coef · cfac(l) · (π/alpha)^1.5 / sqrt(vol)` — computed on
  the host.

Host tables:

| table | type | content |
|---|---|---|
| `ao_r` | `f64`, `(nao, 3)` | centre of the AO's shell |
| `ao_l`, `ao_prim0`, `ao_nprim`, `ao_term0`, `ao_nterm` | `u32`, `(nao)` | `l`; primitive range; term range |
| `prim` | `f64`, `(nprim_total, 2)` | `alpha`, `weight` |
| `prim_eoff` | `u32`, `(nprim_total)` | offset of that primitive's `E` table |
| `etab` | `f64` | all `E` tables, `(l+1)²` values each |
| `term_c` | `f64` | non-zero `T_l[m, c]` |
| `term_pow` | `u32`, `(nterm, 3)` | `ix, iy, iz` |

## Do

1. In `crates/pyscf-kernels/src/pbc/pp_gspace.rs` add
   `pub struct PpFtAoTables` with exactly those fields (all `Vec`), plus
   `pub nao: usize`.
2. The kernel (concrete `f64`):

```rust
/// K-PP1 — one lane per `(q, g)`; index `q * nbs + g` (AO-major).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pp_ftao_kernel(
    gv: &Array<f64>,
    kpt: &Array<f64>,
    ao_r: &Array<f64>,
    ao_l: &Array<u32>,
    ao_prim0: &Array<u32>,
    ao_nprim: &Array<u32>,
    ao_term0: &Array<u32>,
    ao_nterm: &Array<u32>,
    prim: &Array<f64>,
    prim_eoff: &Array<u32>,
    etab: &Array<f64>,
    term_c: &Array<f64>,
    term_pow: &Array<u32>,
    a_re: &mut Array<f64>,
    a_im: &mut Array<f64>,
    nao: usize,
    nbs: usize,
    g0: usize,
    nb: usize,
) {
    let i = ABSOLUTE_POS;
    if i < nao * nbs {
        let q = i / nbs;
        let g = i % nbs;
        if g < nb {
            let o = (g0 + g) * 3;
            let gx = gv[o] + kpt[0];
            let gy = gv[o + 1] + kpt[1];
            let gz = gv[o + 2] + kpt[2];
            let g2 = gx * gx + gy * gy + gz * gz;

            let nt = ao_l[q] as usize + 1;
            let r0 = ao_prim0[q] as usize;
            let nr = ao_nprim[q] as usize;
            let t0 = ao_term0[q] as usize;
            let ntm = ao_nterm[q] as usize;

            let mut acc_re = 0.0;
            let mut acc_im = 0.0;
            for r in r0..(r0 + nr) {
                let alpha = prim[r * 2];
                let w = prim[r * 2 + 1]
                    * cube_math::double::exp::exp(0.0 - g2 / (4.0 * alpha), cube_math::MathConfig::EXACT);
                let eb = prim_eoff[r] as usize;
                for t in t0..(t0 + ntm) {
                    let ix = term_pow[t * 3] as usize;
                    let iy = term_pow[t * 3 + 1] as usize;
                    let iz = term_pow[t * 3 + 2] as usize;
                    let bx = eb + ix * nt;
                    let by = eb + iy * nt;
                    let bz = eb + iz * nt;
                    let mut poly_re = 0.0;
                    let mut poly_im = 0.0;
                    let mut gxp = 1.0;
                    for tt in 0..(ix + 1) {
                        let et = etab[bx + tt];
                        let mut gyp = 1.0;
                        for uu in 0..(iy + 1) {
                            let etu = et * etab[by + uu] * gxp * gyp;
                            let mut gzp = 1.0;
                            for vv in 0..(iz + 1) {
                                let ww = etu * etab[bz + vv] * gzp;
                                let n = (tt + uu + vv) % 4;
                                if n == 0 {
                                    poly_re += ww;
                                } else if n == 1 {
                                    poly_im -= ww;
                                } else if n == 2 {
                                    poly_re -= ww;
                                } else {
                                    poly_im += ww;
                                }
                                gzp *= gz;
                            }
                            gyp *= gy;
                        }
                        gxp *= gx;
                    }
                    let c = term_c[t] * w;
                    acc_re += c * poly_re;
                    acc_im += c * poly_im;
                }
            }

            let theta = 0.0 - (gx * ao_r[q * 3] + gy * ao_r[q * 3 + 1] + gz * ao_r[q * 3 + 2]);
            let (sn, cs) = cube_math::double::trig::sincos(theta, cube_math::MathConfig::EXACT);
            a_re[i] = acc_re * cs - acc_im * sn;
            a_im[i] = acc_re * sn + acc_im * cs;
        }
    }
}
```

3. `launch_pp_ftao_on_handles<R: Runtime>(…)` and a public test entry
   `pp_ftao_block(client, t: &PpFtAoTables, gv: &[f64], kpt: [f64; 3], g0,
   nb) -> Result<(Vec<f64>, Vec<f64>), AlgebraError>` — same shape and same
   validation style as T12 steps 3–4 (`launch_1d(client, nao * nbs, 400)`).
4. Table builder — in `crates/pyscf-pbc-df/src/pp_gspace.rs` (it needs
   `e_coefficients`, which lives in that crate):

```rust
/// K-PP1 tables for `cell`. `None` when a shell has `l > 3` or the Mole is
/// cartesian — the caller then uses the host route.
pub fn ftao_tables(cell: &Cell) -> Result<Option<pyscf_kernels::pbc::pp_gspace::PpFtAoTables>, PbcDfError>
```
   Walk the shells exactly as `ft_ao_mol` does (`single.rs:66-85`): for
   shell `ib`, for `ictr in 0..nctr`, for `m in 0..(2l+1)` → one AO, in that
   order (this is the AO order of `ft_ao_mol`'s output). Per `(ib, ictr)`:
   `nprim` primitive records with `alpha = env[pe + p]`,
   `weight = env[pc + ictr*nprim + p] · common_fac_sp(l) · (π/alpha)^1.5 /
   vol.sqrt()`, and `etab` extended by
   `e_coefficients(l, 0, alpha, 0.0, 0.0, 1.0).data`. Per `l`, the term list
   of row `m` of `cart2sph_l_matrix(l)`: every `c` with `T[m·nc + c] != 0.0`,
   powers from `cart_powers(l)[c]`. Keep every primitive, including one
   whose weight is `0.0`.
5. Test, in `crates/pyscf-pbc-df/tests/pp_gspace_device.rs` (new; it needs a
   real cell, so it lives in this crate). Reuse the cell builder by copying
   `ktao3` from `tests/pp_gspace_oracle.rs`. Client:
   `pyscf_algebra::select_backend().expect("backend").client`.
   - `ftao_block_matches_the_host_transform`: for `basis` in
     `["gth-szv-molopt-sr", "gth-dzvp-molopt-sr"]`, mesh `[5,5,5]`,
     `kpt = make_kpts_default(&cell, [3,3,1])[4]`: device
     `pp_ftao_block(…, g0 = 0, nb = 125)` against host
     `ft_ao_kpt(&cell.mol, &gv, kpt)` — remember the two layouts
     (`device[q*125 + g]` vs `host[g*nao + q]`) and multiply the host value
     by `1/vol.sqrt()`. Tolerance `1e-11 · (1 + |value|)`.

## Verify

1. CHECK `-p pyscf-kernels` and CHECK `-p pyscf-pbc-df`.
2. TEST `-p pyscf-pbc-df --test pp_gspace_device` → passes for both bases.
   Write both printed deviations to `PROGRESS.md`.

## If it fails

- Only `d` / `f` AOs are wrong: the term list. For `l ≥ 2` a spherical AO
  has several cartesian terms; print the terms of one failing AO and compare
  with row `m` of `cart2sph_l_matrix(l)`.
- All AOs wrong by a constant factor: `weight` (check `cfac`, `(π/α)^1.5`,
  `1/sqrt(vol)`).
- Wrong only when `kpt ≠ 0`: the kernel must add `kpt` to `gv` for BOTH the
  polynomial and `theta`.
- Stack overflow on the CPU runtime: a local array crept in. The kernel
  above has none.
