# T12 — Kernel K-PP2: the projector block `S[row, g]`

**Goal.** A CubeCL kernel that writes the projector table for one block of
G-points, equal to T09's `proj_values_host`.

## Read first (required before writing the kernel)

CubeCL manual:
- `Cubecl_conditionals.md` — "Avoid If Expressions".
- `Cubecl_loop_control.md` — whole file.
- `cubecl_error_solution_guide/calling a “normal” Rust function from inside
  a cube macro function fails in CubeCL.md`.

Repo templates:
- `crates/pyscf-kernels/src/pbc/struct_factor.rs` lines 21–85 — a concrete
  `f64` kernel that calls `cube_math::double::trig::sincos`.
- `crates/pyscf-kernels/src/pbc/ft_aopair.rs` lines 27–37 (why transcendental
  kernels are concrete `f64`), 184–187 (`exp`, `sincos` calls), 262–266
  (`upload_u32`).

## The value a lane writes

With row `p`, block point `g` (global point `g0 + g`), `Gv` the mesh
G-vector, `Gk = Gv + kpt`, `q2 = |Gk|²`:

```
poly = Σ_{t in terms(p)} term_c[t] · Gk_x^ix · Gk_y^iy · Gk_z^iz
val  = poly · coef(p) · exp(−alpha(p)·q2) · (c0 + c1·x2 + c2·x2²),   x2 = q2 · 2·alpha(p)
S    = val · (cos θ + i·sin θ),   θ = +(Gv · R_atom(p))
```

Host-side tables (built in T14 from T09's `ProjTables`):

| table | type | content |
|---|---|---|
| `row_r` | `f64`, `(nrow, 3)` | atom position of the row |
| `row_par` | `f64`, `(nrow, 5)` | `alpha = ½·rl²`, `coef = rl^(l+1.5)·π^1.25·cfac(l)`, `c0`, `c1`, `c2` |
| `row_term0`, `row_nterm` | `u32`, `(nrow)` | this row's range in the term tables |
| `term_c` | `f64` | non-zero entries `T_l[m, c]` |
| `term_pow` | `u32`, `(nterm, 3)` | `ix, iy, iz` of that entry (each ≤ 3) |

## Do

1. In `crates/pyscf-kernels/src/pbc/pp_gspace.rs` add the table struct:

```rust
/// Host tables for K-PP2. Built by `pyscf_pbc_df::pp_gspace`.
#[derive(Debug, Clone, Default)]
pub struct PpProjTables {
    pub row_r: Vec<f64>,
    pub row_par: Vec<f64>,
    pub row_term0: Vec<u32>,
    pub row_nterm: Vec<u32>,
    pub term_c: Vec<f64>,
    pub term_pow: Vec<u32>,
}
```

2. The kernel (concrete `f64`: it calls `cube_math::double`):

```rust
/// K-PP2 — one lane per `(row, g)`; index `row * nbs + g`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pp_proj_kernel(
    gv: &Array<f64>,
    kpt: &Array<f64>,
    row_r: &Array<f64>,
    row_par: &Array<f64>,
    row_term0: &Array<u32>,
    row_nterm: &Array<u32>,
    term_c: &Array<f64>,
    term_pow: &Array<u32>,
    s_re: &mut Array<f64>,
    s_im: &mut Array<f64>,
    nrow: usize,
    nbs: usize,
    g0: usize,
    nb: usize,
) {
    let i = ABSOLUTE_POS;
    if i < nrow * nbs {
        let p = i / nbs;
        let g = i % nbs;
        if g < nb {
            let o = (g0 + g) * 3;
            let gx = gv[o];
            let gy = gv[o + 1];
            let gz = gv[o + 2];
            let qx = gx + kpt[0];
            let qy = gy + kpt[1];
            let qz = gz + kpt[2];
            let q2 = qx * qx + qy * qy + qz * qz;

            let alpha = row_par[p * 5];
            let coef = row_par[p * 5 + 1];
            let x2 = q2 * 2.0 * alpha;
            let ql = row_par[p * 5 + 2] + row_par[p * 5 + 3] * x2 + row_par[p * 5 + 4] * x2 * x2;

            let t0 = row_term0[p] as usize;
            let nt = row_nterm[p] as usize;
            let mut poly = 0.0;
            for t in t0..(t0 + nt) {
                let ix = term_pow[t * 3] as usize;
                let iy = term_pow[t * 3 + 1] as usize;
                let iz = term_pow[t * 3 + 2] as usize;
                let mut w = term_c[t];
                if ix == 1 { w *= qx; }
                if ix == 2 { w *= qx * qx; }
                if ix == 3 { w *= qx * qx * qx; }
                if iy == 1 { w *= qy; }
                if iy == 2 { w *= qy * qy; }
                if iy == 3 { w *= qy * qy * qy; }
                if iz == 1 { w *= qz; }
                if iz == 2 { w *= qz * qz; }
                if iz == 3 { w *= qz * qz * qz; }
                poly += w;
            }

            let rad = coef * cube_math::double::exp::exp(0.0 - alpha * q2, cube_math::MathConfig::EXACT);
            let val = poly * rad * ql;
            let theta = gx * row_r[p * 3] + gy * row_r[p * 3 + 1] + gz * row_r[p * 3 + 2];
            let (sn, cs) = cube_math::double::trig::sincos(theta, cube_math::MathConfig::EXACT);
            s_re[i] = val * cs;
            s_im[i] = val * sn;
        }
    }
}
```

3. `launch_pp_proj_on_handles<R: Runtime>(client, gv, kpt, tables: &[Handle;
   6], s_re, s_im, t: &PpProjTables, ngrids, nrow, nbs, g0, nb)` — shape of
   `launch_on_handles` in `ft_aopair.rs:218-258`; `launch_1d(client, nrow *
   nbs, 60)`; `kernel::launch_unchecked::<R>(…)` (not generic → only `R`).
4. Public test entry:

```rust
/// K-PP2 on host slices for one block: `(s_re, s_im)`, each `(nrow, nb)`
/// row-major (the test passes `nbs = nb`).
pub fn pp_proj_block(
    client: &AlgebraClient,
    t: &PpProjTables,
    gv: &[f64],        // (ngrids, 3) flattened
    kpt: [f64; 3],
    g0: usize,
    nb: usize,
) -> Result<(Vec<f64>, Vec<f64>), AlgebraError>
```
   Validate: `gv.len() % 3 == 0`, `g0 + nb <= ngrids`, table lengths
   (`row_r = 3·nrow`, `row_par = 5·nrow`, `row_nterm = nrow`, `term_pow =
   3·term_c.len()`), every `row_term0[p] + row_nterm[p] <= term_c.len()`,
   every power `≤ 3`. Upload `f64` tables with `upload::<R, f64>`, `u32`
   tables with a local `upload_u32` copied from `ft_aopair.rs:262-266`.
   Outputs: `client.empty(nrow * nb * 8)`. One batched read.
5. Create `crates/pyscf-kernels/tests/pbc_pp_proj.rs`: build SMALL tables
   by hand (no chemistry needed) — 3 rows:
   - row 0: `alpha 0.4`, `coef 1.3`, `c = [2.0, 0.0, 0.0]`, one term
     `(1.0; 0,0,0)`, atom at `[0.1, 0.2, 0.3]`
   - row 1: `alpha 0.7`, `coef 0.9`, `c = [5.0, -1.0, 0.0]`, one term
     `(0.5; 1,0,0)`, atom at `[1.0, -0.5, 0.25]`
   - row 2: `alpha 0.3`, `coef 1.1`, `c = [63.0, -18.0, 1.0]`, two terms
     `(0.7; 1,1,0)` and `(-0.3; 0,0,2)`, atom at `[-0.4, 0.6, 0.9]`
   and 40 pseudo-random G-vectors in `[-3, 3]`, `kpt = [0.11, -0.07, 0.05]`.
   Host reference: the formula above in plain Rust (`f64::exp`,
   `f64::sin_cos`). Tests:
   - `projector_block_matches_the_host_formula` — whole range, `1e-12`
     relative to `1 + |value|`.
   - `a_block_offset_selects_the_right_points` — `g0 = 17`, `nb = 9` equals
     entries `17..26` of the full result, `to_bits` identical.

## Verify

1. CHECK `-p pyscf-kernels`.
2. TEST `-p pyscf-kernels --test pbc_pp_proj --test pbc_pp_fold` → all pass.

## If it fails

- `E0433 … is not a crate or module` inside the kernel: you called a normal
  Rust function. Only `cube_math::double::…` calls and operators are
  allowed; open the error guide file for this error.
- Type error on `w *= qx` after an `if`: keep `let mut w = term_c[t];`
  exactly as written (no `if` expression).
- Values wrong by a sign in the imaginary plane: `theta` must be `+G·R`
  (this is `conj(SI)`).
