//! Cartesian-to-spherical coefficients for `CINTc2s_ket_sph1`.
//!
//! Source: `sunqm/libcint` tag `v6.1.3`, file `src/cart2sph.c`, array
//! `g_trans_cart2sph` (offsets `+10` for d, `+40` for f, `+110` for g).
//! Every value is the C decimal literal digit for digit (the default,
//! non-`PYPZPX` build, which the wheel uses — probed: upstream p-shell
//! AO order at `(0.5, 0, 0)` is `(px, py, pz)`). A probe against
//! `pyscf.gto.mole.cart2sph(l)` (which calls `CINTc2s_ket_sph` on an
//! identity matrix) shows each literal parses to the same `f64` bits as
//! the compiled table (`target/probes-task2/coeff_probe.py`, 0 mismatches).
//!
//! Only the entries the `d/f/g_ket_cart2spheric` transforms read are
//! exported, named `C<l>_C<offset>` by table offset within the l-block.
//!
//! The ket transforms below mirror `d/f/g_ket_cart2spheric` statement for
//! statement: one `i` loop over `nbra` grid points, `gsph[m*lds+i]` accumulated
//! left to right (`(a*x + b*y) + c*z`), no zero-skipping beyond the table's
//! own sparsity (unreferenced zeros are simply not read), plain `*`/`+`
//! (Barcelona has no FMA and the C uses none).

#![allow(clippy::excessive_precision)]
// `1 * nbra` below spells C's `gcart[1*nbra+i]` index digit for digit; do not
// "simplify" it away from the table correspondence.
#![allow(clippy::identity_op)]

// `g_c2s[2].cart2sph` (`g_trans_cart2sph[10..40]`).
pub const D_C01: f64 = 1.092548430592079070;
pub const D_C10: f64 = 1.092548430592079070;
pub const D_C12: f64 = -0.315391565252520002;
pub const D_C15: f64 = -0.315391565252520002;
pub const D_C17: f64 = 0.630783130505040012;
pub const D_C20: f64 = 1.092548430592079070;
pub const D_C24: f64 = 0.546274215296039535;
pub const D_C27: f64 = -0.546274215296039535;

// `g_c2s[3].cart2sph` (`g_trans_cart2sph[40..110]`).
pub const F_C01: f64 = 1.770130769779930531;
pub const F_C06: f64 = -0.590043589926643510;
pub const F_C14: f64 = 2.890611442640554055;
pub const F_C21: f64 = -0.457045799464465739;
pub const F_C26: f64 = -0.457045799464465739;
pub const F_C28: f64 = 1.828183197857862944;
pub const F_C32: f64 = -1.119528997770346170;
pub const F_C37: f64 = -1.119528997770346170;
pub const F_C39: f64 = 0.746352665180230782;
pub const F_C40: f64 = -0.457045799464465739;
pub const F_C43: f64 = -0.457045799464465739;
pub const F_C45: f64 = 1.828183197857862944;
pub const F_C52: f64 = 1.445305721320277020;
pub const F_C57: f64 = -1.445305721320277020;
pub const F_C60: f64 = 0.590043589926643510;
pub const F_C63: f64 = -1.770130769779930530;

// `g_c2s[4].cart2sph` (`g_trans_cart2sph[110..245]`).
pub const G_C01: f64 = 2.503342941796704538;
pub const G_C06: f64 = -2.503342941796704530;
pub const G_C19: f64 = 5.310392309339791593;
pub const G_C26: f64 = -1.770130769779930530;
pub const G_C31: f64 = -0.946174695757560014;
pub const G_C36: f64 = -0.946174695757560014;
pub const G_C38: f64 = 5.677048174545360108;
pub const G_C49: f64 = -2.007139630671867500;
pub const G_C56: f64 = -2.007139630671867500;
pub const G_C58: f64 = 2.676186174229156671;
pub const G_C60: f64 = 0.317356640745612911;
pub const G_C63: f64 = 0.634713281491225822;
pub const G_C65: f64 = -2.538853125964903290;
pub const G_C70: f64 = 0.317356640745612911;
pub const G_C72: f64 = -2.538853125964903290;
pub const G_C74: f64 = 0.846284375321634430;
pub const G_C77: f64 = -2.007139630671867500;
pub const G_C82: f64 = -2.007139630671867500;
pub const G_C84: f64 = 2.676186174229156671;
pub const G_C90: f64 = -0.473087347878780002;
pub const G_C95: f64 = 2.838524087272680054;
pub const G_C100: f64 = 0.473087347878780009;
pub const G_C102: f64 = -2.838524087272680050;
pub const G_C107: f64 = 1.770130769779930531;
pub const G_C112: f64 = -5.310392309339791590;
pub const G_C120: f64 = 0.625835735449176134;
pub const G_C123: f64 = -3.755014412695056800;
pub const G_C130: f64 = 0.625835735449176134;

/// `d_ket_cart2spheric`: `sph[m*lds+i]` from the 6 Cartesian `d` values
/// (`xx, xy, xz, yy, yz, zz`) at `cart[x*nbra+i]`.
pub fn c2s_ket_d(sph: &mut [f64], cart: &[f64], lds: usize, nbra: usize) {
    for i in 0..nbra {
        sph[i] = D_C01 * cart[1 * nbra + i];
        sph[lds + i] = D_C10 * cart[4 * nbra + i];
        sph[2 * lds + i] =
            D_C12 * cart[i] + D_C15 * cart[3 * nbra + i] + D_C17 * cart[5 * nbra + i];
        sph[3 * lds + i] = D_C20 * cart[2 * nbra + i];
        sph[4 * lds + i] = D_C24 * cart[i] + D_C27 * cart[3 * nbra + i];
    }
}

/// `f_ket_cart2spheric`: `sph[m*lds+i]` from the 10 Cartesian `f` values
/// (`xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz`).
pub fn c2s_ket_f(sph: &mut [f64], cart: &[f64], lds: usize, nbra: usize) {
    for i in 0..nbra {
        sph[i] = F_C01 * cart[1 * nbra + i] + F_C06 * cart[6 * nbra + i];
        sph[lds + i] = F_C14 * cart[4 * nbra + i];
        sph[2 * lds + i] =
            F_C21 * cart[1 * nbra + i] + F_C26 * cart[6 * nbra + i] + F_C28 * cart[8 * nbra + i];
        sph[3 * lds + i] =
            F_C32 * cart[2 * nbra + i] + F_C37 * cart[7 * nbra + i] + F_C39 * cart[9 * nbra + i];
        sph[4 * lds + i] =
            F_C40 * cart[i] + F_C43 * cart[3 * nbra + i] + F_C45 * cart[5 * nbra + i];
        sph[5 * lds + i] = F_C52 * cart[2 * nbra + i] + F_C57 * cart[7 * nbra + i];
        sph[6 * lds + i] = F_C60 * cart[i] + F_C63 * cart[3 * nbra + i];
    }
}

/// `g_ket_cart2spheric`: `sph[m*lds+i]` from the 15 Cartesian `g` values
/// (`xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz, yyyy, yyyz,
/// yyzz, yzzz, zzzz`).
pub fn c2s_ket_g(sph: &mut [f64], cart: &[f64], lds: usize, nbra: usize) {
    for i in 0..nbra {
        sph[i] = G_C01 * cart[1 * nbra + i] + G_C06 * cart[6 * nbra + i];
        sph[lds + i] = G_C19 * cart[4 * nbra + i] + G_C26 * cart[11 * nbra + i];
        sph[2 * lds + i] =
            G_C31 * cart[1 * nbra + i] + G_C36 * cart[6 * nbra + i] + G_C38 * cart[8 * nbra + i];
        sph[3 * lds + i] =
            G_C49 * cart[4 * nbra + i] + G_C56 * cart[11 * nbra + i] + G_C58 * cart[13 * nbra + i];
        sph[4 * lds + i] = G_C60 * cart[i]
            + G_C63 * cart[3 * nbra + i]
            + G_C65 * cart[5 * nbra + i]
            + G_C70 * cart[10 * nbra + i]
            + G_C72 * cart[12 * nbra + i]
            + G_C74 * cart[14 * nbra + i];
        sph[5 * lds + i] =
            G_C77 * cart[2 * nbra + i] + G_C82 * cart[7 * nbra + i] + G_C84 * cart[9 * nbra + i];
        sph[6 * lds + i] = G_C90 * cart[i]
            + G_C95 * cart[5 * nbra + i]
            + G_C100 * cart[10 * nbra + i]
            + G_C102 * cart[12 * nbra + i];
        sph[7 * lds + i] = G_C107 * cart[2 * nbra + i] + G_C112 * cart[7 * nbra + i];
        sph[8 * lds + i] =
            G_C120 * cart[i] + G_C123 * cart[3 * nbra + i] + G_C130 * cart[10 * nbra + i];
    }
}

/// `CINTc2s_ket_sph1` for `l = 2, 3, 4`: `sph` holds `deg*bgrids` values per
/// `(comp, contraction)`, `cart` `dcart*bgrids`; `lds == ldc == bgrids` at the
/// single call site (`PBCeval_sph_iter`).
pub fn c2s_ket_sph1(l: i32, sph: &mut [f64], cart: &[f64], lds: usize, nbra: usize) {
    match l {
        2 => c2s_ket_d(sph, cart, lds, nbra),
        3 => c2s_ket_f(sph, cart, lds, nbra),
        4 => c2s_ket_g(sph, cart, lds, nbra),
        _ => panic!("c2s_ket_sph1: l={l} not ported (only d, f, g); refuse, do not approximate"),
    }
}
