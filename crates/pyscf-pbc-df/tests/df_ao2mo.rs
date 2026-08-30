//! Plan 14-05 acceptance — MO integrals from `cderi`, the `ao2mo_7d` index
//! contract, and the out-of-core 3-index drivers.
//!
//! # Why most of this runs on a SYNTHETIC `cderi`
//!
//! The five algebra tests below are about the *contraction and the packing* —
//! the `s2` unpack, the four `get_eri` branches, the bra-only conjugation in
//! `r_e2`, and above all the `kconserv` wiring in `ao2mo_7d`. None of that
//! depends on where `cderi` came from, and a real GDF build costs minutes
//! (`tests/gdf.rs` measures ~90 s for He-fcc at gamma in release, and diamond
//! is ~36 min because it is one screening group). Driving [`Gdf::with_cderi`]
//! with a deterministic pseudo-random tensor makes those tests exact,
//! millisecond-fast, and — crucially — able to use a system with `nao = 4` and
//! four DIFFERENT `nmo`s, which is what gives the index-order contract teeth.
//! He-fcc/`sto-3g` has `nao = 1`, so a real-build contract test could not tell
//! `[i,j,k,l]` from `[l,k,j,i]`.
//!
//! The synthetic tensor is not arbitrary: it obeys
//! `C[kj,ki][L, q p] = conj(C[ki,kj][L, p q])`, the Hermitian relation a real
//! 3-centre integral over a real auxiliary function satisfies. That is what
//! makes `get_eri`'s branch 3 (one block, `zdotNC`, transposed ket) and branch
//! 4 (two blocks, `zdotNN`) mutually checkable — see
//! `get_eri_branches_agree_where_they_overlap`.
//!
//! The end-to-end tests against a real GDF and against FFTDF/AFTDF follow.

mod common;

use std::collections::HashMap;

use pyscf_algebra::CTensor;
use pyscf_pbc_df::df_ao2mo::{Eri, MoCoeff, ao2mo_7d, general, get_eri};
use pyscf_pbc_df::gdf_builder::j3c::{Cderi, CderiBlock};
use pyscf_pbc_df::incore::Aosym;
use pyscf_pbc_df::pbc_ao2mo::{aft_ao2mo_7d, aft_general, fft_ao2mo_7d, fft_general};
use pyscf_pbc_df::{Aftdf, Fftdf, Gdf};
use pyscf_pbc_gto::Cell;

// ---------------------------------------------------------------------------
// deterministic pseudo-random data
// ---------------------------------------------------------------------------

/// A 64-bit LCG. Deterministic across platforms and runs — a test whose data
/// depends on the system RNG cannot be debugged from its failure message.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1))
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // 53 significant bits, mapped to [-1, 1).
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn kpts_of(cell: &Cell, mesh: [usize; 3]) -> Vec<[f64; 3]> {
    cell.make_kpts(mesh).expect("make_kpts")
}

/// A `GDF` carrying a synthetic `s1` tensor over `cell`'s own `nao`.
fn synth_gdf(cell: Cell, kpts: &[[f64; 3]], naux: usize, seed: u64) -> Gdf {
    let nao = cell.mol.nao_nr;
    Gdf::with_cderi(cell, synth_cderi(kpts, nao, naux, seed))
}

/// A synthetic `s1` `cderi` over `kpts`, obeying
/// `C[kj,ki][L, q p] = conj(C[ki,kj][L, p q])`.
fn synth_cderi(kpts: &[[f64; 3]], nao: usize, naux: usize, seed: u64) -> Cderi {
    let nk = kpts.len();
    let n2 = nao * nao;
    let mut rng = Rng::new(seed);
    let mut blocks: HashMap<usize, CderiBlock> = HashMap::new();
    for (ki, kpt_i) in kpts.iter().enumerate() {
        for kj in ki..nk {
            let mut d = CTensor::zeros(naux * n2);
            for l in 0..naux {
                for p in 0..nao {
                    for q in 0..nao {
                        d.re[l * n2 + p * nao + q] = rng.next_f64();
                        // A diagonal k-pair at gamma must be REAL for the
                        // gamma branch to mean anything; every other pair is
                        // genuinely complex.
                        d.im[l * n2 + p * nao + q] =
                            if ki == kj && kpt_i.iter().all(|v| v.abs() < 1e-12) {
                                0.0
                            } else {
                                rng.next_f64()
                            };
                    }
                }
            }
            // The (ki, ki) block of a Hermitian tensor is itself Hermitian in
            // (p, q); impose that so the diagonal pairs are self-consistent.
            if ki == kj {
                let src = d.clone();
                for l in 0..naux {
                    for p in 0..nao {
                        for q in 0..nao {
                            let (a, b) = (l * n2 + p * nao + q, l * n2 + q * nao + p);
                            d.re[a] = 0.5 * (src.re[a] + src.re[b]);
                            d.im[a] = 0.5 * (src.im[a] - src.im[b]);
                        }
                    }
                }
            }
            blocks.insert(
                ki * nk + kj,
                CderiBlock {
                    data: d.clone(),
                    rank: naux,
                    nao_pair: n2,
                    negative: None,
                },
            );
            if ki != kj {
                let mut t = CTensor::zeros(naux * n2);
                for l in 0..naux {
                    for p in 0..nao {
                        for q in 0..nao {
                            t.re[l * n2 + q * nao + p] = d.re[l * n2 + p * nao + q];
                            t.im[l * n2 + q * nao + p] = -d.im[l * n2 + p * nao + q];
                        }
                    }
                }
                blocks.insert(
                    kj * nk + ki,
                    CderiBlock {
                        data: t,
                        rank: naux,
                        nao_pair: n2,
                        negative: None,
                    },
                );
            }
        }
    }
    Cderi {
        blocks,
        kpts: kpts.to_vec(),
        aosym: Aosym::S1,
    }
}

/// The same tensor packed `s2` — only reachable at gamma, where it is real.
fn synth_cderi_s2(nao: usize, naux: usize, seed: u64) -> Cderi {
    let n2 = nao * nao;
    let np = nao * (nao + 1) / 2;
    let s1 = synth_cderi(&[[0.0; 3]], nao, naux, seed);
    let b = &s1.blocks[&0];
    let mut d = CTensor::zeros(naux * np);
    for l in 0..naux {
        for p in 0..nao {
            for q in 0..=p {
                d.re[l * np + p * (p + 1) / 2 + q] = b.data.re[l * n2 + p * nao + q];
                d.im[l * np + p * (p + 1) / 2 + q] = b.data.im[l * n2 + p * nao + q];
            }
        }
    }
    let mut blocks = HashMap::new();
    blocks.insert(
        0,
        CderiBlock {
            data: d,
            rank: naux,
            nao_pair: np,
            negative: None,
        },
    );
    Cderi {
        blocks,
        kpts: vec![[0.0; 3]],
        aosym: Aosym::S2,
    }
}

fn rand_mo(nao: usize, nmo: usize, rng: &mut Rng, complex: bool) -> MoCoeff {
    let mut c = CTensor::zeros(nao * nmo);
    for i in 0..nao * nmo {
        c.re[i] = rng.next_f64();
        c.im[i] = if complex { rng.next_f64() } else { 0.0 };
    }
    MoCoeff::new(nao, nmo, c)
}

fn dev(a: &Eri, b: &Eri) -> f64 {
    assert_eq!(a.data.re.len(), b.data.re.len(), "Eri shapes differ");
    a.data
        .re
        .iter()
        .zip(&b.data.re)
        .chain(a.data.im.iter().zip(&b.data.im))
        .fold(0.0f64, |w, (x, y)| w.max((x - y).abs()))
}

// ---------------------------------------------------------------------------
// Test 1 — 8-fold symmetry at gamma
// ---------------------------------------------------------------------------

/// **Task 5.1** — `(pq|rs) = (qp|rs) = (pq|sr) = (rs|pq)` at gamma, and the
/// block is real. No oracle.
///
/// Phase 13 used exactly this residue as a screening probe for `ft_aopair`;
/// here it tests the `cderi` contraction and the `s2 -> s1` unpack, which is
/// where the symmetry actually comes from.
#[test]
fn gamma_eri_from_cderi_has_eightfold_symmetry() {
    let cell = common::diamond();
    let nao = cell.mol.nao_nr;
    let df = Gdf::with_cderi(cell, synth_cderi_s2(nao, 6, 11));
    let eri = get_eri(&df, [0; 4], false).expect("get_eri");

    let imax = eri.data.im.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    assert!(
        imax == 0.0,
        "a gamma ERI must be exactly real, |Im| = {imax:e}"
    );

    let mut worst = 0.0f64;
    for p in 0..nao {
        for q in 0..nao {
            for r in 0..nao {
                for s in 0..nao {
                    let v = eri.get(p, q, r, s).0;
                    worst = worst.max((v - eri.get(q, p, r, s).0).abs());
                    worst = worst.max((v - eri.get(p, q, s, r).0).abs());
                    worst = worst.max((v - eri.get(r, s, p, q).0).abs());
                    worst = worst.max((v - eri.get(s, r, q, p).0).abs());
                }
            }
        }
    }
    assert!(worst < 1e-12, "8-fold ERI symmetry broken by {worst:e}");
}

// ---------------------------------------------------------------------------
// Test 2 — compact <-> s1
// ---------------------------------------------------------------------------

/// **Task 5.2** — `compact` and `s1` are the same numbers, bit-identically,
/// after unpacking. No oracle.
#[test]
fn compact_and_s1_are_bit_identical_after_unpacking() {
    let cell = common::diamond();
    let nao = cell.mol.nao_nr;
    let df = Gdf::with_cderi(cell, synth_cderi_s2(nao, 6, 12));
    let packed = get_eri(&df, [0; 4], true).expect("compact");
    let plain = get_eri(&df, [0; 4], false).expect("s1");

    assert!(packed.row.packed && packed.col.packed, "compact must pack");
    assert_eq!(
        packed.row.len(),
        nao * (nao + 1) / 2,
        "s2 packs to nao(nao+1)/2"
    );
    assert!(!plain.row.packed && !plain.col.packed);

    let unpacked = packed.restore_s1();
    let d = dev(&unpacked, &plain);
    assert!(
        d == 0.0,
        "compact/s1 must be BIT-identical, differ by {d:e}"
    );
}

// ---------------------------------------------------------------------------
// Test 3 — general with identity coefficients is get_eri
// ---------------------------------------------------------------------------

/// **Task 5.3** — `general` with identity MO coefficients reproduces
/// `get_eri`, on all four of `get_eri`'s branches. No oracle.
///
/// The four quadruples below are chosen to hit them: `[0,0,0,0]` is gamma,
/// `[1,2,1,2]` is `k0 == k2 && k1 == k3`, `[1,2,2,1]` is `k0 == k3 && k1 == k2`
/// and `[1,2,3,kl]` falls through to the two-block branch. A transposition in
/// any one branch shows up here and nowhere else.
#[test]
fn general_with_identity_mos_reproduces_get_eri() {
    let cell = common::diamond();
    let nao = cell.mol.nao_nr;
    let kpts = kpts_of(&cell, [2, 2, 2]);
    let kc = pyscf_pbc_lib::kpts_helper::get_kconserv(&cell.a, &kpts);
    let df = synth_gdf(cell, &kpts, 5, 13);
    let id = MoCoeff::identity(nao);
    let mos = [&id, &id, &id, &id];

    let quads = [
        [0, 0, 0, 0],
        [1, 2, 1, 2],
        [1, 2, 2, 1],
        [1, 2, 3, kc.get(1, 2, 3) as usize],
    ];
    for q in quads {
        let a = get_eri(&df, q, false).expect("get_eri").restore_s1();
        let b = general(&df, mos, q, false).expect("general").restore_s1();
        let d = dev(&a, &b);
        assert!(d < 1e-13, "general(identity) != get_eri at {q:?}: {d:e}");
    }
}

/// The two `get_eri` branches that can be reached for the SAME quadruple must
/// agree. No oracle, and it is the sharpest structural check in the file:
/// branch 3 uses one `cderi` block through `zdotNC` plus a ket transpose,
/// branch 4 uses two blocks through `zdotNN`, and they coincide only if the
/// transpose, the conjugation and the `(kj, ki)` block relation are all right.
#[test]
fn get_eri_branches_agree_where_they_overlap() {
    let cell = common::diamond();
    let nao = cell.mol.nao_nr;
    let naux = 5;
    let kpts = kpts_of(&cell, [2, 2, 2]);
    let cderi = synth_cderi(&kpts, nao, naux, 14);
    let df = Gdf::with_cderi(cell.clone(), cderi.clone());

    // `[1, 2, 2, 1]` takes branch 3. Recompute it through branch 4's algebra by
    // hand: eri[pq, rs] = SUM_L C[1,2][L, pq] · C[2,1][L, rs].
    let got = get_eri(&df, [1, 2, 2, 1], false).expect("branch 3");
    let a = &cderi.get(1, 2).expect("block").data;
    let b = &cderi.get(2, 1).expect("block").data;
    let n2 = nao * nao;
    let mut want = CTensor::zeros(n2 * n2);
    for l in 0..naux {
        for r in 0..n2 {
            let (ar, ai) = (a.re[l * n2 + r], a.im[l * n2 + r]);
            for c in 0..n2 {
                let (br, bi) = (b.re[l * n2 + c], b.im[l * n2 + c]);
                want.re[r * n2 + c] += ar * br - ai * bi;
                want.im[r * n2 + c] += ar * bi + ai * br;
            }
        }
    }
    let d = got
        .data
        .re
        .iter()
        .zip(&want.re)
        .chain(got.data.im.iter().zip(&want.im))
        .fold(0.0f64, |w, (x, y)| w.max((x - y).abs()));
    assert!(d < 1e-13, "get_eri branch 3 != the two-block form: {d:e}");
}

// ---------------------------------------------------------------------------
// Test 4 — the ao2mo_7d index contract
// ---------------------------------------------------------------------------

/// **Task 5.4 — THE CONTRACT.** No oracle.
///
/// `eri7d[ki, kj, kk][i, j, k, l] == general(mo0[ki], mo1[kj], mo2[kk], mo3[kl])`
/// with `kl = kconserv[ki, kj, kk]`, for every `(ki, kj, kk)` on a 2x2x2 mesh,
/// with FOUR DIFFERENT `nmo`s so no index permutation can pass by accident.
///
/// The `kconserv` table comes from `pyscf_pbc_lib::kpts_helper::get_kconserv`,
/// never a hand-written one — plan 14-05 Task 5.4's requirement, and the reason
/// is that a hand-written table would encode the very convention under test.
///
/// **This is the fact Phase 15 is blocked on.** Plan 13-06 declined to guess it.
#[test]
fn ao2mo_7d_index_order_is_the_phase_15_contract() {
    let cell = common::diamond();
    let nao = cell.mol.nao_nr;
    let kpts = kpts_of(&cell, [2, 2, 2]);
    let nk = kpts.len();
    let kc = pyscf_pbc_lib::kpts_helper::get_kconserv(&cell.a, &kpts);
    let df = synth_gdf(cell, &kpts, 4, 15);

    let mut rng = Rng::new(99);
    let nmo = [2usize, 3, 1, 4];
    let sets: Vec<Vec<MoCoeff>> = nmo
        .iter()
        .map(|&m| (0..nk).map(|_| rand_mo(nao, m, &mut rng, true)).collect())
        .collect();
    let mos = [
        sets[0].as_slice(),
        sets[1].as_slice(),
        sets[2].as_slice(),
        sets[3].as_slice(),
    ];

    let seven = ao2mo_7d(&df, mos, 1.0).expect("ao2mo_7d");
    assert_eq!(seven.nkpts, nk);
    assert_eq!(
        seven.nmo, nmo,
        "shape is (nk, nk, nk, nmoi, nmoj, nmok, nmol)"
    );
    assert_eq!(
        seven.data.re.len(),
        nk * nk * nk * nmo.iter().product::<usize>(),
        "ao2mo_7d must be dense over three free k-axes"
    );

    let mut worst = 0.0f64;
    for ki in 0..nk {
        for kj in 0..nk {
            for kk in 0..nk {
                let kl = kc.get(ki, kj, kk) as usize;
                let g = general(
                    &df,
                    [&sets[0][ki], &sets[1][kj], &sets[2][kk], &sets[3][kl]],
                    [ki, kj, kk, kl],
                    false,
                )
                .expect("general");
                for i in 0..nmo[0] {
                    for j in 0..nmo[1] {
                        for k in 0..nmo[2] {
                            for l in 0..nmo[3] {
                                let (gr, gi) = g.get(i, j, k, l);
                                let (sr, si) = seven.get(ki, kj, kk, [i, j, k, l]);
                                worst = worst.max((gr - sr).abs()).max((gi - si).abs());
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        worst < 1e-13,
        "ao2mo_7d disagrees with general by {worst:e} — the index order or the \
         kconserv wiring is wrong"
    );
}

/// The `factor` argument scales the whole tensor linearly, as upstream's does.
#[test]
fn ao2mo_7d_factor_scales_linearly() {
    let cell = common::diamond();
    let nao = cell.mol.nao_nr;
    let kpts = kpts_of(&cell, [2, 1, 1]);
    let nk = kpts.len();
    let df = synth_gdf(cell, &kpts, 3, 16);
    let mut rng = Rng::new(5);
    let set: Vec<MoCoeff> = (0..nk).map(|_| rand_mo(nao, 2, &mut rng, true)).collect();
    let m = [set.as_slice(); 4];
    let a = ao2mo_7d(&df, m, 1.0).expect("f=1");
    let b = ao2mo_7d(&df, m, -2.5).expect("f=-2.5");
    let w = a
        .data
        .re
        .iter()
        .zip(&b.data.re)
        .fold(0.0f64, |w, (x, y)| w.max((x * -2.5 - y).abs()));
    assert!(w < 1e-13, "factor is not linear: {w:e}");
}

// ---------------------------------------------------------------------------
// Test 5 — a REAL GDF end to end
// ---------------------------------------------------------------------------

/// The whole path on a real `cderi`: build He-fcc at gamma, contract, and check
/// the block against the definition `SUM_L C[L,pq] C[L,rs]` computed straight
/// off `sr_loop`. No oracle — it pins `get_eri` to the store rather than to a
/// synthetic stand-in.
#[test]
fn get_eri_on_a_real_gdf_matches_the_sr_loop_definition() {
    let cell = common::he_all_electron();
    let mut df = Gdf::new(cell, &[[0.0; 3]]);
    df.build().expect("gdf build");
    let nao = df.cell.mol.nao_nr;

    let eri = get_eri(&df, [0; 4], false).expect("get_eri");
    let blocks = df.sr_loop(0, 0, false).expect("sr_loop");
    let n2 = nao * nao;
    let mut want = vec![0.0f64; n2 * n2];
    for b in &blocks {
        let s = f64::from(b.sign);
        for l in 0..b.naux {
            for r in 0..n2 {
                for c in 0..n2 {
                    want[r * n2 + c] += s * b.re[l * n2 + r] * b.re[l * n2 + c];
                }
            }
        }
    }
    let d = eri
        .data
        .re
        .iter()
        .zip(&want)
        .fold(0.0f64, |w, (x, y)| w.max((x - y).abs()));
    assert!(d < 1e-13, "get_eri != the sr_loop definition: {d:e}");

    // And `general` with identity coefficients is the same block.
    let id = MoCoeff::identity(nao);
    let g = general(&df, [&id; 4], [0; 4], false).expect("general");
    let dd = dev(&eri, &g);
    assert!(
        dd < 1e-13,
        "general(identity) != get_eri on a real GDF: {dd:e}"
    );
}

// ---------------------------------------------------------------------------
// Test 6 — cross-builder
// ---------------------------------------------------------------------------

/// **Task 5.5** — `ao2mo_7d(FFTDF)` vs `ao2mo_7d(AFTDF)`, gated the way Phase
/// 13 gated the same pair: as a MESH LADDER, not a fixed number.
///
/// The two builders evaluate the SAME exact integral by different means, so
/// their difference is `ft_aopair` against the FFT of the real-space AO
/// product — the quantity Phase 13's Gate 2 measured — and it falls with the
/// mesh. A fixed tolerance here would be a tolerance on FFTDF's aliasing, not
/// on the transform, which is why `tests/pbc_ao2mo.rs::aft_and_fft_eri_converge`
/// states its 2-index sibling the same way.
///
/// Diamond, not He-fcc: `gth-szv` is smooth, and the all-electron He `sto-3g`
/// contraction needs a far finer mesh before FFTDF resolves it at all (measured
/// 1.011e-01 at mesh 15 — which is FFTDF's aliasing, and is what a naive gate
/// would have mis-attributed to this plan's code).
#[test]
fn ao2mo_7d_converges_between_fftdf_and_aftdf() {
    let cell = common::diamond();
    let kpts = kpts_of(&cell, [2, 1, 1]);
    let nk = kpts.len();
    let nao = cell.mol.nao_nr;

    let mut rng = Rng::new(77);
    let set: Vec<MoCoeff> = (0..nk).map(|_| rand_mo(nao, 2, &mut rng, false)).collect();
    let m = [set.as_slice(); 4];

    let mut devs = Vec::new();
    for mm in [11usize, 15, 21] {
        let mesh = [mm, mm, mm];
        let fft = Fftdf::with_mesh(cell.clone(), &kpts, mesh).expect("fftdf");
        let aft = Aftdf::with_mesh(cell.clone(), &kpts, mesh).expect("aftdf");
        let a = fft_ao2mo_7d(&fft, m, 1.0).expect("fft 7d");
        let b = aft_ao2mo_7d(&aft, m, 1.0).expect("aft 7d");
        let d = a
            .data
            .re
            .iter()
            .zip(&b.data.re)
            .chain(a.data.im.iter().zip(&b.data.im))
            .fold(0.0f64, |w, (x, y)| w.max((x - y).abs()));
        eprintln!("ao2mo_7d mesh {mm}: |FFTDF - AFTDF| = {d:e}");
        devs.push(d);
    }
    assert!(devs[1] < devs[0], "mesh 15 must improve on mesh 11");
    assert!(devs[2] < devs[1], "mesh 21 must improve on mesh 15");
}

/// `general` on the plane-wave builders is `get_eri` transformed — the check
/// that closes 13-06's `general` half.
#[test]
fn plane_wave_general_with_identity_mos_reproduces_get_eri() {
    let cell = common::diamond();
    let nao = cell.mol.nao_nr;
    let mesh = [11, 11, 11];
    let g = [[0.0; 3]; 4];
    let id = MoCoeff::identity(nao);

    let fft = Fftdf::with_mesh(cell.clone(), &[[0.0; 3]], mesh).expect("fftdf");
    let aft = Aftdf::with_mesh(cell, &[[0.0; 3]], mesh).expect("aftdf");

    let fa = pyscf_pbc_df::pbc_ao2mo::fft_get_eri(&fft, g).expect("fft eri");
    let fb = fft_general(&fft, [&id; 4], g).expect("fft general");
    let aa = pyscf_pbc_df::pbc_ao2mo::aft_get_eri(&aft, g).expect("aft eri");
    let ab = aft_general(&aft, [&id; 4], g).expect("aft general");

    for (name, ao, mo) in [("FFTDF", &fa, &fb), ("AFTDF", &aa, &ab)] {
        let d = ao
            .re
            .iter()
            .zip(&mo.data.re)
            .chain(ao.im.iter().zip(&mo.data.im))
            .fold(0.0f64, |w, (x, y)| w.max((x - y).abs()));
        assert!(d < 1e-13, "{name}: general(identity) != get_eri by {d:e}");
    }
}

// ---------------------------------------------------------------------------
// outcore
// ---------------------------------------------------------------------------

/// `balance_segs` against upstream's own algorithm, hand-evaluated.
///
/// `pyscf.ao2mo.outcore.balance_segs([3,3,3,3], 7)` is
/// `[(0, 2, 6), (2, 4, 6)]`: the greedy partition extends while the cumulative
/// width stays `<= blksize`. The degenerate cases matter more than the happy
/// one — a `blksize` smaller than a single segment must still emit every
/// segment rather than looping forever.
#[test]
fn balance_segs_matches_upstream() {
    use pyscf_pbc_df::balance_segs;
    assert_eq!(
        balance_segs(&[3, 3, 3, 3], 7, 0, None),
        vec![(0, 2, 6), (2, 4, 6)]
    );
    assert_eq!(balance_segs(&[3, 3, 3, 3], 12, 0, None), vec![(0, 4, 12)]);
    assert_eq!(
        balance_segs(&[3, 3, 3, 3], 1, 0, None),
        vec![(0, 1, 3), (1, 2, 3), (2, 3, 3), (3, 4, 3)]
    );
    assert_eq!(balance_segs(&[5], 2, 0, None), vec![(0, 1, 5)]);
    assert!(balance_segs(&[], 4, 0, None).is_empty());
}

/// The out-of-core drivers write what `incore::aux_e2` computes — same numbers,
/// one k-pair at a time, in both orientations. This is the whole contract: the
/// blocking is a memory strategy and must not change a digit.
#[test]
fn outcore_drivers_reproduce_the_incore_tensor() {
    use pyscf_pbc_df::incore::{KptPair, aux_e2, make_modrho_basis};
    use pyscf_pbc_df::outcore::{Blocking, Orientation, aux_e1, aux_e2 as aux_e2_out};

    let cell = common::he_all_electron();
    let aux = make_modrho_basis(&cell, None, None).expect("auxcell");
    let kpts = kpts_of(&cell, [2, 1, 1]);
    let pairs: Vec<KptPair> = kpts.iter().map(|k| KptPair { ki: *k, kj: *k }).collect();

    let want = aux_e2(&cell, &aux, Aosym::S2, &pairs, None, None).expect("incore aux_e2");
    let nao_pair = Aosym::S2.nao_pair(cell.mol.nao_nr);
    let naux = aux.naux();

    let dir = std::env::temp_dir().join(format!("pbc_outcore_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");

    // Force one k-pair per block so the blocking is genuinely exercised.
    let blocking = Blocking {
        max_memory: 2000.0,
        blksize: Some(1),
    };
    let f2 = aux_e2_out(
        &cell,
        &aux,
        dir.join("e2.h5"),
        "int3c2e",
        Aosym::S2,
        &pairs,
        "j3c",
        blocking,
        None,
        None,
    )
    .expect("aux_e2 outcore");
    assert_eq!(f2.nkptij(), pairs.len());
    assert_eq!(f2.orientation(), Orientation::PairLeading);
    for (k, w) in want.iter().enumerate() {
        let got = f2.read(k).expect("read");
        let d = got
            .re
            .iter()
            .zip(&w.re)
            .chain(got.im.iter().zip(&w.im))
            .fold(0.0f64, |x, (a, b)| x.max((a - b).abs()));
        assert!(d == 0.0, "aux_e2 outcore pair {k} differs by {d:e}");
    }

    let f1 = aux_e1(
        &cell,
        &aux,
        dir.join("e1.h5"),
        "int3c2e",
        Aosym::S2,
        &pairs,
        "eri_mo",
        blocking,
        None,
        None,
    )
    .expect("aux_e1 outcore");
    assert_eq!(f1.orientation(), Orientation::AuxLeading);
    for (k, w) in want.iter().enumerate() {
        let got = f1.read(k).expect("read");
        let mut d = 0.0f64;
        for p in 0..nao_pair {
            for l in 0..naux {
                d = d.max((got.re[l * nao_pair + p] - w.re[p * naux + l]).abs());
                d = d.max((got.im[l * nao_pair + p] - w.im[p * naux + l]).abs());
            }
        }
        assert!(
            d == 0.0,
            "aux_e1 outcore pair {k} is not the transpose: {d:e}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Task 5.6 — the oracle
// ---------------------------------------------------------------------------

/// One script, two entry points. `what` selects `get_eri` or `ao2mo_7d`.
///
/// **Three upstream knobs are set, and each is a measured finding, not a
/// convenience.**
///
/// 1. `exclude_dd_block = False` — D-PBC-23. On He-fcc the two routes are
///    bit-identical (measured at exactly 0), so the substitution is inert
///    there; on any cell with a smooth shell it is what makes the comparison
///    an assertion rather than a price tag.
/// 2. `direct_scf_tol` — 14-02 measured that upstream's default derives
///    `cell.precision / lattice_sum_factor**2 * .1` = **1.46e-11** here, FOUR
///    orders looser than this port's 1e-14 Gaussian-product prescreen, and
///    that the difference discards a P-independent term worth **1.98e-9** in
///    `fuse(j3c)`. That propagates straight through `cderi` into the ERI. The
///    script therefore takes the tolerance as an argument: the 1e-11 gate is
///    stated with the screens EQUALISED, and the default-screen deviation is
///    recorded beside it as an upper bound. Same device as
///    `tests/gdf_builder.rs::he_fuse_j3c_matches_upstream`.
///
/// 3. `estimate_rcut` is made UNIFORM at its own maximum. `_CCGDFBuilder.build`
///    (`gdf_builder.py:116-121`) takes a PER-SHELL-PAIR radius array and hands
///    it to `ft_ao.ExtendedMole.strip_basis(rcut)`, which then drops images
///    pair by pair. This port has no `ExtendedMole` (D-PBC-21, extended by
///    D-PBC-23) and evaluates every shell pair out to the maximum radius, so it
///    keeps images upstream discards — **the port is the more converged of the
///    two, not the less.** Measured on He-fcc at gamma, on the fused `j3c`
///    that feeds `solve_cderi`:
///
///    | upstream screening | max\|d j3c\| vs this port |
///    |---|---|
///    | `strip_basis`, per-pair (upstream default) | **1.054e-09** |
///    | uniform at `rcut.max()` (what this port does) | **7.333e-13** |
///
///    Through `solve_cderi` that becomes 6.7e-9 in `cderi` and **2.4e-9** in
///    the ERI, which is exactly the residual this gate would otherwise be
///    asserting against. 14-02's own `fuse(j3c)` gate could not see it: it
///    compared against a standalone `incore.Int3cBuilder`, which strips
///    nothing.
///
/// Carrying all three substitutions in the suite is what Phase 13 did for the
/// `get_pp` / `_IntPPBuilder` attribution, so none of them can rot.
const ORACLE: &str = r#"
import json, sys
import numpy
import pyscf
assert pyscf.__version__ == '2.12.1', pyscf.__version__
from pyscf.pbc import gto as pgto
from pyscf.pbc import df as pdf
from pyscf.pbc.df import df_ao2mo
import pyscf.pbc.df.gdf_builder as gb

a, xyz, sym, basis, pseudo, kmesh, what, payload, tol = (
    json.loads(sys.argv[1]), json.loads(sys.argv[2]), json.loads(sys.argv[3]),
    sys.argv[4], sys.argv[5], json.loads(sys.argv[6]), sys.argv[7],
    json.loads(sys.argv[8]), sys.argv[9])

cell = pgto.Cell()
cell.a = numpy.array(a)
cell.atom = [(s, tuple(c)) for s, c in zip(sym, xyz)]
cell.basis = basis
if pseudo != 'none':
    cell.pseudo = pseudo
cell.unit = 'Bohr'
cell.verbose = 0
cell.build()
kpts = cell.make_kpts(kmesh)
nao = cell.nao_nr()
nk = len(kpts)

_init = gb._CCGDFBuilder.__init__
def _patched_init(self, *args, **kwargs):
    _init(self, *args, **kwargs)
    self.exclude_dd_block = False
gb._CCGDFBuilder.__init__ = _patched_init

# `_CCGDFBuilder.build` OVERWRITES direct_scf_tol unconditionally
# (gdf_builder.py:107-111), so the screen has to be reset AFTER it, not in
# __init__ — the kernel reads it later, at gen_int3c_kernel time
# (incore.py:241-255).
_build = gb._CCGDFBuilder.build
def _patched_build(self, *args, **kwargs):
    out = _build(self, *args, **kwargs)
    if tol != 'default':
        self.direct_scf_tol = float(tol)
    return out
gb._CCGDFBuilder.build = _patched_build

# ExtendedMole.strip_basis, defeated: a per-shell-pair radius array flattened to
# its own maximum keeps every image the port keeps. See the Rust-side comment.
if tol != 'default':
    _erc = gb.estimate_rcut
    def _uniform_rcut(*args, **kwargs):
        r = numpy.asarray(_erc(*args, **kwargs), dtype=float)
        return numpy.full_like(r, r.max())
    gb.estimate_rcut = _uniform_rcut

mydf = pdf.GDF(cell, kpts)
mydf._prefer_ccdf = True
mydf.build()

if what == 'get_eri':
    parts = []
    for q in payload:
        e = numpy.asarray(df_ao2mo.get_eri(mydf, kpts[q], compact=False))
        parts.append(e.reshape(-1))
    e = numpy.concatenate(parts)
elif what == 'ao2mo_7d':
    mo = numpy.array(payload, dtype=float).reshape(nk, nao, -1, 2)
    mo = mo[..., 0] + 1j * mo[..., 1]
    e = numpy.asarray(df_ao2mo.ao2mo_7d(mydf, mo, kpts)).reshape(-1)
else:
    raise SystemExit('unknown request ' + what)

e = numpy.asarray(e, dtype=complex)
print(json.dumps({'re': e.real.tolist(), 'im': e.imag.tolist()}))
"#;

fn oracle_args(
    cell: &Cell,
    basis: &str,
    pseudo: &str,
    kmesh: [usize; 3],
    what: &str,
    payload: String,
    tol: &str,
) -> Vec<String> {
    let a: Vec<Vec<f64>> = cell.a.iter().map(|r| r.to_vec()).collect();
    let xyz: Vec<Vec<f64>> = cell.mol.atom_coords().iter().map(|r| r.to_vec()).collect();
    let sym: Vec<String> = cell.mol._atom.iter().map(|(s, _)| s.clone()).collect();
    vec![
        serde_json::to_string(&a).expect("json"),
        serde_json::to_string(&xyz).expect("json"),
        serde_json::to_string(&sym).expect("json"),
        basis.to_string(),
        pseudo.to_string(),
        serde_json::to_string(&kmesh.to_vec()).expect("json"),
        what.to_string(),
        payload,
        tol.to_string(),
    ]
}

fn pull(v: &serde_json::Value, key: &str) -> Vec<f64> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("oracle payload has no {key}"))
        .iter()
        .map(|x| x.as_f64().expect("f64"))
        .collect()
}

fn worst_vs(re: &[f64], im: &[f64], got: &CTensor) -> f64 {
    assert_eq!(re.len(), got.re.len(), "shape mismatch vs upstream");
    let mut w = 0.0f64;
    for i in 0..re.len() {
        w = w.max((re[i] - got.re[i]).abs());
        w = w.max((im[i] - got.im[i]).abs());
    }
    w
}

/// Interleaved `[re, im, re, im, …]` per k-point, which the oracle reshapes to
/// a complex `(nk, nao, nmo)` array.
fn mo_payload(set: &[MoCoeff]) -> String {
    let v: Vec<Vec<f64>> = set
        .iter()
        .map(|m| {
            m.c.re
                .iter()
                .zip(&m.c.im)
                .flat_map(|(r, i)| [*r, *i])
                .collect()
        })
        .collect();
    serde_json::to_string(&v).expect("json")
}

/// **Task 5.6, Gate 1 — `get_eri` vs upstream on the ALL-ELECTRON control.**
///
/// He-fcc is where `exclude_dd_block` is provably inert (D-PBC-23 measured it
/// at exactly **0**), so this is a 1e-11 gate on the algebra with no deferral
/// in the way. Three quadruples, one per non-trivial `get_eri` branch.
#[test]
#[ignore = "oracle: needs PYSCF_ORACLE_VENV and a real GDF build"]
fn get_eri_matches_upstream_on_he_fcc() {
    let Some(py) = common::oracle_python() else {
        eprintln!("{} unset — skipping", common::GATE);
        return;
    };
    let cell = common::he_all_electron();
    let kmesh = [2, 1, 1];
    let kpts = kpts_of(&cell, kmesh);
    let mut df = Gdf::new(cell.clone(), &kpts);
    df.build().expect("gdf build");

    let kc = pyscf_pbc_lib::kpts_helper::get_kconserv(&cell.a, &kpts);
    let quads: Vec<Vec<usize>> = vec![
        vec![0, 0, 0, 0],
        vec![0, 1, 1, 0],
        vec![0, 1, 0, kc.get(0, 1, 0) as usize],
    ];

    let mut got = CTensor::default();
    for q in &quads {
        let e = get_eri(&df, [q[0], q[1], q[2], q[3]], false)
            .expect("get_eri")
            .restore_s1();
        got.re.extend_from_slice(&e.data.re);
        got.im.extend_from_slice(&e.data.im);
    }

    let payload = serde_json::to_string(&quads).expect("json");

    // The gate: screens equalised at 1e-14.
    let args = oracle_args(
        &cell,
        "sto-3g",
        "none",
        kmesh,
        "get_eri",
        payload.clone(),
        "1e-14",
    );
    let v = common::run_python(&py, ORACLE, &args);
    let w = worst_vs(&pull(&v, "re"), &pull(&v, "im"), &got);
    println!("df_ao2mo.get_eri vs upstream (He-fcc {kmesh:?}, screens equalised): {w:e}");
    assert!(
        w < 1e-11,
        "get_eri vs upstream with both prescreens at 1e-14 = {w:e}. This is the gate \
         on the contraction and it has no escape hatch"
    );

    // And upstream's DEFAULT screening, recorded as an upper bound rather than
    // a target: `strip_basis`'s per-shell-pair radii plus the looser
    // `direct_scf_tol`. Localised above to `strip_basis` — 1.054e-9 in the
    // fused `j3c`, 6.7e-9 in `cderi`, 2.75e-9 here.
    let args = oracle_args(
        &cell, "sto-3g", "none", kmesh, "get_eri", payload, "default",
    );
    let v = common::run_python(&py, ORACLE, &args);
    let d = worst_vs(&pull(&v, "re"), &pull(&v, "im"), &got);
    println!("df_ao2mo.get_eri vs upstream (He-fcc {kmesh:?}, upstream default screen): {d:e}");
    assert!(
        d < 1e-8,
        "the default-screen deviation grew to {d:e}; it was 2.750e-9 when 14-05 \
         shipped, and it is upstream's `direct_scf_tol`, not the port's algebra"
    );
}

/// **Task 5.6, Gate 1 — `ao2mo_7d` vs upstream, which is what pins the index
/// order to something outside this repository.**
///
/// Complex MO coefficients, so a missing conjugate cannot cancel.
#[test]
#[ignore = "oracle: needs PYSCF_ORACLE_VENV and a real GDF build"]
fn ao2mo_7d_matches_upstream_on_he_fcc() {
    let Some(py) = common::oracle_python() else {
        eprintln!("{} unset — skipping", common::GATE);
        return;
    };
    let cell = common::he_all_electron();
    let nao = cell.mol.nao_nr;
    let kmesh = [2, 1, 1];
    let kpts = kpts_of(&cell, kmesh);
    let nk = kpts.len();
    let mut df = Gdf::new(cell.clone(), &kpts);
    df.build().expect("gdf build");

    let mut rng = Rng::new(2024);
    let set: Vec<MoCoeff> = (0..nk).map(|_| rand_mo(nao, nao, &mut rng, true)).collect();
    let seven = ao2mo_7d(&df, [set.as_slice(); 4], 1.0).expect("ao2mo_7d");

    let args = oracle_args(
        &cell,
        "sto-3g",
        "none",
        kmesh,
        "ao2mo_7d",
        mo_payload(&set),
        "1e-14",
    );
    let v = common::run_python(&py, ORACLE, &args);
    let w = worst_vs(&pull(&v, "re"), &pull(&v, "im"), &seven.data);
    println!("df_ao2mo.ao2mo_7d vs upstream (He-fcc {kmesh:?}, screens equalised): {w:e}");
    assert!(w < 1e-11, "ao2mo_7d vs upstream: {w:e}");
}

/// **Gate 1b** — diamond `gth-szv` at gamma, against upstream run with
/// `exclude_dd_block = False`, at 1e-11.
///
/// `#[ignore]` on cost, not on doubt: one diamond `make_j3c` at gamma is a
/// single screening group and runs for tens of minutes (14-02 SUMMARY). It is
/// the acceptance run, and the numbers it prints belong in
/// `14-VERIFICATION.md`.
#[test]
#[ignore = "acceptance: diamond make_j3c at gamma runs for tens of minutes"]
fn get_eri_matches_upstream_on_diamond_gamma() {
    let Some(py) = common::oracle_python() else {
        eprintln!("{} unset — skipping", common::GATE);
        return;
    };
    let cell = common::diamond();
    let kmesh = [1, 1, 1];
    let mut df = Gdf::new(cell.clone(), &[[0.0; 3]]);
    df.build().expect("gdf build");

    let quads: Vec<Vec<usize>> = vec![vec![0, 0, 0, 0]];
    let e = get_eri(&df, [0; 4], false).expect("get_eri").restore_s1();
    let args = oracle_args(
        &cell,
        "gth-szv",
        "gth-pade",
        kmesh,
        "get_eri",
        serde_json::to_string(&quads).expect("json"),
        "1e-14",
    );
    let v = common::run_python(&py, ORACLE, &args);
    let w = worst_vs(&pull(&v, "re"), &pull(&v, "im"), &e.data);
    println!("df_ao2mo.get_eri vs upstream (diamond gamma, dd-block off): {w:e}");
    assert!(w < 1e-11, "get_eri vs upstream on diamond: {w:e}");
}

/// **The attribution device.** Runs UPSTREAM's `df_ao2mo.get_eri` over the
/// PORT's own `cderi`, by handing a stub `mydf` whose `sr_loop` yields this
/// port's blocks.
///
/// Measured at **1.110e-16** — one ulp at `|eri| ~ 0.5`. The two sides run the
/// same contraction over the same inputs and differ only in SUMMATION ORDER:
/// this port reduces sequentially over `L`, upstream calls BLAS `ddot`, which
/// vectorises. (It has come out at 6e-36 on some builds; that is luck for a
/// small `naux`, not a guarantee, so the gate is stated at round-off and not at
/// zero.)
///
/// That separates the two questions the 1e-11 gate above conflates: *is the
/// contraction upstream's?* (yes, to round-off) and *is the `cderi` upstream's?*
/// (to 1.667e-12 with the screens equalised, 6.7e-9 against upstream's own
/// `strip_basis`). Without this test a future `cderi` regression would look
/// like a `df_ao2mo` regression, and the fix would go in the wrong file.
#[test]
#[ignore = "oracle: needs PYSCF_ORACLE_VENV and a real GDF build"]
fn get_eri_is_bit_exact_with_upstream_over_the_same_cderi() {
    let Some(py) = common::oracle_python() else {
        eprintln!("{} unset — skipping", common::GATE);
        return;
    };
    let cell = common::he_all_electron();
    let nao = cell.mol.nao_nr;
    let kmesh = [2, 1, 1];
    let kpts = kpts_of(&cell, kmesh);
    let nk = kpts.len();
    let mut df = Gdf::new(cell.clone(), &kpts);
    df.build().expect("gdf build");

    let mut blocks: Vec<Vec<f64>> = Vec::new();
    for ki in 0..nk {
        for kj in 0..nk {
            let mut flat = Vec::new();
            for b in df.sr_loop(ki, kj, false).expect("sr_loop") {
                let s = f64::from(b.sign);
                for i in 0..b.re.len() {
                    flat.push(s * b.re[i]);
                    flat.push(s * b.im[i]);
                }
            }
            blocks.push(flat);
        }
    }

    let kc = pyscf_pbc_lib::kpts_helper::get_kconserv(&cell.a, &kpts);
    let quads: Vec<Vec<usize>> = vec![
        vec![0, 0, 0, 0],
        vec![0, 1, 1, 0],
        vec![0, 1, 0, kc.get(0, 1, 0) as usize],
    ];
    let mut got = CTensor::default();
    for q in &quads {
        let e = get_eri(&df, [q[0], q[1], q[2], q[3]], false)
            .expect("get_eri")
            .restore_s1();
        got.re.extend_from_slice(&e.data.re);
        got.im.extend_from_slice(&e.data.im);
    }

    let a: Vec<Vec<f64>> = cell.a.iter().map(|r| r.to_vec()).collect();
    let xyz: Vec<Vec<f64>> = cell.mol.atom_coords().iter().map(|r| r.to_vec()).collect();
    let sym: Vec<String> = cell.mol._atom.iter().map(|(s, _)| s.clone()).collect();
    let args = vec![
        serde_json::to_string(&a).expect("json"),
        serde_json::to_string(&xyz).expect("json"),
        serde_json::to_string(&sym).expect("json"),
        serde_json::to_string(&kmesh.to_vec()).expect("json"),
        serde_json::to_string(&blocks).expect("json"),
        nao.to_string(),
        serde_json::to_string(&quads).expect("json"),
    ];
    let v = common::run_python(&py, ORACLE_SAME_CDERI, &args);
    let w = worst_vs(&pull(&v, "re"), &pull(&v, "im"), &got);
    println!("port get_eri vs UPSTREAM get_eri over the PORT's OWN cderi: {w:e}");
    assert!(
        w < 1e-14,
        "with identical inputs the two contractions may differ only by summation \
         order — one ulp at |eri| ~ 0.5 is 1.1e-16. Got {w:e}, which is a \
         difference in the ALGEBRA, not in the rounding"
    );
}

const ORACLE_SAME_CDERI: &str = r#"
import json, sys
import numpy
import pyscf
assert pyscf.__version__ == '2.12.1', pyscf.__version__
from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc.df import df_ao2mo

a, xyz, sym, kmesh, blocks, nao, quads = (
    json.loads(sys.argv[1]), json.loads(sys.argv[2]), json.loads(sys.argv[3]),
    json.loads(sys.argv[4]), json.loads(sys.argv[5]), int(sys.argv[6]),
    json.loads(sys.argv[7]))

cell = pgto.Cell()
cell.a = numpy.array(a)
cell.atom = [(s, tuple(c)) for s, c in zip(sym, xyz)]
cell.basis = 'sto-3g'
cell.unit = 'Bohr'
cell.verbose = 0
cell.build()
kpts = cell.make_kpts(kmesh)
nk = len(kpts)

class PortCderi:
    """A `mydf` whose only real method is sr_loop, yielding the PORT's blocks."""
    def __init__(self):
        self.cell = cell
        self.kpts = kpts
        self._cderi = 'already built'
        self.max_memory = 4000
    def build(self):
        raise AssertionError('build() must not be called: _cderi is set')
    def sr_loop(self, kpti_kptj, max_memory=2000, compact=True, blksize=None):
        ki = min(range(nk), key=lambda i: abs(kpts[i] - kpti_kptj[0]).sum())
        kj = min(range(nk), key=lambda j: abs(kpts[j] - kpti_kptj[1]).sum())
        v = numpy.array(blocks[ki * nk + kj], dtype=float).reshape(-1, nao * nao, 2)
        z = v[..., 0] + 1j * v[..., 1]
        if compact:
            z = lib.pack_tril(z.reshape(-1, nao, nao))
        yield z.real.copy(), z.imag.copy(), 1

mydf = PortCderi()
parts = [numpy.asarray(df_ao2mo.get_eri(mydf, kpts[q], compact=False)).reshape(-1)
         for q in quads]
e = numpy.concatenate(parts).astype(complex)
print(json.dumps({'re': e.real.tolist(), 'im': e.imag.tolist()}))
"#;
