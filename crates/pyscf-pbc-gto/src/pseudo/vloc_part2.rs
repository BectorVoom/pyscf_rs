//! `get_pp_loc_part2` — the SHORT-RANGE half of the GTH local pseudopotential,
//! as a real-space matrix (plan 10-05).
//!
//! Ports `pyscf/pbc/gto/pseudo/pp_int.py:118-165`
//! (`get_pp_loc_part2`, `get_pp_loc_part2_gamma`),
//! `pyscf/pbc/df/incore.py:491-560` (`aux_e2_sum_auxbas`,
//! `wrap_int3c_sum_auxbas`) and the C driver
//! `pyscf/lib/pbc/fill_ints_screened.c:208-300`
//! (`_nr3c_screened_sum_auxbas_fill_g`).
//!
//! # What it computes
//!
//! The `C_1 … C_4` polynomial terms of `V_loc` are s-type Gaussians on the
//! atoms, so their matrix elements are 3-centre integrals over an auxiliary
//! basis ([`crate::pseudo::fake_cell_vloc`]):
//!
//! ```text
//! V2[mu, nu] = Σ_cn Σ_P Σ_{Li, Lj} ( phi_mu(Li) phi_nu(Lj) | g_P^{cn} )
//! ```
//!
//! with the operator taken from [`PART2_INTORS`]: `cn = 1` is the plain
//! 3-centre overlap and `cn = 2, 3, 4` carry the `r²`, `r⁴`, `r⁶` weights about
//! the AUXILIARY centre (`origk`).
//!
//! Three conventions come straight from the C driver and must not drift:
//!
//! 1. **The auxiliary centre never moves.** `fill_ints_screened.c:266-281`
//!    shifts `ish` by `iL` and `jsh` by `jL`; `ksh` stays in the origin cell.
//!    That is what makes the DOUBLE lattice sum converge instead of diverging by
//!    a translation.
//! 2. **Both orbital centres are summed independently** — not over a single
//!    relative displacement.
//! 3. **Screening is per `(auxiliary, orbital)` shell pair**, from a neighbor
//!    list built between the auxiliary cell (at `precision = `[`EPS_PPL`]) and
//!    the real cell (`pp_int.py:151`).
//!
//! # Gamma point only
//!
//! Upstream's k-resolved branch (`pp_int.py:124-125`) routes through
//! `pyscf.pbc.df.aft._IntPPBuilder`, which needs `ft_ao` — Phase 13. The gamma
//! branch is the one Phase 10 owns, and [`get_pp_loc_part2`] refuses anything
//! else loudly rather than returning a gamma matrix for a k-point.
//!
//! # Screening
//!
//! Upstream's neighbor-list screen alone leaves `O(nimgs²)` shell triples per
//! `(ish, jsh, P)` — ~40 000 for diamond — which its OpenMP C driver absorbs and
//! a per-request safe API does not. This port therefore adds a SECOND,
//! mathematically-conservative screen on the Gaussian product itself
//! ([`prescreen_exponent`]), which is the standard bound
//!
//! ```text
//! |(ab|c)| <~ Cmax · exp( −θ_ab |A−B|² − θ_(ab)c |P_ab − C|² )
//! ```
//!
//! evaluated on each shell's MOST DIFFUSE primitive (the slowest-decaying one),
//! against [`PRESCREEN_EPS`] — four decades below `cell.precision`. Everything it
//! drops is provably below that bound; `tests/gth_pp_loc.rs` additionally pins
//! the result against upstream, which applies no such screen.

use crate::cell::Cell;
use crate::cutoff::{bas_exp, bas_nprim, libcint_ctr_coeff_max, pgf_rcut};
use crate::pseudo::vloc::{VlocAux, fake_cell_vloc};
use cintx_core::{BasisSet as CintxBasisSet, Representation};
use cintx_ops::resolver::Resolver;
use cintx_rs::{EvaluationContext, SessionRequest};
use cintx_runtime::ExecutionOptions;
use pyscf_core::{CoreError, ParsedAtom, ParsedBasis, PyscfRsError, ShellSpec};
use std::collections::HashMap;

/// `EPS_PPL` — `pp_int.py:34`. The precision the AUXILIARY cell is screened at;
/// deliberately looser than `cell.precision`, because the `C_n` terms are a
/// short-range correction whose absolute size is small.
pub const EPS_PPL: f64 = 1e-2;

/// The 3-centre operator per `C_n` term — `pp_int.py:148-149`, indexed by `cn`.
/// Index 0 (`int3c2e`, the erf/long-range term) is never used by
/// `get_pp_loc_part2`, whose loop runs `cn = 1..=4`; it is listed so the
/// indexing matches upstream's tuple exactly.
pub const PART2_INTORS: [&str; 5] = [
    "int3c2e",
    "int3c1e",
    "int3c1e_r2_origk",
    "int3c1e_r4_origk",
    "int3c1e_r6_origk",
];

/// The threshold of the Gaussian-product prescreen — see the module docs.
/// SIX decades below the default `cell.precision` of 1e-8: at 1e-12 the
/// accumulated effect of the dropped triples reached 7e-10 against upstream on
/// diamond, which is the same order as the acceptance gate itself.
pub const PRESCREEN_EPS: f64 = 1e-14;

/// `get_pp_loc_part2(cell, kpts=None)` — `pp_int.py:118-165`.
///
/// Returns the symmetric real `nao x nao` matrix in F-order (which, being
/// symmetric, is also its C-order transpose).
///
/// # Errors
/// * [`PyscfRsError::NotYetImplemented`] `{ phase: 13 }` when `kpts` holds
///   anything but the gamma point.
/// * [`CoreError::InvalidMolecule`] on a cintx failure — most likely the
///   `gth-pp` feature being off, which makes `int3c1e_r{2,4,6}_origk`
///   unavailable.
pub fn get_pp_loc_part2(cell: &Cell, kpts: &[[f64; 3]]) -> Result<Vec<f64>, PyscfRsError> {
    if kpts.iter().any(|k| !crate::pbc_intor::is_gamma(k)) {
        return Err(PyscfRsError::NotYetImplemented {
            phase: 13,
            what: "get_pp_loc_part2 away from the gamma point needs \
                   pyscf.pbc.df.aft._IntPPBuilder (ft_ao)",
        });
    }
    get_pp_loc_part2_gamma(cell)
}

/// `get_pp_loc_part2_gamma(cell)` — `pp_int.py:128-165`.
///
/// # Errors
/// As [`get_pp_loc_part2`].
pub fn get_pp_loc_part2_gamma(cell: &Cell) -> Result<Vec<f64>, PyscfRsError> {
    let nao = cell.mol.nao_nr;
    let mut out = vec![0.0_f64; nao * nao];
    if cell.pseudo.is_none() || nao == 0 {
        return Ok(out);
    }

    // pp_int.py:153 — ONE image list for every C_n term.
    let ls = crate::lattice::get_lattice_ls_default(cell)?;
    // Per-shell radii of the real cell, at cell.precision (neighborlist.py:88).
    let cell_rcut = cell.rcut_by_shells(None);

    for cn in 1..=4usize {
        let aux = fake_cell_vloc(cell, cn)?;
        if aux.is_empty() {
            continue;
        }
        accumulate_cn(cell, &aux, cn, &ls, &cell_rcut, &mut out)?;
    }
    Ok(out)
}

/// How many KET images share one cintx `BasisSet`.
///
/// A `SessionRequest` costs O(total shells in the basis) — measured at ~15 us
/// fixed plus ~0.12 us per shell — so a basis holding every lattice image makes
/// each of the millions of triples in this double sum 20x more expensive than
/// it needs to be. Building `[bra image | KET_CHUNK ket images | auxiliaries]`
/// keeps every request near the fixed-cost floor while amortising the basis
/// construction over `KET_CHUNK * nbas^2 * naux` evaluations.
const KET_CHUNK: usize = 16;

/// One `C_n` term's contribution.
fn accumulate_cn(
    cell: &Cell,
    aux: &[VlocAux],
    cn: usize,
    ls: &[[f64; 3]],
    cell_rcut: &[f64],
    out: &mut [f64],
) -> Result<(), PyscfRsError> {
    let nao = cell.mol.nao_nr;
    let nbas = cell.mol.nbas;
    let coords = cell.mol.atom_coords();
    let naux = aux.len();

    // --- The auxiliary radii, at EPS_PPL (pp_int.py:139). ---
    // `rcut_by_shells` on the fake cell reduces, for a one-primitive s shell,
    // to `pgf_rcut(l=0, alpha, |coeff|, EPS_PPL)`.
    let aux_rcut: Vec<f64> = aux
        .iter()
        .map(|a| {
            pgf_rcut(
                0,
                a.alpha,
                a.coeff.abs(),
                EPS_PPL,
                0.0,
                crate::cutoff::RCUT_MAX_CYCLE,
                crate::cutoff::RCUT_EPS,
            )
        })
        .collect();

    let shell_atom = |s: usize| -> usize {
        use pyscf_core::raw_layout::{ATOM_OF, BAS_SLOTS};
        cell.mol._bas[s * BAS_SLOTS + ATOM_OF] as usize
    };

    // --- Upstream's neighbor-list screen, `|R_s + L - R_P| < r_s + r_P`
    //     (`build_neighbor_list_for_shlpairs(fake_cell, cell, Ls)`,
    //     pp_int.py:151). `reach[k][s]` is the image list of shell `s` around
    //     auxiliary `k`. ---
    let reach: Vec<Vec<Vec<usize>>> = (0..naux)
        .map(|k| {
            let rp = coords[aux[k].atom];
            (0..nbas)
                .map(|s| {
                    let rmax = cell_rcut[s] + aux_rcut[k];
                    let rs = coords[shell_atom(s)];
                    ls.iter()
                        .enumerate()
                        .filter(|(_, l)| {
                            let d = [
                                rs[0] + l[0] - rp[0],
                                rs[1] + l[1] - rp[1],
                                rs[2] + l[2] - rp[2],
                            ];
                            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() < rmax
                        })
                        .map(|(m, _)| m)
                        .collect()
                })
                .collect()
        })
        .collect();

    // Images that any (shell, auxiliary) pair reaches — the BRA loop's domain.
    let mut used: Vec<usize> = reach
        .iter()
        .flatten()
        .flatten()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    used.sort_unstable();
    if used.is_empty() {
        return Ok(());
    }

    // --- Per-shell screening data for the Gaussian-product prescreen. ---
    let diffuse: Vec<(f64, f64)> = (0..nbas).map(|s| most_diffuse(cell, s)).collect();

    // --- The auxiliary basis (unit-normalised; the raw-coefficient ratio is
    //     applied to each finished block). ---
    let aux_atoms: Vec<ParsedAtom> = aux.iter().map(|a| cell.mol._atom[a.atom].clone()).collect();
    let mut aux_basis: HashMap<String, ParsedBasis> = HashMap::new();
    for a in aux {
        let key = crate::pseudo::normalise_symbol(&cell.mol._atom[a.atom].0);
        aux_basis.entry(key).or_insert_with(|| ParsedBasis {
            shells: vec![ShellSpec {
                l: 0,
                exponents: vec![a.alpha],
                coeffs: vec![vec![1.0]],
            }],
        });
    }

    let intor = PART2_INTORS[cn];
    let descriptor = Resolver::descriptor_by_symbol(&pyscf_gto::add_suffix(intor, cell.mol.cart))
        .map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cintx-ops resolver does not know '{intor}': {e}. The `gth-pp` feature \
                     (cintx `unstable-source-api`) must be on for the origk family."
        )))
    })?;
    let representation = if cell.mol.cart {
        Representation::Cart
    } else {
        Representation::Spheric
    };
    let opts = ExecutionOptions::default();
    // One evaluation context for the whole term: it caches the backend client
    // and a reusable host scratch arena across requests.
    let ctx = EvaluationContext::new();

    // AO offsets/counts come off a probe basis; they are image-independent.
    let (probe, probe_nbas, probe_naux) = build_basis(cell, ls, &[0, 0], &aux_atoms, &aux_basis)?;
    if probe_nbas != nbas || probe_naux != naux {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "get_pp_loc_part2: basis layout mismatch (shells {probe_nbas} vs {nbas}, \
             auxiliaries {probe_naux} vs {naux}); a ghost atom or a mixed-element \
             auxiliary basis?"
        ))));
    }
    let pmeta = probe.meta();
    let ao_off: Vec<usize> = (0..nbas)
        .map(|s| pmeta.shell_offset(s).unwrap_or(0))
        .collect();
    let ao_cnt: Vec<usize> = (0..nbas).map(|s| pmeta.ao_count(s).unwrap_or(0)).collect();
    drop(probe);

    // --- The double lattice sum, bra image outer. ---
    let mut triples: Vec<(usize, usize, usize, usize)> = Vec::new(); // (mj, ish, jsh, k)
    for &mi in &used {
        triples.clear();
        for k in 0..naux {
            let rp = coords[aux[k].atom];
            let ck = aux[k].alpha;
            for ish in 0..nbas {
                if ao_cnt[ish] == 0 || !reach[k][ish].contains(&mi) {
                    continue;
                }
                let ri = coords[shell_atom(ish)];
                let ai = [ri[0] + ls[mi][0], ri[1] + ls[mi][1], ri[2] + ls[mi][2]];
                let (ea, ca) = diffuse[ish];
                for jsh in 0..nbas {
                    if ao_cnt[jsh] == 0 {
                        continue;
                    }
                    let rj = coords[shell_atom(jsh)];
                    let (eb, cb) = diffuse[jsh];
                    let cmax = ca * cb * aux[k].coeff.abs();
                    for &mj in &reach[k][jsh] {
                        let bj = [rj[0] + ls[mj][0], rj[1] + ls[mj][1], rj[2] + ls[mj][2]];
                        if prescreen_exponent(ea, &ai, eb, &bj, ck, &rp) * cmax < PRESCREEN_EPS {
                            continue;
                        }
                        triples.push((mj, ish, jsh, k));
                    }
                }
            }
        }
        if triples.is_empty() {
            continue;
        }
        triples.sort_unstable();

        // Chunk the distinct ket images so each basis stays small.
        let mut cursor = 0usize;
        while cursor < triples.len() {
            let mut kets: Vec<usize> = Vec::with_capacity(KET_CHUNK);
            let start = cursor;
            while cursor < triples.len() && kets.len() < KET_CHUNK {
                let mj = triples[cursor].0;
                if kets.last() != Some(&mj) {
                    if kets.len() == KET_CHUNK {
                        break;
                    }
                    kets.push(mj);
                }
                cursor += 1;
            }
            // `shifts[0]` is the bra image; `shifts[1 + p]` are the ket images.
            let mut shifts: Vec<usize> = Vec::with_capacity(1 + kets.len());
            shifts.push(mi);
            shifts.extend_from_slice(&kets);
            let (basis, _, _) = build_basis(cell, ls, &shifts, &aux_atoms, &aux_basis)?;
            let aux_shell0 = shifts.len() * nbas;

            for &(mj, ish, jsh, k) in &triples[start..cursor] {
                let p = kets.iter().position(|m| *m == mj).expect("ket in chunk");
                let di = ao_cnt[ish];
                let dj = ao_cnt[jsh];
                let block = eval3c(
                    &basis,
                    &ctx,
                    descriptor.id,
                    representation,
                    &opts,
                    ish,
                    (1 + p) * nbas + jsh,
                    aux_shell0 + k,
                    di * dj,
                    intor,
                )?;
                // Block F-order [di, dj]; the auxiliary shell is s, so dk = 1.
                let scale = aux[k].rescale_from_unit_norm();
                let oi = ao_off[ish];
                let oj = ao_off[jsh];
                for jj in 0..dj {
                    for ii in 0..di {
                        out[(oi + ii) + (oj + jj) * nao] += scale * block[ii + jj * di];
                    }
                }
            }
        }
    }
    Ok(())
}

/// `[cell + Ls[shifts[0]] | cell + Ls[shifts[1]] | … | auxiliaries]`.
fn build_basis(
    cell: &Cell,
    ls: &[[f64; 3]],
    shifts: &[usize],
    aux_atoms: &[ParsedAtom],
    aux_basis: &HashMap<String, ParsedBasis>,
) -> Result<(std::sync::Arc<CintxBasisSet>, usize, usize), PyscfRsError> {
    let translations: Vec<[f64; 3]> = shifts.iter().map(|m| ls[*m]).collect();
    pyscf_gto::build_image_expanded_with_aux(
        &cell.mol._atom,
        &cell.mol._basis,
        cell.mol.cart,
        &translations,
        aux_atoms,
        aux_basis,
        cell.mol.cart,
    )
}

/// Evaluate one 3-centre shell triple.
#[allow(clippy::too_many_arguments)]
fn eval3c(
    basis: &CintxBasisSet,
    ctx: &EvaluationContext,
    operator: cintx_core::OperatorId,
    representation: Representation,
    opts: &ExecutionOptions,
    i: usize,
    j: usize,
    k: usize,
    expected: usize,
    name: &str,
) -> Result<Vec<f64>, PyscfRsError> {
    let shells = basis.shell_tuple_for_indices([i, j, k]).map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "shell_tuple_for_indices({i}, {j}, {k}) failed for '{name}': {e}"
        )))
    })?;
    let outcome = SessionRequest::new(operator, representation, basis, shells, opts.clone())
        .query_workspace_in(ctx)
        .map_err(|e| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "cintx workspace query failed for '{name}' triple ({i},{j},{k}): {e}"
            )))
        })?
        .evaluate()
        .map_err(|e| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "cintx evaluate failed for '{name}' triple ({i},{j},{k}): {e}"
            )))
        })?;
    let v = &outcome.tensor.owned_values;
    if v.len() != expected {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cintx returned {} elements for '{name}' triple ({i},{j},{k}), expected \
             {expected} (extents {:?})",
            v.len(),
            outcome.tensor.extents,
        ))));
    }
    Ok(v.clone())
}

/// `(exponent, |coefficient|)` of a shell's most DIFFUSE primitive — the one
/// that decays slowest and therefore bounds the whole shell.
fn most_diffuse(cell: &Cell, bas_id: usize) -> (f64, f64) {
    let es = bas_exp(cell, bas_id);
    let cs = libcint_ctr_coeff_max(cell, bas_id);
    let n = bas_nprim(cell, bas_id).min(es.len()).min(cs.len());
    let mut best = (f64::INFINITY, 0.0);
    for p in 0..n {
        if es[p] < best.0 {
            best = (es[p], cs[p]);
        }
    }
    if best.0.is_finite() { best } else { (1.0, 0.0) }
}

/// The exponential bound on `|(ab|c)|`:
///
/// ```text
/// exp( −θ_ab |A−B|² − θ_(ab)c |P_ab − C|² ),
/// θ_ab = ab/(a+b),  P_ab = (aA+bB)/(a+b),  θ_(ab)c = (a+b)c/(a+b+c)
/// ```
///
/// Both factors are exact Gaussian-product identities, so the product of this
/// and the coefficient magnitudes is a genuine (if loose) upper bound on the
/// integral — dropping a triple below [`PRESCREEN_EPS`] cannot move the result
/// by more than that.
pub fn prescreen_exponent(
    a: f64,
    ra: &[f64; 3],
    b: f64,
    rb: &[f64; 3],
    c: f64,
    rc: &[f64; 3],
) -> f64 {
    let ab = a + b;
    let mut d2 = 0.0;
    let mut p = [0.0_f64; 3];
    for x in 0..3 {
        let d = ra[x] - rb[x];
        d2 += d * d;
        p[x] = (a * ra[x] + b * rb[x]) / ab;
    }
    let mut d2pc = 0.0;
    for x in 0..3 {
        let d = p[x] - rc[x];
        d2pc += d * d;
    }
    let e1 = a * b / ab * d2;
    let e2 = ab * c / (ab + c) * d2pc;
    (-(e1 + e2)).exp()
}
