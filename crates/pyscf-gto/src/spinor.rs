//! F-03 — spinor (relativistic 2-component) integral representation.
//!
//! Implements `intor_spinor(mol, name)` for the scalar one-electron spinor
//! families (`int1e_{ovlp,kin,nuc}_spinor`), tracked in
//! `.planning/features/F-03-spinor-integrals-PLAN.md`.
//!
//! ## Architecture — route through cintx (not transform-from-spherical)
//!
//! The original plan proposed building spinor integrals from the spherical
//! integrals via a hand-transcribed `sph→spinor` Clebsch–Gordan congruence,
//! on the premise that cintx did not ship spinor operators. That premise was
//! wrong: cintx **does** provide a libcint-parity-validated cart→spinor
//! transform (`cintx-cubecl` `transform::c2spinor`, with the
//! `g_trans_cart2j` CG tables in `c2spinor_coeffs`) and resolves `_spinor`
//! operator symbols. cintx's own oracle suite gates the scalar 1e spinor path
//! against vendor libcint at `atol = 1e-12`
//! (`cintx-oracle/tests/one_electron_scalar_spinor_parity.rs`).
//!
//! So `intor_spinor` mirrors the real [`crate::intor`] arity-2 path: build a
//! **`Representation::Spinor`-tagged** [`BasisSet`](cintx_core::BasisSet) (so
//! cintx applies its cart→spinor transform and reports `n2c`-unit AO counts),
//! evaluate each shell pair, read the complex block via
//! [`complex_values`](cintx_rs::IntegralTensor::complex_values), and stitch
//! into a single F-order [`IntorOutputComplex`]. The unverified hand-written
//! CG transcription is gone; correctness of the per-pair numerics inherits
//! cintx's vendor parity.
//!
//! ## Why spinor integrals need their own surface
//!
//! Spinor integrals are **complex-valued**: the cart→spinor transform couples
//! the cartesian gaussians × {α,β} spin functions into the `j = l ± 1/2`
//! adapted basis using complex CG coefficients. The non-relativistic
//! `IntorOutput` carries `Vec<f64>` (real only), so the spinor path returns a
//! distinct [`IntorOutputComplex`] rather than mutating the real type that
//! scf/mp2/df/PyO3 all consume.

use std::sync::Arc;

use cintx_core::{BasisSet as CintxBasisSet, Representation};
use cintx_ops::resolver::Resolver;
use cintx_rs::SessionRequest;
use cintx_runtime::ExecutionOptions;

use crate::layout_table::IntorLayout;
use pyscf_core::{CoreError, Mole, PyscfRsError};

/// Complex integral output — the spinor analogue of [`crate::IntorOutput`].
///
/// `re`/`im` are parallel flat F-order buffers (same length, same logical
/// `shape`), so element `k` of the matrix is `re[k] + i·im[k]`. Kept as two
/// `f64` buffers (rather than pulling in a `num_complex` dependency) so the
/// type is self-contained and trivially FFI/serialise-friendly.
#[derive(Debug, Clone)]
pub struct IntorOutputComplex {
    /// Real part, flat F-order. `re.len() == shape.iter().product()`.
    pub re: Vec<f64>,
    /// Imaginary part, flat F-order. `im.len() == re.len()`.
    pub im: Vec<f64>,
    /// Logical shape; for an arity-2 spinor integral this is `[n2c, n2c]`.
    pub shape: Vec<usize>,
    /// Per-intor layout convention (Pitfall 8), shared with the real path.
    pub layout: IntorLayout,
}

impl IntorOutputComplex {
    /// Number of complex elements (`re.len()`), i.e. `shape.iter().product()`.
    pub fn len(&self) -> usize {
        self.re.len()
    }

    /// True when the buffer holds no elements.
    pub fn is_empty(&self) -> bool {
        self.re.is_empty()
    }
}

/// Number of 2-component spinor AOs contributed by a single shell of angular
/// momentum `l` with `nctr` contractions.
///
/// A shell of angular momentum `l` splits into `j = l − 1/2` (`2j+1 = 2l`
/// states) and `j = l + 1/2` (`2j+1 = 2l+2` states); the total per
/// contraction is `2l + (2l+2) = 2(2l+1)` — exactly twice the spherical count
/// `(2l+1)`. Hence the whole-molecule relation `n2c = 2 · nao_sph`.
///
/// This is pure arithmetic and fully unit-tested below; it is the building
/// block `mol.nao_2c` sums over shells (task T1, see `build_from`).
pub fn n2c_per_shell(l: u32, nctr: usize) -> usize {
    nctr * 2 * (2 * l as usize + 1)
}

/// Normalise an intor name to its `_spinor` symbol: strip any `_sph`/`_cart`
/// suffix and ensure a single trailing `_spinor`. `int1e_ovlp`,
/// `int1e_ovlp_sph`, and `int1e_ovlp_spinor` all map to `int1e_ovlp_spinor`.
fn spinor_symbol(name: &str) -> String {
    let base = name
        .strip_suffix("_spinor")
        .or_else(|| name.strip_suffix("_sph"))
        .or_else(|| name.strip_suffix("_cart"))
        .unwrap_or(name);
    format!("{base}_spinor")
}

/// Compute a named **complex** spinor integral, the 2-component analogue of
/// [`crate::intor`]. Covers the scalar one-electron families
/// (`int1e_{ovlp,kin,nuc}_spinor`, arity 2 → `[n2c, n2c]`) and the
/// two-electron ERIs (`int2e_spinor`, arity 4 → `[n2c, n2c, n2c, n2c]`).
///
/// Mirrors the real dispatch (see module docs): builds a
/// `Representation::Spinor` basis, evaluates each shell tuple through cintx, and
/// stitches the per-tuple complex blocks (F-order) into an
/// [`IntorOutputComplex`]. The per-tuple cart→spinor numerics are cintx's,
/// vendor-libcint-validated.
///
/// Errors:
/// - `Core(InvalidMolecule)` — unbuilt Mole, unknown/unsupported symbol, or a
///   cintx workspace/evaluate failure.
/// - `NotYetImplemented{phase:3}` — a symbol that is neither arity 2 nor 4, or
///   `int2e_spinor` on a general-contracted basis (cintx wires the 4D
///   cart→spinor transform for segmented `nctr==1` shells only).
pub fn intor_spinor(mol: &Mole, name: &str) -> Result<IntorOutputComplex, PyscfRsError> {
    if !mol._built {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "Mole not built — call pyscf_gto::M(args) or mol.build() first".into(),
        )));
    }

    let full_name = spinor_symbol(name);

    // cintx OperatorId resolution (strips `_spinor`, consults the manifest).
    let descriptor = Resolver::descriptor_by_symbol(&full_name).map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cintx-ops resolver does not know spinor symbol '{full_name}': {e}. \
             Bump cintx or check the operator name."
        )))
    })?;

    if !descriptor
        .entry
        .supports_representation(Representation::Spinor)
    {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cintx does not provide a spinor representation for '{full_name}'",
        ))));
    }

    let operator = descriptor.id;
    let arity = descriptor.entry.arity as usize;
    if arity != 2 && arity != 4 {
        return Err(PyscfRsError::NotYetImplemented {
            phase: 3,
            what: "intor_spinor supports arity-2 one-electron families \
                   (int1e_{ovlp,kin,nuc}_spinor) and arity-4 two-electron \
                   (int2e_spinor); this symbol is neither",
        });
    }

    // Spinor-tagged basis: cintx applies its cart→spinor transform and reports
    // n2c-unit AO offsets/counts via the representation-aware BasisMeta.
    let basis_arc: Arc<CintxBasisSet> =
        crate::projection::build_cintx_spinor_basis_set(&mol._atom, &mol._basis)?;
    let basis: &CintxBasisSet = &basis_arc;
    let nbas = basis.shells().len();
    let meta = basis.meta();
    let n2c = meta.total_ao;

    // The spinor basis AO count must agree with mol.nao_2c (F-03 T1) — both are
    // Σ 2·(2l+1)·nctr over shells. A mismatch means the two builders diverged.
    if n2c != mol.nao_2c {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "spinor basis total_ao={n2c} disagrees with mol.nao_2c={} for '{full_name}'",
            mol.nao_2c,
        ))));
    }

    let shell_offsets: Vec<usize> = (0..nbas)
        .map(|s| meta.shell_offset(s).unwrap_or(0))
        .collect();
    let shell_counts: Vec<usize> = (0..nbas).map(|s| meta.ao_count(s).unwrap_or(0)).collect();

    let (re, im, shape) = match arity {
        2 => evaluate_spinor_arity2(
            operator,
            basis,
            nbas,
            n2c,
            &shell_offsets,
            &shell_counts,
            &full_name,
        )?,
        4 => {
            // cintx wires the 4D cart→spinor transform for SEGMENTED shells only
            // (`two_electron.rs`: nctr>1 → UnsupportedApi). Guard upfront with a
            // clear message rather than failing deep in the quartet loop.
            ensure_segmented_for_spinor_2e(mol)?;
            evaluate_spinor_arity4(
                operator,
                basis,
                nbas,
                n2c,
                &shell_offsets,
                &shell_counts,
                &full_name,
            )?
        }
        _ => unreachable!("arity already validated to be 2 or 4"),
    };

    Ok(IntorOutputComplex {
        re,
        im,
        shape,
        layout: IntorLayout::ScalarFOrder,
    })
}

/// cintx's 4D cart→spinor transform only covers segmented shells (`nctr == 1`);
/// general contraction (`nctr > 1`) returns `UnsupportedApi`. Surface that as a
/// clean, documented error before the (expensive) quartet loop.
fn ensure_segmented_for_spinor_2e(mol: &Mole) -> Result<(), PyscfRsError> {
    use pyscf_core::raw_layout::{BAS_SLOTS, NCTR_OF};
    let nbas = mol._bas.len() / BAS_SLOTS;
    for s in 0..nbas {
        if mol._bas[s * BAS_SLOTS + NCTR_OF] != 1 {
            return Err(PyscfRsError::NotYetImplemented {
                phase: 3,
                what: "int2e_spinor currently requires a SEGMENTED basis (nctr==1 per \
                       shell): cintx does not wire the 4D cart→spinor transform for \
                       general contraction (nctr>1). Use a segmented basis (e.g. STO-3G) \
                       or decontract first.",
            });
        }
    }
    Ok(())
}

/// Evaluate + stitch a complex arity-2 spinor integral into F-order
/// `[n2c, n2c]` `(re, im)` buffers. Mirrors the validated scalar
/// `intor::stitch_arity2_block` (per-pair block F-order, bra-fastest).
#[allow(clippy::type_complexity)]
fn evaluate_spinor_arity2(
    operator: cintx_core::OperatorId,
    basis: &CintxBasisSet,
    nbas: usize,
    n2c: usize,
    shell_offsets: &[usize],
    shell_counts: &[usize],
    full_name: &str,
) -> Result<(Vec<f64>, Vec<f64>, Vec<usize>), PyscfRsError> {
    let mut re = vec![0.0f64; n2c * n2c];
    let mut im = vec![0.0f64; n2c * n2c];

    for i in 0..nbas {
        for j in 0..nbas {
            let cv = eval_spinor_block(operator, basis, &[i, j], full_name)?;
            let ni = shell_counts[i];
            let nj = shell_counts[j];
            let oi = shell_offsets[i];
            let oj = shell_offsets[j];

            if cv.len() != ni * nj {
                return Err(block_size_err(full_name, &[i, j], cv.len(), ni * nj));
            }

            // block[ii + jj*ni] → global F-order (oi+ii) + (oj+jj)*n2c.
            for jj in 0..nj {
                for ii in 0..ni {
                    let (cr, ci) = cv[ii + jj * ni];
                    let g = (oi + ii) + (oj + jj) * n2c;
                    re[g] = cr;
                    im[g] = ci;
                }
            }
        }
    }

    Ok((re, im, vec![n2c, n2c]))
}

/// Evaluate + stitch a complex arity-4 spinor ERI (`int2e_spinor`) into F-order
/// `[n2c, n2c, n2c, n2c]` `(re, im)` buffers. Mirrors the scalar
/// `intor::evaluate_arity4` quartet stitch (per-quad block F-order, i-fastest).
#[allow(clippy::type_complexity)]
fn evaluate_spinor_arity4(
    operator: cintx_core::OperatorId,
    basis: &CintxBasisSet,
    nbas: usize,
    n2c: usize,
    shell_offsets: &[usize],
    shell_counts: &[usize],
    full_name: &str,
) -> Result<(Vec<f64>, Vec<f64>, Vec<usize>), PyscfRsError> {
    let total = n2c.checked_pow(4).ok_or_else(|| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "intor_spinor '{full_name}': n2c={n2c} overflows the arity-4 buffer (n2c^4)",
        )))
    })?;
    let mut re = vec![0.0f64; total];
    let mut im = vec![0.0f64; total];

    let n2c2 = n2c * n2c;
    let n2c3 = n2c2 * n2c;

    for i in 0..nbas {
        for j in 0..nbas {
            for k in 0..nbas {
                for l in 0..nbas {
                    let cv = eval_spinor_block(operator, basis, &[i, j, k, l], full_name)?;
                    let ni = shell_counts[i];
                    let nj = shell_counts[j];
                    let nk = shell_counts[k];
                    let nl = shell_counts[l];
                    let oi = shell_offsets[i];
                    let oj = shell_offsets[j];
                    let ok = shell_offsets[k];
                    let ol = shell_offsets[l];

                    let expected = ni * nj * nk * nl;
                    if cv.len() != expected {
                        return Err(block_size_err(full_name, &[i, j, k, l], cv.len(), expected));
                    }

                    // block[ii + jj*ni + kk*ni*nj + ll*ni*nj*nk] → global F-order
                    // (oi+ii) + (oj+jj)*n2c + (ok+kk)*n2c^2 + (ol+ll)*n2c^3.
                    for ll in 0..nl {
                        for kk in 0..nk {
                            for jj in 0..nj {
                                for ii in 0..ni {
                                    let b = ii + jj * ni + kk * ni * nj + ll * ni * nj * nk;
                                    let g = (oi + ii)
                                        + (oj + jj) * n2c
                                        + (ok + kk) * n2c2
                                        + (ol + ll) * n2c3;
                                    let (cr, ci) = cv[b];
                                    re[g] = cr;
                                    im[g] = ci;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok((re, im, vec![n2c, n2c, n2c, n2c]))
}

/// Evaluate one spinor shell tuple through cintx and return its complex block
/// as `(re, im)` pairs (avoids a direct `num_complex` dependency; the order
/// matches `IntegralTensor::complex_values`).
fn eval_spinor_block(
    operator: cintx_core::OperatorId,
    basis: &CintxBasisSet,
    idx: &[usize],
    full_name: &str,
) -> Result<Vec<(f64, f64)>, PyscfRsError> {
    let shells = basis
        .shell_tuple_for_indices(idx.iter().copied())
        .map_err(|e| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "shell_tuple_for_indices({idx:?}) failed for '{full_name}': {e}",
            )))
        })?;

    let outcome = SessionRequest::new(
        operator,
        Representation::Spinor,
        basis,
        shells,
        ExecutionOptions::default(),
    )
    .query_workspace()
    .map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cintx workspace query failed for '{full_name}' shells {idx:?}: {e}",
        )))
    })?
    .evaluate()
    .map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cintx evaluate failed for '{full_name}' shells {idx:?}: {e}",
        )))
    })?;

    let cv = outcome.tensor.complex_values().ok_or_else(|| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cintx spinor evaluate for '{full_name}' shells {idx:?} returned a real tensor \
             (complex_interleaved == false)",
        )))
    })?;
    Ok(cv.into_iter().map(|c| (c.re, c.im)).collect())
}

fn block_size_err(full_name: &str, idx: &[usize], got: usize, expected: usize) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
        "spinor block {idx:?} for '{full_name}' has {got} elements; expected {expected}",
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Verifiable-today scaffold: the AO-count formula ──────────────────

    /// `n2c_per_shell` against hand-computed values for s/p/d/f and the
    /// `n2c = 2·(2l+1)·nctr` relation. Pure arithmetic — no oracle needed.
    #[test]
    fn n2c_per_shell_matches_2_times_spherical() {
        // (l, nctr) → expected 2-component count.
        assert_eq!(n2c_per_shell(0, 1), 2); // s: 2·1
        assert_eq!(n2c_per_shell(1, 1), 6); // p: 2·3
        assert_eq!(n2c_per_shell(2, 1), 10); // d: 2·5
        assert_eq!(n2c_per_shell(3, 1), 14); // f: 2·7
        // nctr scaling + the 2·nsph relation.
        for l in 0..=6u32 {
            for nctr in 0..=4usize {
                let nsph = nctr * (2 * l as usize + 1);
                assert_eq!(n2c_per_shell(l, nctr), 2 * nsph, "l={l} nctr={nctr}");
            }
        }
    }

    /// The complex container's `len`/`is_empty` track the real buffer.
    #[test]
    fn complex_output_len_tracks_buffers() {
        let out = IntorOutputComplex {
            re: vec![1.0, 2.0, 3.0, 4.0],
            im: vec![0.0, -1.0, 0.0, 1.0],
            shape: vec![2, 2],
            layout: IntorLayout::ScalarFOrder,
        };
        assert_eq!(out.len(), 4);
        assert_eq!(out.len(), out.shape.iter().product::<usize>());
        assert!(!out.is_empty());
    }

    // ── Name normalisation ───────────────────────────────────────────────

    #[test]
    fn spinor_symbol_normalises_suffixes() {
        assert_eq!(spinor_symbol("int1e_ovlp"), "int1e_ovlp_spinor");
        assert_eq!(spinor_symbol("int1e_ovlp_sph"), "int1e_ovlp_spinor");
        assert_eq!(spinor_symbol("int1e_ovlp_cart"), "int1e_ovlp_spinor");
        assert_eq!(spinor_symbol("int1e_ovlp_spinor"), "int1e_ovlp_spinor");
        // No double suffix.
        assert!(!spinor_symbol("int1e_kin_spinor").ends_with("_spinor_spinor"));
    }

    #[test]
    fn intor_spinor_rejects_unbuilt_mole() {
        let mol = Mole::default();
        let r = intor_spinor(&mol, "int1e_ovlp_spinor");
        assert!(
            matches!(r, Err(PyscfRsError::Core(_))),
            "unbuilt Mole must error before touching cintx, got {r:?}"
        );
    }

    // End-to-end numerical verification (shape, n2c, Hermiticity, finiteness)
    // lives in `tests/spinor_intor.rs` — it needs the public `M` builder.
    // Per-pair cart→spinor numerics inherit cintx's vendor-libcint parity
    // (`cintx-oracle/.../one_electron_scalar_spinor_parity.rs`); the global
    // shell-block ordering vs upstream PySCF remains the live-PySCF gate
    // documented as `#[ignore]`d in the integration suite.
}
