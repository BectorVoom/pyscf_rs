//! `KRHF` restricted to the irreducible Brillouin zone — `khf_ksymm.py`
//! (410 l), plan 17-07.
//!
//! # The whole mechanism, in one sentence
//!
//! [`KOverrideHooks::kpts`] returns `kpts.kpts_ibz` instead of the full BZ,
//! and every hook that sums over k learns which weight to use. There is **no
//! second driver**: [`crate::kscf::kernel`] is untouched, and this type is
//! another implementation of the same eleven-method trait `Krhf` implements
//! (D-PBC-15). `nkpts`, the `idx(set, k) = set * nkpts + k` layout in
//! [`KScfResult`], and DIIS's Fock subspace all then operate on IBZ indices
//! with no change anywhere else.
//!
//! # Every weighted sum in this module, named (17-CONTEXT §3.5)
//!
//! `15-CONTEXT §3` recorded the analogous KMP2 trap — `1/nkpts` appeared at
//! three sites and was *two distinct divisions*. So the table is written out
//! rather than left to a diff against upstream:
//!
//! | quantity | upstream | weight | why |
//! |---|---|---|---|
//! | `energy_elec`'s `e1` | `khf_ksymm.py:76` | **`weights_ibz`** | one term per IBZ point, weighted by its star size |
//! | `energy_elec`'s `e_coul` | `:77` | **`weights_ibz`** | same |
//! | `get_init_guess`'s electron count | `:397` | **`weights_ibz`** | `ne = einsum('k,kij,kji', weights_ibz, dm, s1e).real` |
//! | `get_init_guess`'s rescale target | `:398` | **`* nkpts`** | the BZ count — `weights_ibz` already summed to 1, so this restores the supercell electron number |
//! | `nelectron` | `:38` | **`nkpts`, NOT `nkpts_ibz`** | `cell.tot_electrons(kpts.nkpts)` — the Fermi level is a full-BZ quantity (§3.4) |
//! | `get_occ`'s aufbau sort | `:39-40` | **unfolded BZ**, bare | the occupations are assigned over every BZ point, then folded back |
//! | `get_rho` | `:101` | unfold first | the density is a real-space BZ sum |
//!
//! **`weights_ibz` sums to 1**, so an `energy_elec` written against it needs
//! no further `1/nkpts`. Writing `1/nkpts` here instead would silently drop
//! every star multiplicity — the single most likely defect in this file, and
//! invisible on any cell whose stars all happen to have the same size.
//!
//! # What is deliberately NOT done here
//!
//! [`KScfResult`] keeps **IBZ-length** arrays. Nothing is unfolded silently at
//! the end of the driver: upstream does not (`khf_ksymm.py:336`'s `to_khf()`
//! is an explicit converter), and a silent unfold would make `mo_coeff`'s
//! meaning depend on which class produced it.

use num_complex::Complex64;

use pyscf_algebra::CTensor;
use pyscf_core::PyscfRsError;
use pyscf_pbc_df::{JkOpts, PeriodicDf};
use pyscf_pbc_gto::{Cell, ExxDiv};
use pyscf_pbc_symm::kpts::KPoints;

use crate::khooks::KOverrideHooks;
use crate::krdm::make_rdm1;
use crate::kscf::kernel;
use crate::types::{KDms, KInitGuess, KMats, KScfConfig, KScfResult};

/// How equal two star members' occupations must be before
/// `check_mo_occ_symmetry` calls the solution symmetry-broken.
///
/// Upstream's own threshold (`kpts.py:717`'s `RuntimeError`). This is a
/// *physical* condition, not an internal tolerance: exceeding it means the
/// SCF converged to a state that does not carry the lattice symmetry, which
/// is a real (and occasionally desired) outcome — hence a typed error naming
/// the two k-points, never a panic.
pub const OCC_SYMMETRY_TOL: f64 = 1e-4;

/// Which `get_jk` route [`KsymAdaptedKrhf::get_veff`] takes.
///
/// D-PBC-26 / 17-CONTEXT §8. The reference route is the literal port of
/// `khf_ksymm.py:250-277` and is the **default** and the Gate C/D reference;
/// the fast route is validated against it at 1e-13, never against upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JkRoute {
    /// Unfold the IBZ density to the full BZ, call [`PeriodicDf::get_jk`] over
    /// all `nkpts`, fold the result back. The DF layer never learns about
    /// symmetry — it only ever sees a plain k-point list.
    #[default]
    Reference,
    /// Call [`PeriodicDf::get_jk`] at `kpts_ibz` only and unfold `vj`/`vk`
    /// with `transform_1e_operator`. Same contract, fewer DF k-points.
    Fast,
}

/// Restricted periodic Hartree-Fock over an irreducible k-point set.
///
/// `khf_ksymm.py:410` — upstream's `KsymAdaptedKRHF`.
#[derive(Debug)]
pub struct KsymAdaptedKrhf {
    /// The density-fitting object, built over the **full BZ**.
    ///
    /// `get_hcore`/`get_ovlp`/`get_jk` all take their k-points as an explicit
    /// argument (`fftdf.rs:447`, and `Krhf`'s own calls), so one full-BZ DF
    /// serves both routes: the one-electron hooks pass `kpts_ibz`, and the
    /// reference `get_veff` passes the full BZ.
    pub with_df: Box<dyn PeriodicDf>,
    /// The k-point symmetry. Held by composition (D-PBC-25).
    pub kpts: KPoints,
    /// Materialised so [`KOverrideHooks::kpts`] — which returns a borrowed
    /// slice — has something to borrow.
    kpts_ibz: Vec<[f64; 3]>,
    /// Exchange divergence treatment; upstream's default is Ewald.
    pub exxdiv: Option<ExxDiv>,
    /// `ksymm_scf_common_init` (`khf_ksymm.py:142`) defaults this to **true**:
    /// it is the DEFAULT branch, not an opt-in. `false` selects the plain
    /// `Krhf::eig` and exists so that any 17-04 defect is immediately
    /// bisectable.
    pub use_ao_symmetry: bool,
    /// Which `get_jk` route `get_veff` takes.
    pub jk_route: JkRoute,
}

impl KsymAdaptedKrhf {
    /// Build over an explicitly configured full-BZ density-fitting object.
    ///
    /// `kpts` must already be built (`KPoints::build`), and `with_df`'s
    /// k-points must be the full BZ that `kpts` folded.
    pub fn from_df(with_df: Box<dyn PeriodicDf>, kpts: KPoints) -> Self {
        let kpts_ibz = kpts.kpts_ibz.clone();
        Self {
            with_df,
            kpts,
            kpts_ibz,
            exxdiv: Some(ExxDiv::Ewald),
            use_ao_symmetry: true,
            jk_route: JkRoute::Reference,
        }
    }

    /// Electrons in the whole BZ supercell — `khf_ksymm.py:38`.
    ///
    /// **`kpts.nkpts()`, not `nkpts_ibz()`.** The Fermi level is a full-BZ
    /// quantity (17-CONTEXT §3.4); counting electrons over the IBZ would
    /// under-fill every star with more than one member.
    pub fn nelectron(&self) -> usize {
        self.cell().tot_electrons(self.kpts.nkpts())
    }

    /// The full-BZ k-points.
    pub fn kpts_bz(&self) -> &[[f64; 3]] {
        &self.kpts.kpts
    }

    /// Run the SCF over the IBZ.
    ///
    /// # Errors
    /// Propagates every hook and the driver.
    pub fn kernel(&self, cfg: &KScfConfig) -> Result<KScfResult, PyscfRsError> {
        kernel(self, cfg)
    }
}

// ---------------------------------------------------------------------
// CTensor <-> Complex64 seam.
//
// The SCF layer speaks planar `CTensor { re, im }` (RULE 8); `KPoints`'s
// transforms speak `Vec<Complex64>`. These two helpers are the only place
// the two representations meet, so a layout mistake has exactly one home.
// ---------------------------------------------------------------------

fn ctensor_to_complex(t: &CTensor) -> Vec<Complex64> {
    t.re.iter()
        .zip(t.im.iter())
        .map(|(&re, &im)| Complex64::new(re, im))
        .collect()
}

fn complex_to_ctensor(v: &[Complex64]) -> CTensor {
    CTensor {
        re: v.iter().map(|c| c.re).collect(),
        im: v.iter().map(|c| c.im).collect(),
    }
}

fn symm_err(e: pyscf_pbc_symm::PbcSymmError) -> PyscfRsError {
    PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
        "khf_ksymm: {e}"
    )))
}

fn df_err(e: pyscf_pbc_df::PbcDfError) -> PyscfRsError {
    PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
        "khf_ksymm: {e}"
    )))
}

fn missing(what: &str) -> PyscfRsError {
    PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
        "khf_ksymm: density fitting returned no {what}"
    )))
}

impl KsymAdaptedKrhf {
    /// 17-04's symmetry-adapted AO basis, as `khf_ksymm.py:104-119` reads it.
    ///
    /// Upstream reaches straight into `cell.symm_orb` / `cell.irrep_id`
    /// (`khf_ksymm.py:110-111`), which `Cell::build_symmetry` fills. A missing
    /// value is a **usage** error, not an internal one — it means
    /// `build_symmetry` was never called for this `KPoints` — so it says so
    /// and names the fix rather than unwrapping.
    ///
    /// # Errors
    /// When `symm_orb`/`irrep_id` are absent, or sized for a different k-set.
    fn symmetry_basis(&self) -> Result<(&[CTensor], &[Vec<i32>]), PyscfRsError> {
        let cell = self.cell();
        let need = |what: &str| {
            PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
                "khf_ksymm: use_ao_symmetry = true needs cell.{what}, which is unset. \
                 Call Cell::build_symmetry(&kpts) before the SCF, or set \
                 use_ao_symmetry = false to take the plain eig route."
            )))
        };
        let so = cell.symm_orb.as_deref().ok_or_else(|| need("symm_orb"))?;
        let ids = cell.irrep_id.as_deref().ok_or_else(|| need("irrep_id"))?;
        let n = self.kpts_ibz.len();
        if so.len() != n || ids.len() != n {
            return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
                format!(
                    "khf_ksymm: cell.symm_orb has {} entries and cell.irrep_id has {}, \
                     but this SCF runs over {n} IBZ k-points — they were built for a \
                     different k-set",
                    so.len(),
                    ids.len()
                ),
            )));
        }
        Ok((so, ids))
    }

    /// Unfold an IBZ-length set of `nao x nao` matrices to the full BZ.
    fn unfold_1e(&self, ibz: &KMats) -> Result<KMats, PyscfRsError> {
        let nao = self.nao();
        let as_c: Vec<Vec<Complex64>> = ibz.iter().map(ctensor_to_complex).collect();
        let bz = self
            .kpts
            .transform_1e_operator(self.cell(), &as_c, nao)
            .map_err(symm_err)?;
        Ok(bz.iter().map(|m| complex_to_ctensor(m)).collect())
    }

    /// Unfold an IBZ-length set of density matrices to the full BZ.
    fn unfold_dm(&self, ibz: &KMats) -> Result<KMats, PyscfRsError> {
        let nao = self.nao();
        let as_c: Vec<Vec<Complex64>> = ibz.iter().map(ctensor_to_complex).collect();
        let bz = self
            .kpts
            .transform_dm(self.cell(), &as_c, nao)
            .map_err(symm_err)?;
        Ok(bz.iter().map(|m| complex_to_ctensor(m)).collect())
    }

    /// Fold a full-BZ set of matrices back to the IBZ by selecting each IBZ
    /// point's own representative — `khf_ksymm.py:274-277`.
    ///
    /// This is a SELECTION, not an average: `ibz2bz[i]` is the BZ index of
    /// IBZ point `i`, so the representative is already the right matrix at
    /// the right k-point and averaging over the star would be wrong.
    fn fold_to_ibz(&self, bz: &KMats) -> KMats {
        self.kpts.ibz2bz.iter().map(|&k| bz[k].clone()).collect()
    }

    /// `vj - 0.5 * vk` — `khf.py:632`, in place on `vj`.
    fn combine_jk(vj: &mut KMats, vk: &KMats) {
        for (k, m) in vj.iter_mut().enumerate() {
            for i in 0..m.re.len() {
                m.re[i] -= 0.5 * vk[k].re[i];
                m.im[i] -= 0.5 * vk[k].im[i];
            }
        }
    }

    /// The reference `get_jk` route — `khf_ksymm.py:250-277`, and the Gate
    /// C/D reference.
    ///
    /// Unfold the IBZ density to the full BZ, call the ordinary
    /// [`PeriodicDf::get_jk`] over all `nkpts`, fold back. **The DF layer
    /// stays untouched** — this is the crux of D-PBC-15, and why 17-05 Task 6
    /// only made `KPoints` *nameable* from `pyscf-pbc-df` rather than changing
    /// it.
    fn veff_reference(&self, dms: &KDms) -> Result<KDms, PyscfRsError> {
        let dm_bz = self.unfold_dm(&dms[0])?;
        let r = self
            .with_df
            .get_jk(
                &vec![dm_bz],
                self.kpts_bz(),
                JkOpts {
                    hermi: 1,
                    kpts_band: None,
                    with_j: true,
                    with_k: true,
                    exxdiv: self.exxdiv,
                    omega: None,
                    kk_symmetry: JkOpts::kk_symmetry_default(),
                },
            )
            .map_err(df_err)?;
        let mut vj = r.vj.ok_or_else(|| missing("vj"))?.remove(0);
        let vk = r.vk.ok_or_else(|| missing("vk"))?.remove(0);
        Self::combine_jk(&mut vj, &vk);
        Ok(vec![self.fold_to_ibz(&vj)])
    }

    /// The fast `get_jk` route — D-PBC-26, 17-CONTEXT §8.
    ///
    /// Call [`PeriodicDf::get_jk`] at `kpts_ibz` only, then unfold `vj`/`vk`
    /// with `transform_1e_operator` and fold back exactly as the reference
    /// route does, so both present the same [`KOverrideHooks::get_veff`]
    /// contract. Still no symmetry inside the DF layer: it sees a plain
    /// k-point list that is merely shorter.
    ///
    /// **Validated against [`Self::veff_reference`] at 1e-13, never against
    /// upstream** — two routes to the same number inside one process is the
    /// stronger test (the same idiom as 17-10's MO-factorised `get_k_kpts`).
    fn veff_fast(&self, dms: &KDms) -> Result<KDms, PyscfRsError> {
        let r = self
            .with_df
            .get_jk(
                dms,
                &self.kpts_ibz,
                JkOpts {
                    hermi: 1,
                    kpts_band: None,
                    with_j: true,
                    with_k: true,
                    exxdiv: self.exxdiv,
                    omega: None,
                    kk_symmetry: JkOpts::kk_symmetry_default(),
                },
            )
            .map_err(df_err)?;
        let mut vj = r.vj.ok_or_else(|| missing("vj"))?.remove(0);
        let vk = r.vk.ok_or_else(|| missing("vk"))?.remove(0);
        Self::combine_jk(&mut vj, &vk);
        // Unfold to the BZ and fold back, so the returned IBZ matrices have
        // been through the same symmetrisation the reference route's have.
        let bz = self.unfold_1e(&vj)?;
        Ok(vec![self.fold_to_ibz(&bz)])
    }
}

impl KOverrideHooks for KsymAdaptedKrhf {
    fn cell(&self) -> &Cell {
        self.with_df.cell()
    }

    /// **The whole indirection.** Everything downstream — `nkpts`, the
    /// `idx(set, k)` layout, DIIS — then works on IBZ indices unchanged.
    fn kpts(&self) -> &[[f64; 3]] {
        &self.kpts_ibz
    }

    fn nset(&self) -> usize {
        1
    }

    fn get_ovlp(&self) -> Result<KMats, PyscfRsError> {
        let nao = self.nao();
        Ok(pyscf_pbc_scf_to_row_major(
            pyscf_pbc_gto::get_ovlp(self.cell(), self.kpts())?,
            nao,
        ))
    }

    fn get_hcore(&self) -> Result<KMats, PyscfRsError> {
        pyscf_pbc_df::get_hcore(self.with_df.as_ref(), self.kpts()).map_err(df_err)
    }

    fn get_init_guess(&self, mode: &KInitGuess, s1e: &KMats) -> Result<KDms, PyscfRsError> {
        crate::init_guess::get_init_guess(
            self.cell(),
            self.kpts_ibz.len(),
            1,
            mode,
            s1e,
            self.nelectron() as f64,
        )
    }

    fn get_veff(&self, dms: &KDms) -> Result<KDms, PyscfRsError> {
        match self.jk_route {
            JkRoute::Reference => self.veff_reference(dms),
            JkRoute::Fast => self.veff_fast(dms),
        }
    }

    fn eig(
        &self,
        fock: &KDms,
        s1e: &KMats,
    ) -> Result<(Vec<Vec<f64>>, Vec<CTensor>), PyscfRsError> {
        if !self.use_ao_symmetry {
            // `use_ao_symmetry = false` — the plain generalised eigenproblem
            // `Krhf` solves, one IBZ k-point at a time. Not a fallback for
            // lack of effort: having it makes any 17-04 defect immediately
            // bisectable against this branch.
            return crate::krhf::eig_channel(&fock[0], s1e, self.nao());
        }
        let (symm_orb, irrep_id) = self.symmetry_basis()?;
        let nao = self.nao();
        let mut es = Vec::with_capacity(fock[0].len());
        let mut cs = Vec::with_capacity(fock[0].len());
        for (k, f) in fock[0].iter().enumerate() {
            let (e, c) = eig_symm_adapted(f, &s1e[k], &symm_orb[k], &irrep_id[k], nao)
                .map_err(|err| {
                    PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
                        "khf_ksymm: symmetry-adapted eig failed at IBZ k = {k}: {err}"
                    )))
                })?;
            es.push(e);
            cs.push(c);
        }
        Ok((es, cs))
    }

    fn get_occ(
        &self,
        mo_energy: &[Vec<f64>],
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>), PyscfRsError> {
        // 17-CONTEXT §3.4, and the trait's own contract: ONE Fermi level over
        // the UNFOLDED BZ. Shared with the DFT ksymm adapters (17-08), which
        // upstream gets by inheritance instead.
        ksymm_get_occ_restricted(&self.kpts, mo_energy, self.nelectron())
    }

    fn make_rdm1(
        &self,
        mo_coeff: &[CTensor],
        mo_occ: &[Vec<f64>],
    ) -> Result<KDms, PyscfRsError> {
        Ok(vec![make_rdm1(mo_coeff, mo_occ, self.nao())])
    }

    fn energy_elec(
        &self,
        dms: &KDms,
        h1e: &KMats,
        vhf: &KDms,
    ) -> Result<(f64, f64), PyscfRsError> {
        // `khf_ksymm.py:69-86`. **`weights_ibz`, NOT `1/nkpts`** — see the
        // module doc's table. `weights_ibz` already sums to 1.
        let nao = self.nao();
        let w = &self.kpts.weights_ibz;
        let mut e1 = 0.0;
        let mut e_coul = 0.0;
        for k in 0..self.kpts_ibz.len() {
            e1 += w[k] * trace_prod(&dms[0][k], &h1e[k], nao);
            e_coul += w[k] * trace_prod(&dms[0][k], &vhf[0][k], nao);
        }
        Ok((e1 + 0.5 * e_coul, 0.5 * e_coul))
    }
}

/// `khf_ksymm.py:104-119` — solve the Fock matrix **one irrep block at a
/// time** in the symmetry-adapted AO basis.
///
/// `pub` because plan 17-08's `KsymAdaptedKrks` needs the identical routine:
/// upstream gets it by inheriting `khf_ksymm.KRHF`
/// (`class KsymAdaptedKRKS(krks.KRKS, khf_ksymm.KRHF)`, `krks_ksymm.py:88`),
/// and this port shares the function rather than copying it.
///
/// # Layouts — the 14-05 shape, so they are stated rather than assumed
///
/// * `fock`, `ovlp` — ROW-MAJOR `nao x nao`: element `(i, j)` at `i*nao + j`.
/// * `symm_orb` — COLUMN-MAJOR `nao x nao`: element `(i, p)` at `p*nao + i`.
///   (17-04 built it that way; `tests/basis.rs`'s `col_at` is the same
///   accessor.)
/// * the returned `mo_coeff` — COLUMN-MAJOR, per `KOverrideHooks::eig`.
///
/// Getting a transpose wrong here is exactly the class of defect that cost
/// plan 14-05 **+6 306 866.73 Ha** and was invisible to every gate then
/// existing, which is why 17-04 ships an independent block-diagonality test
/// (`symm_orbᴴ F symm_orb` block-diagonal by `irrep_id` to 1e-11) as this
/// function's precondition.
///
/// Why this is correct at all: `S` and `F` both commute with every operation
/// of the little co-group, so by Schur's lemma they have **no matrix elements
/// between distinct irreps**. Solving `F_ir c = S_ir c e` separately per irrep
/// is therefore exact, not an approximation — and it is cheaper, which is the
/// point.
pub fn eig_symm_adapted(
    fock: &CTensor,
    ovlp: &CTensor,
    symm_orb: &CTensor,
    irrep_id: &[i32],
    nao: usize,
) -> Result<(Vec<f64>, CTensor), pyscf_algebra::AlgebraError> {
    // Distinct irrep labels, in first-appearance order. NOT a HashSet: the
    // column order of `symm_orb` is load-bearing (17-04 Task 2), and a
    // nondeterministic iteration order here would permute `mo_coeff`.
    let mut labels: Vec<i32> = Vec::new();
    for &ir in irrep_id {
        if !labels.contains(&ir) {
            labels.push(ir);
        }
    }

    let col = |t: &CTensor, i: usize, p: usize| Complex64::new(t.re[p * nao + i], t.im[p * nao + i]);
    let row = |t: &CTensor, i: usize, j: usize| Complex64::new(t.re[i * nao + j], t.im[i * nao + j]);

    // (energy, full-length column) pairs, gathered across irreps and sorted
    // by energy at the end — upstream returns one ascending list per k-point.
    let mut solved: Vec<(f64, Vec<Complex64>)> = Vec::with_capacity(nao);

    for &ir in &labels {
        let cols: Vec<usize> = (0..nao).filter(|&p| irrep_id[p] == ir).collect();
        let n = cols.len();
        if n == 0 {
            continue;
        }

        // F_ir = Cᴴ F C and S_ir = Cᴴ S C, both ROW-MAJOR n x n, so they can
        // go straight into `zeigh_gen` alongside every other call site.
        let mut f_ir = CTensor::zeros(n * n);
        let mut s_ir = CTensor::zeros(n * n);
        for (a, &p) in cols.iter().enumerate() {
            for (b, &q) in cols.iter().enumerate() {
                let mut acc_f = Complex64::new(0.0, 0.0);
                let mut acc_s = Complex64::new(0.0, 0.0);
                for i in 0..nao {
                    let cip = col(symm_orb, i, p).conj();
                    for j in 0..nao {
                        let cjq = col(symm_orb, j, q);
                        acc_f += cip * row(fock, i, j) * cjq;
                        acc_s += cip * row(ovlp, i, j) * cjq;
                    }
                }
                f_ir.re[a * n + b] = acc_f.re;
                f_ir.im[a * n + b] = acc_f.im;
                s_ir.re[a * n + b] = acc_s.re;
                s_ir.im[a * n + b] = acc_s.im;
            }
        }

        // `zeigh_gen` returns eigenvectors COLUMN-MAJOR, same as everywhere.
        let (e_ir, c_ir) = pyscf_algebra::zeigh_gen(&f_ir, &s_ir, n)?;

        // Back-transform into the full AO basis: column `m` of this irrep's
        // solution is `sum_b C[:, cols[b]] * c_ir[b, m]`.
        for m in 0..n {
            let mut full = vec![Complex64::new(0.0, 0.0); nao];
            for (b, &q) in cols.iter().enumerate() {
                let coeff = Complex64::new(c_ir.re[m * n + b], c_ir.im[m * n + b]);
                if coeff == Complex64::new(0.0, 0.0) {
                    continue;
                }
                for (i, slot) in full.iter_mut().enumerate() {
                    *slot += col(symm_orb, i, q) * coeff;
                }
            }
            solved.push((e_ir[m], full));
        }
    }

    // One ascending list per k-point. `total_cmp` rather than `partial_cmp`
    // so the order is total and deterministic even if a NaN ever appears.
    solved.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut mo_energy = Vec::with_capacity(solved.len());
    let mut mo_coeff = CTensor::zeros(nao * solved.len());
    for (p, (e, vec_p)) in solved.iter().enumerate() {
        mo_energy.push(*e);
        for (i, v) in vec_p.iter().enumerate() {
            mo_coeff.re[p * nao + i] = v.re;
            mo_coeff.im[p * nao + i] = v.im;
        }
    }
    Ok((mo_energy, mo_coeff))
}

/// One Fermi level over the **unfolded** BZ, then folded back to the IBZ —
/// `khf_ksymm.py:31-67`, shared by the SCF and DFT ksymm adapters.
///
/// 17-CONTEXT §3.4: fusing the unfold into the IBZ loop gives a different
/// occupation on any cell with a k-dependent gap (risk R-06). `nelectron` is
/// the FULL-BZ electron count.
///
/// # Errors
/// Propagates `transform_mo_energy`, the occupation assignment, and
/// `check_mo_occ_symmetry` — the last of which reports a genuinely
/// symmetry-broken state, naming both k-points.
pub fn ksymm_get_occ_restricted(
    kpts: &KPoints,
    mo_energy_ibz: &[Vec<f64>],
    nelectron: usize,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), PyscfRsError> {
    let e_bz = kpts.transform_mo_energy(mo_energy_ibz).map_err(symm_err)?;
    let (occ_bz, fermi) = crate::kocc::get_occ_restricted(&e_bz, nelectron / 2)?;
    let occ_ibz = kpts
        .check_mo_occ_symmetry(&occ_bz, OCC_SYMMETRY_TOL)
        .map_err(symm_err)?;
    Ok((occ_ibz, vec![fermi]))
}

/// The two-channel [`ksymm_get_occ_restricted`] — `kuhf_ksymm.py`'s `get_occ`,
/// shared with plan 17-08's `KsymAdaptedKuks`.
///
/// `mo_energy` is channel-major: `nkpts_ibz` alpha entries followed by
/// `nkpts_ibz` beta ones, matching `Kuhf::get_occ`'s own layout.
///
/// **Each channel gets its own Fermi level, and each is computed over the
/// UNFOLDED BZ** (17-CONTEXT §3.4) — the same rule as the restricted case,
/// applied twice. `nelec` is the FULL-BZ `(nalpha, nbeta)` pair.
///
/// # Errors
/// As [`ksymm_get_occ_restricted`].
pub fn ksymm_get_occ_unrestricted(
    kpts: &KPoints,
    mo_energy: &[Vec<f64>],
    nelec: (usize, usize),
) -> Result<(Vec<Vec<f64>>, Vec<f64>), PyscfRsError> {
    let nibz = kpts.nkpts_ibz();
    if mo_energy.len() != 2 * nibz {
        return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            format!(
                "kuhf_ksymm: get_occ expects 2 * nkpts_ibz = {} channel-major \
                 entries, got {}",
                2 * nibz,
                mo_energy.len()
            ),
        )));
    }
    let (ea_ibz, eb_ibz) = mo_energy.split_at(nibz);
    let ea = kpts.transform_mo_energy(ea_ibz).map_err(symm_err)?;
    let eb = kpts.transform_mo_energy(eb_ibz).map_err(symm_err)?;
    let (occ_a_bz, occ_b_bz, fermi) =
        crate::kocc::get_occ_unrestricted(&ea, &eb, nelec.0, nelec.1)?;
    // Fold each channel back independently; a symmetry-broken solution in
    // either one is a real physical condition and is reported as such.
    let occ_a = kpts
        .check_mo_occ_symmetry(&occ_a_bz, OCC_SYMMETRY_TOL)
        .map_err(symm_err)?;
    let occ_b = kpts
        .check_mo_occ_symmetry(&occ_b_bz, OCC_SYMMETRY_TOL)
        .map_err(symm_err)?;
    let mut occ = occ_a;
    occ.extend(occ_b);
    Ok((occ, fermi.to_vec()))
}

/// `Re Tr(A B)` for row-major `nao x nao` complex `A`, `B`.
///
/// D-PBC-17: the accumulation is ordered, so the result is bit-identical
/// under any thread count.
fn trace_prod(a: &CTensor, b: &CTensor, nao: usize) -> f64 {
    let mut acc = 0.0;
    for i in 0..nao {
        for j in 0..nao {
            let aij = i * nao + j;
            let bji = j * nao + i;
            acc += a.re[aij] * b.re[bji] - a.im[aij] * b.im[bji];
        }
    }
    acc
}

/// Local alias for the F-order to row-major conversion `Krhf::get_ovlp` uses.
fn pyscf_pbc_scf_to_row_major(v: Vec<CTensor>, nao: usize) -> KMats {
    crate::krhf::to_row_major(v, nao)
}
