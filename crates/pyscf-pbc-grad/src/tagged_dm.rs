//! Orbital provenance for one spin channel of a periodic density matrix.
//!
//! The exchange derivative can use occupied orbitals in place of the full AO
//! ket index. Private storage keeps the orbital tag consistent with the dense
//! density: tagged inputs are constructed through the SCF density builder.

use pyscf_algebra::CTensor;
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_scf::{KMats, krdm::make_rdm1};

#[derive(Debug, Clone)]
pub struct TaggedDm {
    nao: usize,
    matrices: KMats,
    orbitals: Option<(Vec<CTensor>, Vec<Vec<f64>>)>,
}

impl TaggedDm {
    /// Dense row-major matrices without orbital provenance.
    pub fn dense(nao: usize, matrices: KMats) -> Result<Self, PyscfRsError> {
        let n2 = nao
            .checked_mul(nao)
            .ok_or_else(|| invalid("density shape overflow"))?;
        if matrices.is_empty() || matrices.iter().any(|m| !valid_matrix(m, n2)) {
            return Err(invalid(
                "density requires nonempty k-points and finite nao x nao planes",
            ));
        }
        Ok(Self {
            nao,
            matrices,
            orbitals: None,
        })
    }

    /// Build density and tag together. Coefficients are column-major
    /// `nao x nmo`, and occupations may be fractional or zero.
    pub fn from_orbitals(
        nao: usize,
        mo_coeff: Vec<CTensor>,
        mo_occ: Vec<Vec<f64>>,
    ) -> Result<Self, PyscfRsError> {
        if mo_coeff.is_empty() || mo_coeff.len() != mo_occ.len() {
            return Err(invalid(
                "orbital and occupation k-point counts must match and be nonzero",
            ));
        }
        for (c, occ) in mo_coeff.iter().zip(&mo_occ) {
            let n = nao
                .checked_mul(occ.len())
                .ok_or_else(|| invalid("orbital shape overflow"))?;
            if !valid_matrix(c, n) || occ.iter().any(|o| !o.is_finite() || *o < 0.0) {
                return Err(invalid("invalid orbital planes or occupations"));
            }
        }
        let mut out = Self::dense(nao, make_rdm1(&mo_coeff, &mo_occ, nao))?;
        out.orbitals = Some((mo_coeff, mo_occ));
        Ok(out)
    }

    pub fn nao(&self) -> usize {
        self.nao
    }
    pub fn matrices(&self) -> &[CTensor] {
        &self.matrices
    }
    pub fn orbitals(&self) -> Option<(&[CTensor], &[Vec<f64>])> {
        self.orbitals
            .as_ref()
            .map(|(c, o)| (c.as_slice(), o.as_slice()))
    }
}

fn valid_matrix(m: &CTensor, len: usize) -> bool {
    m.re.len() == len && m.im.len() == len && m.re.iter().chain(&m.im).all(|v| v.is_finite())
}

fn invalid(message: &str) -> PyscfRsError {
    CoreError::InvalidMolecule(format!("tagged periodic density: {message}")).into()
}
