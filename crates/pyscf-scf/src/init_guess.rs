//! 5 init_guess modes — declarations.
//! Plan 03-11 ships the '1e' body; minao/atom/huckel stay NotYetImplemented
//! until a Phase 3 follow-up plan or Phase 4 dependency lands.
//! chkfile mode delegated to plan 03-06.
use crate::{error::ScfError, InitGuessMode};
use pyscf_core::{Density, Mole, PyscfRsError};

pub fn default_get_init_guess(
    mol: &Mole,
    mode: &InitGuessMode,
) -> Result<Density, PyscfRsError> {
    match mode {
        InitGuessMode::Minao => {
            Err(ScfError::InitGuessNotYetImplemented("minao", "03-03 follow-up").into())
        }
        InitGuessMode::Atom => {
            Err(ScfError::InitGuessNotYetImplemented("atom", "03-03 follow-up").into())
        }
        InitGuessMode::OneElectron => init_guess_by_1e(mol),
        InitGuessMode::Huckel => {
            Err(ScfError::InitGuessNotYetImplemented("huckel", "03-03 follow-up").into())
        }
        InitGuessMode::Chkfile(_) => {
            Err(ScfError::InitGuessNotYetImplemented("chkfile", "03-06").into())
        }
        InitGuessMode::UserDM(d) => Ok(d.clone()),
    }
}

pub(crate) fn init_guess_by_1e(_mol: &Mole) -> Result<Density, PyscfRsError> {
    unimplemented!("plan 03-11 — diagonalize h_core, Aufbau-fill, make_rdm1")
}

/// Parse a string mode name (used by oracle Arm 4).
pub fn parse_init_guess_mode(name: &str) -> Result<InitGuessMode, PyscfRsError> {
    match name {
        "minao" => Ok(InitGuessMode::Minao),
        "atom" => Ok(InitGuessMode::Atom),
        "1e" => Ok(InitGuessMode::OneElectron),
        "huckel" => Ok(InitGuessMode::Huckel),
        other => Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            format!("unknown init_guess mode '{}'", other),
        ))),
    }
}
