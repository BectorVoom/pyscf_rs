//! The periodic table — port of `pyscf/data/elements.py:18-36`.
//!
//! [`ELEMENTS`] is indexed by atomic number, so `ELEMENTS[63] == "Eu"` and
//! index 0 is the ghost placeholder `"X"`, exactly as upstream's list is laid
//! out. [`charge_for_symbol`] is the reverse direction (upstream `NUC` /
//! `ELEMENTS_PROTON`).
//!
//! This is the single symbol↔charge table for the workspace. It previously
//! existed as two independent Z≤36 stubs — one here and one in `pyscf-gto` —
//! which made every element above krypton a hard "unknown element symbol"
//! error and put lanthanides out of reach entirely.

/// Element symbols indexed by atomic number; `ELEMENTS[0]` is the ghost
/// placeholder. Mirrors `pyscf/data/elements.py` `ELEMENTS`.
pub static ELEMENTS: [&str; 119] = [
    "X", // ghost
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", //   1- 10
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", //  11- 20
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", //  21- 30
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", //  31- 40
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", //  41- 50
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", //  51- 60
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", //  61- 70
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", //  71- 80
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", //  81- 90
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", //  91-100
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", // 101-110
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", // 111-118
];

/// Nuclear charge for an element symbol, or `None` if it names no element.
///
/// Any trailing ghost/label suffix is stripped first (`"Cu1"` → `"Cu"`), and
/// the leading symbol is matched case-insensitively so `"EU"`, `"eu"` and
/// `"Eu"` all resolve. Ghost markers give `Some(0)`, matching upstream
/// `charge()` (`pyscf/data/elements.py:1136-1144`).
pub fn charge_for_symbol(symb: &str) -> Option<i32> {
    let alpha_end = symb
        .find(|c: char| !c.is_alphabetic())
        .unwrap_or(symb.len());
    let leading = &symb[..alpha_end];
    if leading.eq_ignore_ascii_case("ghost") || leading.eq_ignore_ascii_case("x") {
        return Some(0);
    }
    ELEMENTS
        .iter()
        .position(|e| e.eq_ignore_ascii_case(leading))
        .map(|z| z as i32)
}
