//! Per-intor F/C-order layout table (W0-T3, RESEARCH Pitfall 1 / Pitfall 8).
//!
//! Source of truth: `pyscf/gto/moleintor.py:288+` `_INTOR_FUNCTIONS` dict.
//! Upstream PySCF returns every intor as `numpy.ndarray(shape, order='F',
//! buffer=...)` — F-order is the default for all 2c/4c integrals at the
//! `getints2c` / `getints4c` call sites (`moleintor.py` lines 475+, 603+).
//! Component-axis-leading intors (e.g., `int1e_ipovlp`'s 3 derivative
//! components) ship the component axis as axis 0 and the inner AO axes are
//! still F-order.
//!
//! The intor dispatcher in plan 02-05 consults this table to translate
//! cintx `IntegralTensor.owned_values` + `extents` into the upstream layout
//! before returning to the caller. Every name listed here MUST also appear
//! in `cintx_compat::raw::RawApiId` (we won't dispatch a name cintx can't
//! handle); plan 02-05 enforces this with a startup-time consistency
//! check.

/// Output layout convention for an intor name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntorLayout {
    /// Single-component, F-order. Most 1e/2e integrals.
    ScalarFOrder,
    /// Multi-component (e.g., 3 for ∇x/∇y/∇z), component axis is leading
    /// (axis 0). Inner AO axes are F-order.
    ComponentLeadingFOrder { components: u8 },
}

/// Per-intor entry. `name` is the cintx symbol (post-suffix); the `comp`
/// quantity from upstream `_INTOR_FUNCTIONS[name][0]` is the
/// `components` field of `IntorLayout::ComponentLeadingFOrder`. The
/// `hermiticity` bit (upstream's tuple element [1]) is not part of the
/// layout decision and is therefore not stored here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntorEntry {
    pub name: &'static str,
    pub layout: IntorLayout,
}

/// In-scope intor catalogue with output layout. Sorted by family then by name.
/// EVERY name here MUST be supported by cintx (verified by Wave 2 02-05
/// against `cintx_compat::raw::RawApiId`).
///
/// Phase 2 ships ≥ 22 entries covering SCF/DFT/MP2/CCSD/grad needs.
/// Phase 7 (gradients) extends with derivative families if any are missing.
pub const INTOR_LAYOUTS: &[IntorEntry] = &[
    // ── 1e overlap / kinetic / nuclear-attraction (1-component, F-order) ──
    IntorEntry {
        name: "int1e_ovlp_sph",
        layout: IntorLayout::ScalarFOrder,
    },
    IntorEntry {
        name: "int1e_ovlp_cart",
        layout: IntorLayout::ScalarFOrder,
    },
    IntorEntry {
        name: "int1e_kin_sph",
        layout: IntorLayout::ScalarFOrder,
    },
    IntorEntry {
        name: "int1e_kin_cart",
        layout: IntorLayout::ScalarFOrder,
    },
    IntorEntry {
        name: "int1e_nuc_sph",
        layout: IntorLayout::ScalarFOrder,
    },
    IntorEntry {
        name: "int1e_nuc_cart",
        layout: IntorLayout::ScalarFOrder,
    },
    // ── 1e dipole (3-component, leading) — Phase 4/5 properties ──────────
    IntorEntry {
        name: "int1e_r_sph",
        layout: IntorLayout::ComponentLeadingFOrder { components: 3 },
    },
    IntorEntry {
        name: "int1e_r_cart",
        layout: IntorLayout::ComponentLeadingFOrder { components: 3 },
    },
    // ── 1e gradient families (3-component, leading) — SCF analytic Fock
    //    derivative + Phase 7 gradients ─────────────────────────────────
    IntorEntry {
        name: "int1e_ipovlp_sph",
        layout: IntorLayout::ComponentLeadingFOrder { components: 3 },
    },
    IntorEntry {
        name: "int1e_ipkin_sph",
        layout: IntorLayout::ComponentLeadingFOrder { components: 3 },
    },
    IntorEntry {
        name: "int1e_ipnuc_sph",
        layout: IntorLayout::ComponentLeadingFOrder { components: 3 },
    },
    IntorEntry {
        name: "int1e_iprinv_sph",
        layout: IntorLayout::ComponentLeadingFOrder { components: 3 },
    },
    // ── 2e Coulomb (1-component, F-order, 4D) ─────────────────────────────
    IntorEntry {
        name: "int2e_sph",
        layout: IntorLayout::ScalarFOrder,
    },
    IntorEntry {
        name: "int2e_cart",
        layout: IntorLayout::ScalarFOrder,
    },
    // ── 2e gradient (3-component, leading) — Phase 7 ──────────────────────
    IntorEntry {
        name: "int2e_ip1_sph",
        layout: IntorLayout::ComponentLeadingFOrder { components: 3 },
    },
    IntorEntry {
        name: "int2e_ip2_sph",
        layout: IntorLayout::ComponentLeadingFOrder { components: 3 },
    },
    // ── 3-center (DF / density-fitting auxiliary basis) ───────────────────
    IntorEntry {
        name: "int3c2e_sph",
        layout: IntorLayout::ScalarFOrder,
    },
    IntorEntry {
        name: "int3c2e_cart",
        layout: IntorLayout::ScalarFOrder,
    },
    IntorEntry {
        name: "int3c2e_ip1_sph",
        layout: IntorLayout::ComponentLeadingFOrder { components: 3 },
    },
    // ── moment-weighted 1e / 3-center families (GTH pseudopotentials,
    //    PBC Phase 10 plans 10-05 / 10-06) ─────────────────────────────────
    //
    // `<i| r^n |j>` with the origin ON CENTRE i (`origi`) and the 3-centre
    // `origk` analogue with the origin on the auxiliary centre. Consumed by
    // `pp_int.py:626` `_int_vnl` (`origi`) and `pp_int.py:150`
    // `get_pp_loc_part2` (`origk`). cintx ships them behind
    // `unstable-source-api` — see `pyscf-pbc-gto`'s `gth-pp` feature.
    IntorEntry {
        name: "int1e_r2_origi_sph",
        layout: IntorLayout::ScalarFOrder,
    },
    IntorEntry {
        name: "int1e_r4_origi_sph",
        layout: IntorLayout::ScalarFOrder,
    },
    IntorEntry {
        name: "int3c1e_sph",
        layout: IntorLayout::ScalarFOrder,
    },
    IntorEntry {
        name: "int3c1e_r2_origk_sph",
        layout: IntorLayout::ScalarFOrder,
    },
    IntorEntry {
        name: "int3c1e_r4_origk_sph",
        layout: IntorLayout::ScalarFOrder,
    },
    IntorEntry {
        name: "int3c1e_r6_origk_sph",
        layout: IntorLayout::ScalarFOrder,
    },
    // ── 2-center auxiliary (DF metric) ────────────────────────────────────
    IntorEntry {
        name: "int2c2e_sph",
        layout: IntorLayout::ScalarFOrder,
    },
    IntorEntry {
        name: "int2c2e_cart",
        layout: IntorLayout::ScalarFOrder,
    },
    // ── grids (DFT NumInt path; Phase 4) ──────────────────────────────────
    IntorEntry {
        name: "int1e_grids_sph",
        layout: IntorLayout::ScalarFOrder,
    },
];

/// Lookup by name (linear scan; fast enough for ~25 entries).
///
/// Returns `None` for any name not in the in-scope catalogue. The 02-05
/// dispatcher uses this lookup as a feature gate: unknown names error
/// out with `IntorNotInScope { name }` rather than passing through to
/// cintx (so an upstream rename gets caught here, not deep in libcint).
pub fn lookup(name: &str) -> Option<IntorLayout> {
    INTOR_LAYOUTS
        .iter()
        .find(|e| e.name == name)
        .map(|e| e.layout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ovlp_is_scalar_f_order() {
        assert_eq!(lookup("int1e_ovlp_sph"), Some(IntorLayout::ScalarFOrder));
    }

    #[test]
    fn ipovlp_is_component_leading_3() {
        assert_eq!(
            lookup("int1e_ipovlp_sph"),
            Some(IntorLayout::ComponentLeadingFOrder { components: 3 }),
        );
    }

    #[test]
    fn unknown_returns_none() {
        assert_eq!(lookup("int1e_bogus_sph"), None);
    }

    #[test]
    fn every_entry_has_a_known_suffix() {
        for entry in INTOR_LAYOUTS {
            assert!(
                entry.name.ends_with("_sph") || entry.name.ends_with("_cart"),
                "intor entries must carry _sph or _cart suffix: {}",
                entry.name
            );
        }
    }

    #[test]
    fn catalogue_meets_phase_2_floor() {
        assert!(
            INTOR_LAYOUTS.len() >= 20,
            "Phase 2 requires ≥ 20 INTOR_LAYOUTS entries (have {})",
            INTOR_LAYOUTS.len()
        );
    }
}
