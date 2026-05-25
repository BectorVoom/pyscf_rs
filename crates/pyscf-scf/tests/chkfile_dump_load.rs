//! Rust ↔ Rust round-trip: write ScfResult → read → compare.
//!
//! Cross-language ORACLE-08 (h5py reads pyscf-rs-written chkfile, and vice
//! versa) is plan 03-08 — this test ships the Rust-only seal that proves
//! the schema + F-order convention round-trip inside pyscf-rs itself.
use pyscf_chkfile::{Checkpointable, primitives};
use pyscf_core::{Energy, MOCoefficients};
use pyscf_scf::{
    ScfResult,
    chkfile::{dump_scf_to_file, load_scf_from_file},
};

fn sample_result() -> ScfResult {
    let nao = 4;
    let nmo = 4;
    // F-order flat data: data[i + j*nao] for (i, j) entry.
    let mut data = vec![0.0_f64; nao * nmo];
    for j in 0..nmo {
        for i in 0..nao {
            data[i + j * nao] = ((i * 3 + j + 1) as f64) * 0.1;
        }
    }
    ScfResult {
        e_tot: Energy(-76.026123_f64),
        mo_coeff: MOCoefficients {
            nao,
            nmo,
            data: data.clone(),
            energies: vec![-20.55, -1.34, -0.71, -0.57],
            occupations: vec![2.0, 2.0, 2.0, 2.0],
        },
        mo_energy: vec![-20.55, -1.34, -0.71, -0.57],
        mo_occ: vec![2.0, 2.0, 2.0, 2.0],
        converged: true,
        cycles: 12,
    }
}

#[test]
fn rust_rust_round_trip() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let path = tmp.path();
    let original = sample_result();
    dump_scf_to_file(path, r#"{"atom":"H 0 0 0","basis":"sto-3g"}"#, &original).expect("dump");
    let loaded = load_scf_from_file(path).expect("load");
    // e_tot
    assert!((loaded.e_tot.0 - original.e_tot.0).abs() < 1e-12);
    // mo_energy
    assert_eq!(loaded.mo_energy.len(), original.mo_energy.len());
    for (a, b) in loaded.mo_energy.iter().zip(&original.mo_energy) {
        assert!((a - b).abs() < 1e-12);
    }
    // mo_occ
    assert_eq!(loaded.mo_occ.len(), original.mo_occ.len());
    for (a, b) in loaded.mo_occ.iter().zip(&original.mo_occ) {
        assert!((a - b).abs() < 1e-12);
    }
    // mo_coeff: F-order data must match element-wise.
    assert_eq!(loaded.mo_coeff.nao, original.mo_coeff.nao);
    assert_eq!(loaded.mo_coeff.nmo, original.mo_coeff.nmo);
    for (k, (a, b)) in loaded
        .mo_coeff
        .data
        .iter()
        .zip(&original.mo_coeff.data)
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-12,
            "mo_coeff element {} mismatch: {} vs {}",
            k,
            a,
            b
        );
    }
}

#[test]
fn schema_keys_match_upstream() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let path = tmp.path();
    dump_scf_to_file(path, "{}", &sample_result()).expect("dump");
    let f = primitives::open_for_read(path).expect("reopen");
    // Top level: /mol must exist
    assert!(f.link_exists("mol"), "/mol missing");
    // Sub-group /scf with 4 keys must exist
    let scf = f.group("scf").expect("scf group");
    assert!(scf.link_exists("e_tot"), "/scf/e_tot missing");
    assert!(scf.link_exists("mo_energy"), "/scf/mo_energy missing");
    assert!(scf.link_exists("mo_occ"), "/scf/mo_occ missing");
    assert!(scf.link_exists("mo_coeff"), "/scf/mo_coeff missing");
}

#[test]
fn checkpointable_trait_dump_load_directly() {
    // Exercise the Checkpointable::dump/load directly under a custom group name.
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let path = tmp.path();
    let original = sample_result();
    {
        let f = primitives::open_for_write(path).expect("create");
        primitives::write_mol(&f, "{}").expect("mol");
        let g = f.create_group("scf").expect("group");
        original.dump(&g).expect("dump");
        f.flush().expect("flush");
    }
    {
        let f = primitives::open_for_read(path).expect("read");
        let g = f.group("scf").expect("group");
        let loaded = ScfResult::load(&g).expect("load");
        assert!((loaded.e_tot.0 - original.e_tot.0).abs() < 1e-12);
        assert_eq!(loaded.mo_coeff.nao, original.mo_coeff.nao);
        assert_eq!(loaded.mo_coeff.nmo, original.mo_coeff.nmo);
    }
}
