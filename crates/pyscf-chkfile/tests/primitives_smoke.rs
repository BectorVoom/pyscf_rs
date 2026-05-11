//! Round-trip smoke tests for pyscf-chkfile HDF5 primitives.
//!
//! Tests cover:
//! 1. UTF-8 multi-byte VL Unicode mol JSON round-trip (h5py-compat A2)
//! 2. Scalar f64 + 1D f64 round-trip (e_tot, mo_energy, mo_occ schema)
//! 3. F-order 2D round-trip (mo_coeff schema - Pitfall 8)
//! 4. Append-mode (open existing chkfile + add new group)
use ndarray::Array2;
use pyscf_chkfile::{
    open_for_read, open_for_write, read_dataset_1d, read_dataset_2d, read_mol, read_scalar_f64,
    write_dataset_1d, write_dataset_f_order, write_mol, write_scalar_f64,
};

#[test]
fn mol_json_round_trip_utf8() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let path = tmp.path();
    let mol_str = r#"{"atom": "氢 0 0 0", "basis": "sto-3g"}"#;
    {
        let f = open_for_write(path).expect("create");
        write_mol(&f, mol_str).expect("write_mol");
        f.flush().expect("flush");
    }
    {
        let f = open_for_read(path).expect("read");
        let s = read_mol(&f).expect("read_mol");
        assert_eq!(s, mol_str, "UTF-8 mol JSON must round-trip byte-for-byte");
    }
}

#[test]
fn scf_scalar_and_1d_round_trip() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let path = tmp.path();
    let mo_energy_in = vec![-20.55_f64, -1.34, -0.71, -0.57, 0.18, 0.85];
    let mo_occ_in = vec![2.0_f64, 2.0, 2.0, 2.0, 0.0, 0.0];
    {
        let f = open_for_write(path).expect("create");
        let scf = f.create_group("scf").expect("scf group");
        write_scalar_f64(&scf, "e_tot", -76.026123_f64).expect("e_tot");
        write_dataset_1d(&scf, "mo_energy", &mo_energy_in).expect("mo_energy");
        write_dataset_1d(&scf, "mo_occ", &mo_occ_in).expect("mo_occ");
        f.flush().expect("flush");
    }
    {
        let f = open_for_read(path).expect("read");
        let scf = f.group("scf").expect("scf group");
        let e_tot = read_scalar_f64(&scf, "e_tot").expect("e_tot");
        assert!((e_tot - (-76.026123_f64)).abs() < 1e-12);
        let mo_e = read_dataset_1d(&scf, "mo_energy").expect("mo_energy");
        assert_eq!(mo_e.len(), 6);
        for (a, b) in mo_e.iter().zip(&mo_energy_in) {
            assert!((a - b).abs() < 1e-12);
        }
        let mo_o = read_dataset_1d(&scf, "mo_occ").expect("mo_occ");
        for (a, b) in mo_o.iter().zip(&mo_occ_in) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}

#[test]
fn mo_coeff_f_order_round_trip() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let path = tmp.path();
    // Build a 3x3 matrix where C-order and F-order differ visibly.
    // c_order[(i, j)] = original 1..9 row-major.
    let c_order: Array2<f64> = Array2::from_shape_vec(
        (3, 3),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .expect("c");
    {
        let f = open_for_write(path).expect("create");
        let scf = f.create_group("scf").expect("scf group");
        write_dataset_f_order(&scf, "mo_coeff", c_order.view()).expect("write F-order");
        f.flush().expect("flush");
    }
    {
        let f = open_for_read(path).expect("read");
        let scf = f.group("scf").expect("scf group");
        let mat = read_dataset_2d(&scf, "mo_coeff").expect("read");
        // hdf5-metno's `write` requires C-contiguous (standard) layout in memory.
        // Our `write_dataset_f_order` writes the TRANSPOSE so that the on-disk
        // layout corresponds to F-order column-major bytes (h5py reading the
        // raw bytes with column-major interpretation recovers the original).
        // Therefore reading back via `read_2d` (which interprets as C-order)
        // returns the transpose of the original.
        assert_eq!(mat.shape(), &[3, 3]);
        assert!((mat[(0, 0)] - 1.0).abs() < 1e-12);
        // transposed: mat[(0, 1)] == c_order[(1, 0)] == 4.0
        assert!((mat[(0, 1)] - 4.0).abs() < 1e-12);
        assert!((mat[(1, 0)] - 2.0).abs() < 1e-12);
        assert!((mat[(2, 1)] - 6.0).abs() < 1e-12);
    }
}

#[test]
fn append_mode_preserves_existing_data() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let path = tmp.path();
    // First write
    {
        let f = open_for_write(path).expect("create");
        write_mol(&f, r#"{"atom":"H 0 0 0"}"#).expect("write_mol");
        let scf = f.create_group("scf").expect("scf group");
        write_scalar_f64(&scf, "e_tot", -1.117_f64).expect("e_tot");
        f.flush().expect("flush");
    }
    // Reopen for append + add second group
    {
        let f = open_for_write(path).expect("append");
        // /mol still readable
        let s = read_mol(&f).expect("read_mol after append");
        assert!(s.contains("H 0 0 0"));
        // Add /extra/foo dataset
        let extra = f.create_group("extra").expect("extra group");
        write_scalar_f64(&extra, "foo", 3.14_f64).expect("foo");
        f.flush().expect("flush");
    }
    // Final read: both /scf/e_tot AND /extra/foo present
    {
        let f = open_for_read(path).expect("read");
        let scf = f.group("scf").expect("scf group");
        assert!((read_scalar_f64(&scf, "e_tot").expect("e_tot") - (-1.117)).abs() < 1e-12);
        let extra = f.group("extra").expect("extra group");
        assert!((read_scalar_f64(&extra, "foo").expect("foo") - 3.14).abs() < 1e-12);
    }
}
