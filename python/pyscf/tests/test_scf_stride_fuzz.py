"""Phase 3 BIND-04 stride-fuzz oracle (NumPy zero-copy boundary discipline).

Covers: BIND-04 — `a` / `a.T` / `a[::2]` / `a[:, 1:5]` all return identical
answers (Pitfall 5 mitigation: `is_c_contiguous` → `to_owned` policy).

Source: RESEARCH §"Pattern 4 stride-fuzz test" lines 608-635 (verbatim
port; the only deviation is the test fixture uses h2o_sto3g_mol from conftest).

Why bit-identical (not allclose): plan 03-07's `numpy_io::to_density`
runs `is_c_contiguous()` → `to_owned()` fallback, so every input is
materialized as a C-contiguous f64 buffer BEFORE the Rust kernel sees
it. The 4 views below all carry the SAME underlying data (a symmetric
density matrix), so the canonical C-buffer is identical across views,
hence the Rust output is bit-identical.

Note: dm.T transposes the data, but for a SYMMETRIC density matrix
(D = D.T), the transposed buffer equals the original; the test asserts
the input is symmetric (which valid density matrices are) before using
.T as a stride variant.
"""
import numpy as np

from pyscf import scf


def test_get_veff_accepts_4_stride_variants(h2o_sto3g_mol):
    """`get_veff(mol, dm)` returns bit-identical results across 4 stride views."""
    mol = h2o_sto3g_mol
    mf = scf.RHF(mol).run()
    assert mf.converged

    # Use the converged 1-RDM as the test density (guaranteed symmetric).
    nao = int(mol.nao_nr())
    dm = np.ascontiguousarray(np.asarray(mf.make_rdm1(mf.mo_coeff, mf.mo_occ)))
    # Density matrices ARE symmetric (P = P.T) for SCF — assert as fixture
    # guard so the test is self-checking.
    np.testing.assert_allclose(dm, dm.T, atol=1e-12, rtol=0,
                                err_msg="test fixture: density must be symmetric")

    # Build 4 stride-equivalent views of the SAME data:
    view_a = dm                                                   # C-contig
    view_b = dm.T                                                 # F-contig (transposed; symmetric data)
    # view_c: strided via odd-step slicing into a 2× upsampled buffer.
    bigger = np.zeros((nao * 2, nao * 2))
    bigger[::2, ::2] = dm
    view_c = bigger[::2, ::2]
    # view_d: offset C-strided view into a wider buffer.
    wider = np.zeros((nao, nao + 5))
    wider[:, 1 : nao + 1] = dm
    view_d = wider[:, 1 : nao + 1]

    # All 4 views must produce the same get_veff (BIND-04 bit-identical contract).
    out_a = np.asarray(mf.get_veff(mol, view_a))
    out_b = np.asarray(mf.get_veff(mol, view_b))
    out_c = np.asarray(mf.get_veff(mol, view_c))
    out_d = np.asarray(mf.get_veff(mol, view_d))

    # Bit-identical: NumPy testing assert_array_equal (NOT allclose).
    np.testing.assert_array_equal(
        out_a, out_b,
        err_msg="BIND-04: view_b (dm.T) produced different bytes from view_a (C-contig)",
    )
    np.testing.assert_array_equal(
        out_a, out_c,
        err_msg="BIND-04: view_c (dm[::2,::2]) produced different bytes from view_a",
    )
    np.testing.assert_array_equal(
        out_a, out_d,
        err_msg="BIND-04: view_d (dm[:,1:nao+1] offset) produced different bytes from view_a",
    )


def test_stride_fuzz_get_veff_accepts_views(h2o_sto3g_mol):
    """Aggregator name kept for grep continuity (plan 03-02 stub name)."""
    test_get_veff_accepts_4_stride_variants(h2o_sto3g_mol)
