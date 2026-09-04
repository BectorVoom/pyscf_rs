//! M-09 step 1: the parallel pass-2 traversal must be bit-identical to the
//! former serial `(ci, cj)` traversal.  This is deliberately a separate test
//! file: production sources contain no embedded test module.

use pyscf_algebra::oracle_sum;
use pyscf_pbc_dft::multigrid::colloc::{LevelValues, level_pass2};
use pyscf_pbc_dft::multigrid::tasks::{Decontracted, Pshell};

fn pshell(cart_ao0: usize) -> Pshell {
    Pshell {
        orig_bas: cart_ao0,
        l: 0,
        center: [cart_ao0 as f64, 0.0, 0.0],
        alpha: 1.0,
        rcut: 20.0,
        ke_cutoff: 1.0,
        cart_ao0,
        coef: 1.0,
    }
}

fn serial_reference(lv: &LevelValues, decon: &Decontracted, weight: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; decon.nao_p * decon.nao_p];
    let mut buf = vec![0.0; lv.ngrids];
    let mut add_block = |i_range: std::ops::Range<usize>, j_range: std::ops::Range<usize>| {
        for i in i_range.clone() {
            let pi = lv.ids[i];
            let (si0, si1) = (lv.slot0[i], lv.slot0[i + 1]);
            let ci0 = decon.pshells[pi].cart_ao0;
            for j in j_range.clone() {
                let pj = lv.ids[j];
                let (sj0, sj1) = (lv.slot0[j], lv.slot0[j + 1]);
                let cj0 = decon.pshells[pj].cart_ao0;
                for (si, ci) in (si0..si1).zip(ci0..ci0 + (si1 - si0)) {
                    for (sj, cj) in (sj0..sj1).zip(cj0..cj0 + (sj1 - sj0)) {
                        for g in 0..lv.ngrids {
                            buf[g] = weight[g]
                                * lv.values[si * lv.ngrids + g]
                                * lv.values[sj * lv.ngrids + g];
                        }
                        out[ci * decon.nao_p + cj] += oracle_sum(&buf);
                    }
                }
            }
        }
    };
    add_block(0..lv.dense_count, 0..lv.ids.len());
    add_block(lv.dense_count..lv.ids.len(), 0..lv.dense_count);
    out
}

#[test]
fn pass2_parallel_matches_serial_bit_exact() {
    let ngrids = 257;
    let mut values = vec![0.0; 2 * ngrids];
    let mut weight = vec![0.0; ngrids];
    for g in 0..ngrids {
        let x = g as f64;
        values[g] = (x * 0.017).sin() * 0.75 + 0.125;
        values[ngrids + g] = (x * 0.031).cos() * 0.5 - 0.25;
        weight[g] = (x * 0.013).sin() * 0.2 + 1.0;
    }
    let lv = LevelValues {
        ids: vec![0, 1],
        dense_count: 1,
        slot0: vec![0, 1, 2],
        values,
        ngrids,
        mesh: [257, 1, 1],
        coords: Vec::new(),
        pshell_rec0: Vec::new(),
        pshell_nrec: Vec::new(),
        rec_center: Vec::new(),
        rcut: Vec::new(),
    };
    let decon = Decontracted {
        pshells: vec![pshell(0), pshell(1)],
        nao_p: 2,
        nao: 2,
        expand: vec![1.0, 0.0, 0.0, 1.0],
    };

    let expected = serial_reference(&lv, &decon, &weight);
    let mut actual = vec![0.0; 4];
    level_pass2(&lv, &decon, &weight, &mut actual);

    for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "pass2 entry {i}: {a} != {b}");
    }
}
