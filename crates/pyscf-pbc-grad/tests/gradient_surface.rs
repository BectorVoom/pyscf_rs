use pyscf_pbc_grad::Gradients;
use pyscf_pbc_gto::{Cell, pbc_intor, test_systems};

struct Surface {
    cell: Cell,
    kpts: Vec<[f64; 3]>,
}
impl Gradients for Surface {
    fn cell(&self) -> &Cell {
        &self.cell
    }
    fn kpts(&self) -> &[[f64; 3]] {
        &self.kpts
    }
}

#[test]
fn overlap_has_component_k_layout_and_gradient_sign() {
    let surface = Surface {
        cell: test_systems::diamond(),
        kpts: vec![[0.0; 3], [0.13, -0.07, 0.11]],
    };
    let raw = pbc_intor(
        &surface.cell,
        "int1e_ipovlp",
        &surface.kpts,
        Default::default(),
    )
    .unwrap();
    let derivative = surface.get_ovlp().unwrap();
    let n = raw.ni * raw.nj;
    let mut nonzero = false;
    for c in 0..3 {
        for k in 0..2 {
            assert_eq!(derivative[c].len(), 2);
            assert_eq!(derivative[c][k].re.len(), n);
            for i in 0..n {
                nonzero |= raw.kmats[k].re[c * n + i].abs() > 1e-8;
                assert_eq!(derivative[c][k].re[i], -raw.kmats[k].re[c * n + i]);
                assert_eq!(derivative[c][k].im[i], -raw.kmats[k].im[c * n + i]);
            }
        }
    }
    assert!(nonzero, "sign test must not be vacuous");
    assert_eq!(
        surface.grad_nuc().unwrap(),
        pyscf_pbc_gto::ewald_nuc_grad(&surface.cell, None, None).unwrap()
    );
}
