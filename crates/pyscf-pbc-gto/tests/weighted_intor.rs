use pyscf_pbc_gto::{
    PbcIntorOpts, intor_cross_with_image_weights, intor_cross_with_images, test_systems::he_fcc,
};

#[test]
fn image_weights_preserve_and_select_images() {
    let cell = he_fcc();
    let ls = [[0.0; 3], cell.a[0], cell.a[0].map(|x| -x)];
    let kpts = [[0.13, -0.07, 0.21], [-0.04, 0.11, 0.06]];
    let opts = PbcIntorOpts {
        hermi: 0,
        ..Default::default()
    };
    let plain = intor_cross_with_images("int1e_ovlp", &cell, &cell, &kpts, opts.clone(), &ls, None)
        .unwrap();
    let ones = intor_cross_with_image_weights(
        "int1e_ovlp",
        &cell,
        &cell,
        &kpts,
        opts.clone(),
        &ls,
        None,
        Some(&[1.0; 3]),
    )
    .unwrap();
    let weights = [0.3, -0.7, 1.2];
    let weighted = intor_cross_with_image_weights(
        "int1e_ovlp",
        &cell,
        &cell,
        &kpts,
        opts.clone(),
        &ls,
        None,
        Some(&weights),
    )
    .unwrap();
    let singles: Vec<_> = ls
        .iter()
        .map(|l| {
            intor_cross_with_images(
                "int1e_ovlp",
                &cell,
                &cell,
                &kpts,
                opts.clone(),
                std::slice::from_ref(l),
                None,
            )
            .unwrap()
        })
        .collect();
    for k in 0..kpts.len() {
        let a = plain.at(k);
        let b = ones.at(k);
        assert_eq!(
            a.re.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.re.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            a.im.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.im.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        for i in 0..a.re.len() {
            let re: f64 = (0..3).map(|m| weights[m] * singles[m].at(k).re[i]).sum();
            let im: f64 = (0..3).map(|m| weights[m] * singles[m].at(k).im[i]).sum();
            assert!((weighted.at(k).re[i] - re).abs() < 1e-13);
            assert!((weighted.at(k).im[i] - im).abs() < 1e-13);
        }
    }
}

#[test]
fn rejects_invalid_weights_and_hermitian_shortcut() {
    let cell = he_fcc();
    for weights in [&[][..], &[f64::NAN][..], &[f64::INFINITY][..]] {
        assert!(
            intor_cross_with_image_weights(
                "int1e_ovlp",
                &cell,
                &cell,
                &[],
                PbcIntorOpts {
                    hermi: 0,
                    ..Default::default()
                },
                &[[0.0; 3]],
                None,
                Some(weights)
            )
            .is_err()
        );
    }
    assert!(
        intor_cross_with_image_weights(
            "int1e_ovlp",
            &cell,
            &cell,
            &[],
            PbcIntorOpts {
                hermi: 1,
                ..Default::default()
            },
            &[[0.0; 3]],
            None,
            Some(&[1.0])
        )
        .is_err()
    );
}
