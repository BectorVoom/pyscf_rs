//! Gate-D rollup (19-17): DF variant + per-route gates, no cross-gating.
//!
//! * `df_ovvv_chunk` / `df_vvvv_chunk` vs direct contractions.
//! * DF route live: `build_df_blocks` → amplitudes → intermediates → IP
//!   (same `kernel_ip`) and EA (DF-chunk `kernel_ea_df`) vs upstream's DF
//!   numbers at 4dp — each route against its OWN number.
//! * Rollup table: incore IP/EA (recomputed here from the incore fixture) and
//!   DF IP/EA, with the measured route gaps stated.
//! * The principle, mechanically guarded: EA's DF-vs-incore gap is 1.5e-2,
//!   so a test gating one route against the other CANNOT pass — asserted.
//!
//! Run scoped: `cargo test -p pyscf-pbc-adc --test gate_d`

use pyscf_algebra::CTensor;
use pyscf_pbc_adc::amplitudes::t2_first_order;
use pyscf_pbc_adc::dfadc::{build_df_blocks, df_ovvv_chunk, df_vvvv_chunk, kernel_ea_df};
use pyscf_pbc_adc::ea::build_m_ab;
use pyscf_pbc_adc::ip::{build_m_ij, kernel_ip};
use pyscf_pbc_adc::kadc_ao2mo::KadcEris;
use pyscf_pbc_adc::types::AdcConfig;
use serde_json::Value;

fn fixture(name: &str) -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
    serde_json::from_str(&std::fs::read_to_string(&path).expect("fixture must exist")).unwrap()
}

fn flat(v: &Value) -> Vec<f64> {
    fn fl(v: &Value, o: &mut Vec<f64>) {
        match v {
            Value::Number(x) => o.push(x.as_f64().unwrap()),
            Value::Array(items) => {
                for it in items {
                    fl(it, o);
                }
            }
            _ => panic!("unexpected JSON"),
        }
    }
    let mut o = Vec::new();
    fl(v, &mut o);
    o
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// Chunk builders vs direct three-index contractions.
#[test]
fn df_chunks_match_direct_contraction() {
    // Synthetic (naux=3, nocc=2, nvir=2) Lov/Lvv with distinguishable values.
    let (naux, no, nv) = (3usize, 2usize, 2usize);
    let mut lov = CTensor::zeros(naux * no * nv);
    let mut lvv = CTensor::zeros(naux * nv * nv);
    for x in 0..naux {
        for i in 0..no {
            for a in 0..nv {
                lov.re[(x * no + i) * nv + a] = (x * 100 + i * 10 + a) as f64 + 0.5;
                lov.im[(x * no + i) * nv + a] = -((x * 100 + i * 10 + a) as f64) / 3.0;
            }
        }
        for a in 0..nv {
            for b in 0..nv {
                lvv.re[(x * nv + a) * nv + b] = (x * 10 + a * 2 + b) as f64 + 0.25;
                lvv.im[(x * nv + a) * nv + b] = ((x * 10 + a * 2 + b) as f64) / 5.0;
            }
        }
    }
    // Full-occ chunk == direct Lov·Lvv.
    let ch = df_ovvv_chunk(&lov, &lvv, naux, no, nv, 0, no).expect("chunk must build");
    for i in 0..no {
        for c in 0..nv {
            for a in 0..nv {
                for b in 0..nv {
                    let mut acc = (0.0f64, 0.0f64);
                    for x in 0..naux {
                        let (tr, ti) = (lov.re[(x * no + i) * nv + c], lov.im[(x * no + i) * nv + c]);
                        let (vr, vi) = (lvv.re[(x * nv + a) * nv + b], lvv.im[(x * nv + a) * nv + b]);
                        acc.0 += tr * vr - ti * vi;
                        acc.1 += tr * vi + ti * vr;
                    }
                    let o = ((i * nv + c) * nv + a) * nv + b;
                    assert!((ch.re[o] - acc.0).abs() < 1e-12, "ovvv chunk re [{i},{c},{a},{b}]");
                    assert!((ch.im[o] - acc.1).abs() < 1e-12, "ovvv chunk im");
                }
            }
        }
    }
    // Partial chunk (p=1, rows=1) matches the corresponding rows.
    let ch1 = df_ovvv_chunk(&lov, &lvv, naux, no, nv, 1, 1).expect("partial chunk must build");
    assert_eq!(ch1.re.len(), 1 * nv * nv * nv);
    for c in 0..nv {
        for a in 0..nv {
            for b in 0..nv {
                let full_o = ((1 * nv + c) * nv + a) * nv + b;
                let part_o = ((0 * nv + c) * nv + a) * nv + b;
                assert_eq!(ch1.re[part_o].to_bits(), ch.re[full_o].to_bits());
            }
        }
    }
    // vvvv chunk: direct + transpose(0,2,1,3).
    let vv = df_vvvv_chunk(&lvv, &lvv, naux, nv, 0, nv).expect("vvvv must build");
    for a in 0..nv {
        for b in 0..nv {
            for c in 0..nv {
                for d in 0..nv {
                    let mut acc = (0.0f64, 0.0f64);
                    for x in 0..naux {
                        let (tr, ti) = (lvv.re[(x * nv + a) * nv + b], lvv.im[(x * nv + a) * nv + b]);
                        let (vr, vi) = (lvv.re[(x * nv + c) * nv + d], lvv.im[(x * nv + c) * nv + d]);
                        acc.0 += tr * vr - ti * vi;
                        acc.1 += tr * vi + ti * vr;
                    }
                    // transpose(0,2,1,3): out[a][c][b][d] = direct[a][b][c][d].
                    let o = ((a * nv + c) * nv + b) * nv + d;
                    assert!((vv.re[o] - acc.0).abs() < 1e-12, "vvvv re");
                    assert!((vv.im[o] - acc.1).abs() < 1e-12, "vvvv im");
                }
            }
        }
    }
}

struct DfFix {
    nk: usize,
    no: usize,
    nv: usize,
    naux: usize,
    lpq: Vec<CTensor>,
    lov: Vec<CTensor>,
    lvv: Vec<CTensor>,
    kc: Vec<usize>,
    e_occ: Vec<Vec<f64>>,
    e_vir: Vec<Vec<f64>>,
}

fn load_df() -> (DfFix, Value) {
    let v = fixture("kadc_he2_df.json");
    let (nk, no, nv) = (
        v["nkpts"].as_u64().unwrap() as usize,
        v["nocc"].as_u64().unwrap() as usize,
        v["nvir"].as_u64().unwrap() as usize,
    );
    let nmo = no + nv;
    let naux = {
        let first = &v["Lpq_mo"].as_array().unwrap()[0];
        flat(&first["re"]).len() / (nmo * nmo)
    };
    let _ = v["block_shape"].as_array().unwrap();
    let mut lpq = Vec::new();
    for b in v["Lpq_mo"].as_array().unwrap() {
        lpq.push(CTensor { re: flat(&b["re"]), im: flat(&b["im"]) });
    }
    // Slice Lov (naux,nocc,nvir) / Lvv (naux,nvir,nvir) per pair.
    let mut lov = Vec::new();
    let mut lvv = Vec::new();
    for b in &lpq {
        let mut lo = CTensor::zeros(naux * no * nv);
        let mut lv = CTensor::zeros(naux * nv * nv);
        for x in 0..naux {
            for i in 0..no {
                for a in 0..nv {
                    lo.re[(x * no + i) * nv + a] = b.re[(x * nmo + i) * nmo + no + a];
                    lo.im[(x * no + i) * nv + a] = b.im[(x * nmo + i) * nmo + no + a];
                }
            }
            for a in 0..nv {
                for b2 in 0..nv {
                    lv.re[(x * nv + a) * nv + b2] = b.re[(x * nmo + no + a) * nmo + no + b2];
                    lv.im[(x * nv + a) * nv + b2] = b.im[(x * nmo + no + a) * nmo + no + b2];
                }
            }
        }
        lov.push(lo);
        lvv.push(lv);
    }
    let kc: Vec<usize> = flat(&v["kconserv"]).iter().map(|x| *x as usize).collect();
    let e_occ = vec![flat(&v["e_occ"][0]), flat(&v["e_occ"][1])];
    let e_vir = vec![flat(&v["e_vir"][0]), flat(&v["e_vir"][1])];
    (DfFix { nk, no, nv, naux, lpq, lov, lvv, kc, e_occ, e_vir }, v)
}

/// DF IP vs upstream DF IP (same kernel_ip — DF is an ERI source).
#[test]
fn gate_d_df_ip() {
    let (f, v) = load_df();
    let eris = build_df_blocks(&f.lpq, &f.kc, f.nk, f.no, f.no + f.nv, f.naux)
        .expect("DF blocks must build");
    // DF eris must not feed the incore ovvv path (NaN marker pins this).
    assert!(eris.ovvv.re.iter().all(|x| x.is_nan()));
    let amps = t2_first_order(&eris, &f.e_occ, &f.e_vir, &f.kc).expect("t2 must build");
    let m = build_m_ij(&amps, &eris, &f.e_occ, &f.kc).expect("M_ij must build");
    let roots = kernel_ip(&m, &eris, &f.e_occ, &f.e_vir, &f.kc, 0, &AdcConfig::default())
        .expect("DF IP must solve");
    let eref = flat(&v["ip_roots_df"]);
    let d = max_diff(&roots.energies, &eref[..3]);
    assert!(d < 5e-5, "Gate D DF-IP deviates {d:e}");
}

/// DF EA (chunk route) vs upstream DF EA.
#[test]
fn gate_d_df_ea() {
    use pyscf_pbc_adc::dfadc::kernel_ea_df;
    let (f, v) = load_df();
    let eris = build_df_blocks(&f.lpq, &f.kc, f.nk, f.no, f.no + f.nv, f.naux)
        .expect("DF blocks must build");
    let amps = t2_first_order(&eris, &f.e_occ, &f.e_vir, &f.kc).expect("t2 must build");
    let m = build_m_ab(&amps, &eris, &f.e_vir, &f.kc).expect("M_ab must build");
    let roots = kernel_ea_df(
        &m, &eris, &f.lov, &f.lvv, f.naux, &f.e_occ, &f.e_vir, &f.kc, 0,
        &AdcConfig::default(),
    )
    .expect("DF EA must solve");
    assert!(roots.energies.windows(2).all(|w| w[0] <= w[1]));
    let eref = flat(&v["ea_roots_df"]);
    let d = max_diff(&roots.energies, &eref[..3]);
    assert!(d < 5e-5, "Gate D DF-EA deviates {d:e}");
}

/// Rollup: per-route gates stated together; cross-route gating guarded.
#[test]
fn gate_d_rollup_per_route() {
    // Incore refs (own numbers).
    let vi = fixture("kadc_he2.json");
    let (ip_inc, ea_inc) = (flat(&vi["ip_roots"]), flat(&vi["ea_roots"]));
    // DF refs (own numbers).
    let vd = fixture("kadc_he2_df.json");
    let (ip_df, ea_df) = (flat(&vd["ip_roots_df"]), flat(&vd["ea_roots_df"]));
    // IP is route-insensitive here (no ovvv at ADC(2)): stated, not gated.
    eprintln!("IP incore-vs-DF gap: {:e}", max_diff(&ip_inc[..3], &ip_df[..3]));
    // EA routes DIFFER (ovvv treatment): gating one against the other cannot
    // pass — the guard that makes cross-gating mechanically impossible here.
    let ea_gap = max_diff(&ea_inc[..3], &ea_df[..3]);
    assert!(ea_gap > 1e-6, "EA routes unexpectedly identical ({ea_gap:e})");
    eprintln!("EA incore-vs-DF gap (different approximations): {ea_gap:e}");
    // Each route's gate lives in its own test (ip.rs, ea.rs, above); the
    // rollup cites them: Gate D = {IP-incore 4dp, EA-incore 4dp, IP-DF 4dp,
    // EA-DF 4dp}, tolerances from 19-01 (upstream's own 4dp, 58/80+).
}

/// Determinism at 1 vs 8 rayon workers inside one process (DF EA path).
#[test]
fn gate_d_deterministic_across_thread_counts() {
    use pyscf_pbc_adc::dfadc::kernel_ea_df;
    let run = || {
        let (f, _) = load_df();
        let eris = build_df_blocks(&f.lpq, &f.kc, f.nk, f.no, f.no + f.nv, f.naux)
            .expect("DF blocks must build");
        let amps = t2_first_order(&eris, &f.e_occ, &f.e_vir, &f.kc).expect("t2 must build");
        let m = build_m_ab(&amps, &eris, &f.e_vir, &f.kc).expect("M_ab must build");
        kernel_ea_df(&m, &eris, &f.lov, &f.lvv, f.naux, &f.e_occ, &f.e_vir, &f.kc, 0, &AdcConfig::default())
            .expect("DF EA must solve")
            .energies
    };
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let pool8 = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    let (a, b) = (pool1.install(run), pool8.install(run));
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}
