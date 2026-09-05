//! Correlated-method k-point helper (`pyscf/pbc/lib/kpts_helper.py:544-632`).

use std::collections::{HashMap, HashSet};

use pyscf_algebra::CTensor;

use crate::{Kconserv, get_kconserv};

/// Insertion-ordered symmetry orbits plus an O(1) key lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymmMap {
    entries: Vec<([usize; 3], Vec<[usize; 3]>)>,
    lookup: HashMap<[usize; 3], usize>,
}

impl SymmMap {
    pub fn entries(&self) -> &[([usize; 3], Vec<[usize; 3]>)] {
        &self.entries
    }

    pub fn get(&self, key: [usize; 3]) -> Option<&[[usize; 3]]> {
        self.lookup.get(&key).map(|&i| self.entries[i].1.as_slice())
    }
}

/// Helper for momentum conservation and the four ERI permutation symmetries.
#[derive(Debug, Clone)]
pub struct KptsHelper {
    pub kconserv: Kconserv,
    pub nkpts: usize,
    pub symm_map: Option<SymmMap>,
    operation: Option<Vec<u8>>,
}

impl KptsHelper {
    /// `KptsHelper(cell, kpts, init_symm_map=True)` (`kpts_helper.py:544-571`).
    pub fn new(a: &[[f64; 3]; 3], kpts: &[[f64; 3]]) -> Self {
        let mut out = Self::without_symm_map(a, kpts);
        out.build_symm_map(None);
        out
    }

    /// Construct only the table KMP2 consumes, avoiding the `O(nkpts^3)` map.
    pub fn without_symm_map(a: &[[f64; 3]; 3], kpts: &[[f64; 3]]) -> Self {
        Self {
            kconserv: get_kconserv(a, kpts),
            nkpts: kpts.len(),
            symm_map: None,
            operation: None,
        }
    }

    /// Build `symm_map` in upstream's observable `OrderedDict` insertion order.
    ///
    /// PORT: `pyscf/pbc/lib/kpts_helper.py:573-613`. This intentionally remains
    /// sequential: first-claim order determines both the orbit key and iteration
    /// order used by correlated methods. A parallel implementation needs a
    /// separate ordered claim pass, not a bare parallel outer loop.
    pub fn build_symm_map(&mut self, kptlist: Option<&[[usize; 3]]>) {
        let nk = self.nkpts;
        let requested: Vec<[usize; 3]> = match kptlist {
            Some(v) => v.to_vec(),
            None => (0..nk)
                .flat_map(|p| (0..nk).flat_map(move |q| (0..nk).map(move |r| [p, q, r])))
                .collect(),
        };
        let requested_set: Option<HashSet<[usize; 3]>> =
            kptlist.map(|_| requested.iter().copied().collect());
        let included = |v: [usize; 3]| requested_set.as_ref().is_none_or(|set| set.contains(&v));

        let mut completed = vec![false; nk * nk * nk];
        let mut operation = vec![0_u8; nk * nk * nk];
        let flat = |v: [usize; 3]| (v[0] * nk + v[1]) * nk + v[2];
        let mut entries = Vec::new();
        let mut lookup = HashMap::new();

        for key in requested {
            if completed[flat(key)] {
                continue;
            }
            let [kp, kq, kr] = key;
            let ks = self.kconserv.get(kp, kq, kr) as usize;
            let candidates = [
                ([kp, kq, kr], 0_u8),
                ([kr, ks, kp], 1_u8),
                ([kq, kp, ks], 2_u8),
                ([ks, kr, kq], 3_u8),
            ];
            let mut orbit = Vec::with_capacity(4);
            for (member, op) in candidates {
                if included(member) {
                    completed[flat(member)] = true;
                    operation[flat(member)] = op;
                    orbit.push(member);
                }
            }
            lookup.insert(key, entries.len());
            entries.push((key, orbit));
        }
        self.operation = Some(operation);
        self.symm_map = Some(SymmMap { entries, lookup });
    }

    /// `_operation[kp,kq,kr]` (`kpts_helper.py:624`).
    pub fn operation(&self, kp: usize, kq: usize, kr: usize) -> Option<u8> {
        self.operation
            .as_ref()
            .map(|ops| ops[(kp * self.nkpts + kq) * self.nkpts + kr])
    }

    /// Return a symmetry-related rank-four ERI (`kpts_helper.py:615-632`).
    pub fn transform_symm(
        &self,
        eri: &CTensor,
        nmo: [usize; 4],
        kp: usize,
        kq: usize,
        kr: usize,
    ) -> Result<CTensor, &'static str> {
        let op = self
            .operation(kp, kq, kr)
            .ok_or("KptsHelper symmetry map was not built")?;
        if eri.len() != nmo.iter().product::<usize>() {
            return Err("ERI length does not match its four dimensions");
        }
        let axes = match op {
            0 => [0, 1, 2, 3],
            1 => [2, 3, 0, 1],
            2 => [1, 0, 3, 2],
            3 => [3, 2, 1, 0],
            _ => return Err("invalid KptsHelper symmetry operation"),
        };
        Ok(transpose4(eri, nmo, axes, op >= 2))
    }
}

fn transpose4(x: &CTensor, shape: [usize; 4], axes: [usize; 4], conj: bool) -> CTensor {
    let out_shape = axes.map(|a| shape[a]);
    let mut out = CTensor::zeros(x.len());
    for a in 0..out_shape[0] {
        for b in 0..out_shape[1] {
            for c in 0..out_shape[2] {
                for d in 0..out_shape[3] {
                    let dst_idx = ((a * out_shape[1] + b) * out_shape[2] + c) * out_shape[3] + d;
                    let out_idx = [a, b, c, d];
                    let mut src = [0_usize; 4];
                    for out_axis in 0..4 {
                        src[axes[out_axis]] = out_idx[out_axis];
                    }
                    let src_idx =
                        ((src[0] * shape[1] + src[1]) * shape[2] + src[2]) * shape[3] + src[3];
                    out.re[dst_idx] = x.re[src_idx];
                    out.im[dst_idx] = if conj { -x.im[src_idx] } else { x.im[src_idx] };
                }
            }
        }
    }
    out
}
