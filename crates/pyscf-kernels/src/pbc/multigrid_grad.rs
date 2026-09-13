//! Shared Phase-18 atom/grid contraction (D-PBC-31 clause 10).
//!
//! Fields are complex, component-leading per atom: `(natm, 3, ngrids)`.
//! Returns `Re(sum_g field[a,x,g] * density[g])`, WITHOUT conjugation.
//! Callers encode their Fourier convention in the field. Products are formed
//! on device; each row is reduced by oracle_sum, never by device atomics.

use cubecl::bytes::Bytes;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use pyscf_algebra::{AlgebraClient, AlgebraError, dispatch_backend, oracle_sum};

#[cube(launch_unchecked)]
fn real_products<F: Float>(
    field_re: &Array<F>,
    field_im: &Array<F>,
    rho_re: &Array<F>,
    rho_im: &Array<F>,
    out: &mut Array<F>,
    ngrids: usize,
) {
    let t = ABSOLUTE_POS;
    if t < out.len() {
        let g = t % ngrids;
        out[t] = field_re[t] * rho_re[g] - field_im[t] * rho_im[g];
    }
}

fn launch<R: Runtime>(
    client: &ComputeClient<R>,
    fr: &[f64],
    fi: &[f64],
    rr: &[f64],
    ri: &[f64],
) -> Vec<f64> {
    let handles = [fr, fi, rr, ri].map(|v| client.create(Bytes::from_elems(v.to_vec())));
    let out = client.empty(fr.len() * size_of::<f64>());
    // SAFETY: the public entry validates all plane lengths, a nonempty grid,
    // and a u32-addressable launch. Every output is guarded by its length.
    unsafe {
        real_products::launch_unchecked::<f64, R>(
            client,
            CubeCount::Static((fr.len() as u32).div_ceil(256), 1, 1),
            CubeDim::new_1d(256),
            ArrayArg::from_raw_parts(handles[0].clone(), fr.len()),
            ArrayArg::from_raw_parts(handles[1].clone(), fi.len()),
            ArrayArg::from_raw_parts(handles[2].clone(), rr.len()),
            ArrayArg::from_raw_parts(handles[3].clone(), ri.len()),
            ArrayArg::from_raw_parts(out.clone(), fr.len()),
            rr.len(),
        );
    }
    let bytes = client.read(vec![out]);
    bytemuck::cast_slice::<u8, f64>(&bytes[0]).to_vec()
}

/// Batched complex atom/grid contraction. Empty grids return zero rows.
///
/// For memory blocking callers may invoke this on grid slabs, then combine
/// the materialized slab partials with oracle_sum in a fixed order.
pub fn contract_atom_grid(
    client: &AlgebraClient,
    natm: usize,
    field_re: &[f64],
    field_im: &[f64],
    rho_re: &[f64],
    rho_im: &[f64],
) -> Result<Vec<[f64; 3]>, AlgebraError> {
    let count = natm
        .checked_mul(3)
        .and_then(|rows| rows.checked_mul(rho_re.len()));
    if count != Some(field_re.len())
        || field_im.len() != field_re.len()
        || rho_im.len() != rho_re.len()
        || field_re.len() > u32::MAX as usize
        || natm > isize::MAX as usize / size_of::<[f64; 3]>()
    {
        return Err(AlgebraError::ShapeMismatch {
            expected: "field planes (natm,3,ngrids), density planes (ngrids), addressable launch"
                .into(),
            actual: format!(
                "natm={natm}, fields=({},{}), density=({},{})",
                field_re.len(),
                field_im.len(),
                rho_re.len(),
                rho_im.len()
            ),
        });
    }
    if field_re.is_empty() {
        return Ok(vec![[0.0; 3]; natm]);
    }
    let products = dispatch_backend!(
        client,
        c,
        Rt,
        launch::<Rt>(c, field_re, field_im, rho_re, rho_im)
    );
    let mut result = vec![[0.0; 3]; natm];
    for (row, values) in products.chunks_exact(rho_re.len()).enumerate() {
        result[row / 3][row % 3] = oracle_sum(values);
    }
    Ok(result)
}
