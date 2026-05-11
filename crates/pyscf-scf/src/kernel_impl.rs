//! SCF cycle loop. Body filled in plan 03-11 (kernel internals split per WARNING 3).
pub(crate) fn scf_loop<H: crate::OverrideHooks>(
    _mol: &pyscf_core::Mole,
    _hooks: &H,
    _cfg: crate::KernelConfig,
) -> Result<crate::ScfResult, pyscf_core::PyscfRsError> {
    unimplemented!("plan 03-11 — verbatim port of pyscf/scf/hf.py:48-244")
}
