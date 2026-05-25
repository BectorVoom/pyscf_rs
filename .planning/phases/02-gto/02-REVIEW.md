---
phase: 02-gto
reviewed: 2026-05-23T11:31:06Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - crates/pyscf-core/src/density.rs
  - crates/pyscf-gto/src/ecp_engine_cintx.rs
  - crates/pyscf-gto/src/lib.rs
  - crates/pyscf-gto/src/projection.rs
  - crates/pyscf-gto/tests/dump_intor_for_oracle.rs
  - crates/pyscf-gto/tests/ecp_engine_stub.rs
  - crates/pyscf-gto/tests/ecp_int1e_oracle.rs
  - crates/pyscf-gto/tests/intor_smoke.rs
  - tests/oracle/test_ecp_int1e.py
findings:
  critical: 0
  warning: 3
  info: 4
  total: 7
status: issues_found
---

# Phase 02-gto: Code Review Report

**Reviewed:** 2026-05-23T11:31:06Z
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues_found

## Summary

This change closes the GTO-05 evaluation half: a cintx-backed `CintxEcpEngine`,
a `Density::from_flat` helper, an ECP-augmented `cintx_core::BasisSet` projection
builder (`build_cintx_basis_set_with_ecp`), the `int1e_ecp` oracle harness, and
updates to the stub tests after swapping `ecp_engine()` from the
`EcpEngineNotAvailable` stub to the real engine.

The core integral mechanics are sound. I cross-checked the new engine's per-pair
block-stitch against the cintx-side reference collector
(`cintx/crates/cintx-oracle/tests/safe_api_ecp_parity.rs:221`): the engine reads
`block[ii * nj + jj]` (row-major within the shell-pair block, bra slow / ket
fast), which **matches** the cintx safe-API contract verified there at
`atol=1e-12`. The ECP shell projection (`projection.rs`) is a faithful mirror of
`make_ecp_env`'s `(atom, channel, distinct n_power)` grouping
(`format_ecp.rs:208-257`) — same iteration order, same first-occurrence n_power
grouping, and `n_power` maps correctly onto cintx's literal `radial_power`
(confirmed against the NWChem parser at `nwchem_ecp.rs:184` and the cintx fixture
at `safe_api_ecp_parity.rs:152`). `total_ao` is AO-only in cintx
(`BasisMeta::from_shells`), so the ECP-augmented basis and the stored ECP-free
basis report identical `nao`, and the dispatcher's defensive shape check against
`mol.nao_nr` agrees with the engine. Error handling in the new engine path is
clean — no `unwrap`/`expect`/`panic` in any production code path; every fallible
cintx call is `map_err`-wrapped into a structured `PyscfRsError`.

No BLOCKER-class defects found. The warnings center on a routing gap that lets
`int1e_ecp_ipnuc` / `int1e_ecp_iprinv` silently produce a wrong-shaped scalar
matrix, a silent `unwrap_or(0)` that can corrupt the output buffer on a
should-never-happen path, and a stale documentation claim in a test file that the
change touched. The info items are maintainability/test-clarity nits.

## Warnings

### WR-01: `int1e_ecp_ipnuc` / `int1e_ecp_iprinv` silently resolve to the scalar operator instead of erroring

**File:** `crates/pyscf-gto/src/ecp_engine_cintx.rs:68-87` (with `crates/pyscf-gto/src/intor.rs:86`)

**Issue:** The dispatcher routes **every** name with prefix `int1e_ecp` to
`EcpEngine::ecp_int1e` (`intor.rs:86`), never to the trait's `ecp_int1e_ipnuc`
method. Inside the engine, representation is decided purely on the `_sph` /
`_cart` suffix, so a gradient name like `int1e_ecp_ipnuc` is suffix-normalised to
`int1e_ecp_ipnuc_sph`, matches `name.ends_with("_sph")`, and resolves to the
**scalar** `OperatorId::INT1E_ECP_SPH` — *not* the 3-component
`INT1E_ECP_IPNUC_SPH`. On an ECP molecule this returns a finite `nao × nao`
scalar matrix mislabeled as the gradient (the dispatcher then reports
`shape == [nao, nao]`), i.e. a silently wrong answer for any `ipnuc`/`iprinv`
caller. The engine docstring (lines 33-35) claims "the `ipnuc` gradient arm is
gated to Phase 7 GRAD-07 via the trait's `ecp_int1e_ipnuc` default," but that gate
is unreachable through the dispatcher, so it provides no protection. The
`ecp_engine_stub.rs` test `int1e_ecp_iprinv_on_ecpless_mol_...` only passes
because the ECP-less guard fires *first*; on an ECP molecule the same name would
not error.

**Fix:** Reject derivative/gradient ECP names explicitly in the engine until
Phase 7 wires them, e.g. before the representation block:

```rust
// Only the scalar ECP operator is wired in Phase 2. Gradient/derivative
// names (ipnuc, iprinv, ip*) are Phase 7 GRAD-07 and must not silently
// fall through to the scalar operator.
let core = name
    .strip_suffix("_sph")
    .or_else(|| name.strip_suffix("_cart"))
    .unwrap_or(name);
if core != "int1e_ecp" && !core.starts_with("ECPscalar") {
    return Err(PyscfRsError::NotYetImplemented {
        phase: 7,
        what: "ECP derivative integrals (int1e_ecp_ip*/ipnuc — Phase 7 GRAD-07)",
    });
}
```

(Or have the dispatcher route `*ipnuc*`/`*ip*` names to `ecp_int1e_ipnuc`.)

### WR-02: `unwrap_or(0)` on shell offset/count can silently corrupt the output buffer

**File:** `crates/pyscf-gto/src/ecp_engine_cintx.rs:111-114`

**Issue:** `meta.shell_offset(s).unwrap_or(0)` and `meta.ao_count(s).unwrap_or(0)`
swallow a `None` by substituting `0`. If `shell_offset` ever returned `None` for a
valid `s` (a basis/meta invariant break), the stitch loop would write into
`out[0 + 0*nao]` repeatedly, silently overwriting `M[0,0]` with later blocks and
producing a corrupt-but-finite matrix that passes the oracle's finite/non-zero
checks. The indices `s` are always `0..nbas`, so `None` should be impossible — but
a should-never-happen value resolved to `0` is exactly the kind of defect the
adversarial review is meant to surface, and it's silent rather than loud. (Same
pattern pre-exists in `intor.rs:225-228`; this change propagates it.)

**Fix:** Treat a missing offset/count as an internal error rather than `0`:

```rust
let shell_offsets: Vec<usize> = (0..nbas)
    .map(|s| meta.shell_offset(s).ok_or_else(|| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "ECP basis meta missing shell_offset for shell {s} of {nbas}"
        )))
    }))
    .collect::<Result<_, _>>()?;
// likewise for ao_count
```

### WR-03: Stale "synthetic staging" caveat in a test file the change touched is now misleading

**File:** `crates/pyscf-gto/tests/intor_smoke.rs:17-38`

**Issue:** This commit edited `intor_smoke.rs` (docstring + test renames at the top
and around line 160), but left the large "NOTE on the H2/STO-3G analytical
assertion" block asserting that "the current cintx-rs safe-API executor … populates
output staging via `fill_staging_values` — a synthetic pattern." That symbol no
longer exists anywhere in the cintx workspace (`grep fill_staging_values
cintx/crates` → no hits), and the sibling `ecp_int1e_oracle.rs` added in this same
plan asserts a *real, symmetric, non-zero* ECP matrix — which is only meaningful if
cintx is doing real evaluation. The two files now make contradictory claims about
the same dependency's state, which will mislead the next reader into thinking the
overlap numbers are fake when they are not.

**Fix:** Update or delete the `fill_staging_values` paragraph in `intor_smoke.rs`
to reflect that cintx now performs real evaluation (the executor runs
`executor.execute(&plan, &mut io)` in `cintx-rs/src/api.rs:291`), and that the
structural-only assertions are a deliberate de-risk rather than a workaround for
synthetic data.

## Info

### IN-01: `ecp_engine_stub.rs` tests now mostly exercise the engine, not the stub

**File:** `crates/pyscf-gto/tests/ecp_engine_stub.rs:30-67`

**Issue:** Three of the five tests in this stub-titled file
(`int1e_ecp_on_ecpless_mol_...`, `int1e_ecp_iprinv_on_ecpless_mol_...`,
`ECPscalar_prefix_on_ecpless_mol_...`) route through `intor(...)` and therefore
exercise the **cintx engine's** `mol._ecp.is_empty()` guard, not the
`EcpEngineNotAvailable` stub. Only `stub_int1e_returns_engine_not_available` and
`stub_ipnuc_returns_phase_7_not_yet_implemented` call the stub directly. The file
header explains this, but the file name and the first three test names still imply
"stub" coverage. Per the review focus ("whether the stub-test rewrites still
meaningfully assert the stub behavior"): the two direct-stub tests do; the other
three now assert engine-guard behavior that is already covered by the same
assertions in `intor_smoke.rs:159-179`, so they are duplicative.

**Fix:** Either move the three dispatcher-routing tests into `intor_smoke.rs`
(where the engine-guard contract belongs) and keep `ecp_engine_stub.rs` focused on
the two direct-stub assertions, or rename the file to reflect that it now covers
the ECP error-contract end-to-end.

### IN-02: ECP oracle test does not assert any reference values, only structural shape

**File:** `crates/pyscf-gto/tests/ecp_int1e_oracle.rs:36-90`

**Issue:** The in-tree gate asserts shape, finiteness, non-zero count, and
symmetry — all good structural guards — but no numerical reference. Symmetry is a
weak signal because the cintx safe-API engine fills `out[p,q]` and `out[q,p]` from
the transposed `(i,j)`/`(j,i)` shell-pair blocks of the *same* symmetric operator,
so it would pass even if the absolute magnitudes were off by a constant factor.
The byte-identity check correctly lives in the Python oracle, which is venv-gated
and skipped in normal CI — so a magnitude regression that preserves symmetry would
slip past the always-on gate.

**Fix:** Acceptable as scoped (numerical parity is delegated to the oracle), but
consider asserting at least one known matrix element (or the matrix trace) against
a pinned reference value so the always-on gate catches scale regressions without
the upstream venv.

### IN-03: No length-consistency guard on per-channel ECP primitive vectors

**File:** `crates/pyscf-gto/src/projection.rs:107-117`

**Issue:** The zip of `n_powers.iter().zip(exponents).zip(coeffs)` silently
truncates to the shortest of the three vectors. The NWChem parser
(`nwchem_ecp.rs:184-186`) pushes all three in lockstep, so they are equal-length
in practice and this is not a live bug — but neither the parser type nor the
projection enforces the invariant, so a future parser path or a hand-built
`EcpInput::Parsed` with mismatched vectors would be silently mis-projected rather
than rejected. The same latent gap exists in `make_ecp_env`.

**Fix:** Add a defensive `debug_assert_eq!` (or a hard `Err`) that
`channel.n_powers.len() == channel.exponents.len() == channel.coeffs.len()` before
the grouping loop.

### IN-04: Channel `l == 6` ("I") is parseable but unreachable through ECP_LMAX gate

**File:** `crates/pyscf-gto/src/projection.rs:84-94` (with `crates/pyscf-gto/src/basis/nwchem_ecp.rs:136`)

**Issue:** The NWChem parser accepts an `"I"` channel and stores `l = 6`, but cintx
caps projectors at `ECP_LMAX = 5`, so `build_cintx_basis_set_with_ecp` returns
`InvalidMolecule("ECP projector l=6 … exceeds ECP_LMAX=5")`. The error is correct
and loud (good — not a silent failure), but the mismatch between what the parser
admits and what the engine can project means an `I`-channel ECP loads fine via
`format_ecp` and only fails later at evaluation time. This is a latent
"load succeeds, evaluate fails" seam rather than a bug in this change.

**Fix:** No action required for this plan; flagging for the GTO-05 backlog —
either reject `l > ECP_LMAX` at parse time in `nwchem_ecp.rs`, or document the
deferred-failure boundary.

---

_Reviewed: 2026-05-23T11:31:06Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
