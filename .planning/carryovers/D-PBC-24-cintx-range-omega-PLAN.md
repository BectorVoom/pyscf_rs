---
carryover: D-PBC-24
title: "Range-separated integrals — close the cintx `range_omega` (libcint `env[8]`) gap"
raised_by: 14-07 (Task 7b), 14-08 (Tasks 2 and 4)
blocks:
  - "Phase 14 Gate 3 — |E_KRHF(GDF) − E_KRHF(RSDF)| against upstream's floor"
  - "rsdf_builder::_RSGDFBuilder (14-07 sub-tasks 7b/7c) and Task 7d's _prefer_ccdf flip"
  - "mdf::_RSMDFBuilder (14-06's other half; every row of measurements/mdfladder.out)"
  - "rsdf::RSGDF (14-08 Task 2)"
  - "pyscf_pbc_scf::rsjk (14-08 Task 4)"
  - "GDF/MDF get_jk(omega) — the RSH branch"
  - "Phase 4's numerical RSH assertion (CAM-B3LYP/H2O), CI-gated since 2026 on this same gap"
implementation_repo: cintx (crates/cintx-runtime, cintx-rs, cintx-compat, cintx-cubecl)
consumer_repo: pyscf_rs (crates/pyscf-pbc-df, pyscf-pbc-scf, pyscf-gto)
status: NOT STARTED — evidence gathered 2026-08-30, no code written
autonomous: false
must_haves:
  truths:
    - "range separation is NOT a distinct integral symbol; upstream never calls an int2e_sr_*"
    - "SR for rys_order <= 3 needs NO new quadrature — libcint computes it as full minus LR with doubled roots"
    - "the SR nroots doubling changes WORKSPACE SIZE, so query_workspace must see omega — this is a control-plane change, not only a kernel-body one"
    - "a full-range substitute must never ship: it runs, converges, and is silently a different method"
---

# D-PBC-24 — close the cintx `range_omega` gap

## Why this document exists

Phase 14 stopped at four of five gates. The fifth — Gate 3, `|E_KRHF(GDF) −
E_KRHF(RSDF)|` against upstream's measured floor — is not *deferred by choice*.
It is blocked on a capability `cintx` does not expose, and `14-07-PLAN.md`
Task 7b required that outcome be **reported rather than worked around with a
numerically different kernel**. This plan is that report, plus the work needed
to lift it.

Everything below was verified against source on 2026-08-30. File:line citations
are to the trees at `/home/user/Documents/workspace/cintx` and
`/home/user/Documents/workspace/cintx/libcint-master` (the vendored libcint
reference).

---

## 1. The blocker, in three independent parts

### 1a. `ExecutionOptions` has no `range_omega`

`cintx-runtime/src/options.rs:96-133` defines `ExecutionOptions`. Its
operator-parameter fields are exactly:

| field | libcint env slot | line |
|---|---|---|
| `f12_zeta: Option<f64>` | `PTR_F12_ZETA` = `env[9]` | `options.rs:108` |
| `rinv_orig: Option<[f64; 3]>` | `PTR_RINV_ORIG` = `env[4..6]` | `options.rs:113` |
| `common_orig: Option<[f64; 3]>` | `PTR_COMMON_ORIG` = `env[1..3]` | `options.rs:118` |

There is **no `range_omega`**. The same three, and only those three, appear on
`OperatorEnvParams` (`cintx-runtime/src/planner.rs:43-58`), on the builder
(`cintx-rs/src/builder.rs:87-115`), and in the validators
(`cintx-runtime/src/validator.rs:147`, `:182`, `:208`).

So there is no way for a safe-API caller to say "evaluate this `int3c2e` with
`erfc(ωr)/r`".

### 1b. No kernel reads `env[8]`

`cintx-compat/src/raw.rs:35-41` names `PTR_RANGE_OMEGA = 8` **only in a comment
warning callers not to overwrite the slot**. Nothing consumes it.

The 2-electron kernel is unconditional (`cintx-cubecl/src/kernels/two_electron.rs:452-462`):

```rust
let a1 = aij * akl;
let a0 = a1 / (aij + akl);
let fac1 = (a0 / (a1 * a1 * a1)).sqrt() * fac_env;
let x_rys = a0 * rr;

let (u_roots, mut w_weights) = rys_roots_host(shape.nroots, x_rys);
```

That is libcint's `omega == 0` branch verbatim and nothing else.
`center_3c2e.rs` and `center_2c2e.rs` are the same shape, and
`crates/cintx-cubecl/src/math/rys_wheeler.rs:4` states the gap in its own module
docs:

> `lower == 0` only — the short-range `lower != 0` path is out of scope for
> Phase 25

### 1c. The periodic 3-centre driver never builds an `_env`, so the molecular workaround is unreachable

`pyscf-gto` already ships an `env[8]` set/restore contract:
`crates/pyscf-gto/src/range_coulomb.rs` (`PTR_RANGE_OMEGA` at `:52`, `OmegaGuard`
at `:64-102`, `intor_with_omega` at `:119-131`). It writes `mol._env[8]`
directly and restores it on drop, including on unwind. Phase 4 recorded there
that this is a *contract* test only, because cintx does not consume the slot.

That workaround is not even *available* on the periodic path.
`pyscf_pbc_df::incore::aux_e2` reaches cintx through
`pyscf_gto::build_image_expanded_with_aux`
(`crates/pyscf-gto/src/projection.rs:569-632`), which assembles a
`cintx_core::BasisSet` out of `cell.mol._atom` and `cell.mol._basis` — the
parsed per-element basis — via `build_atoms_and_shells_with_base`. **No `_env`
array is constructed or passed.** The same is true of
`build_cintx_basis_set` (`projection.rs:27-44`) and every other periodic entry
point.

**This third point is a reason the *workaround* fails, not a reason the *fix*
is impossible.** It is what makes the `ExecutionOptions` route (1a) the only
one — and that route is also the correct one, because it serves the molecular
and periodic paths with a single mechanism instead of two.

### The upstream precedent that settles the design

libcint has no `int2e_sr_*` symbol and PySCF never asks for one.
`pyscf/pbc/df/rsjk.py:186` sets `supmol_sr.omega = -self.omega` and then calls
the **standard** `int2e`; `pyscf/df/df.py:299-335` (`range_coulomb`) and
`pyscf/pbc/df/aft.py:552` do the same by toggling `mol.omega`. Any design that
introduces a new operator name is wrong on its face.

---

## 2. What the fix is NOT

Each of these was considered and rejected. They are listed so nobody re-proposes
them.

| rejected | why |
|---|---|
| **Substitute the full-range kernel and carry the error** | It runs, converges, and is silently a different method. For `rsjk` — which is EXACT, no fitting — the wrong answer lands *inside* GDF's 1.222e-03 fitting error and looks plausible. `14-07-PLAN.md` Task 7b forbids it in writing. |
| **Add `int2e_sr_*` / `int3c2e_sr` operator symbols** | libcint has none and PySCF never calls one; the resolver correctly knows only `int3c2e` and `int3c2e_ip1` (`cintx-ops/src/resolver.rs:362`). A new symbol would fork the descriptor table and diverge from the reference. |
| **Write `_env[8]` from the periodic driver** | There is no `_env` on that path (§1c). Even on the molecular path it is inert until §3 lands. |
| **Compute SR as `full − LR` at the CONTRACTION level** (evaluate both, subtract in `pyscf-pbc-df`) | Doubles the integral cost, and loses accuracy exactly where SR matters: at small `ω·r` the two terms nearly cancel. libcint does do this internally for `rys_order <= 3`, but at the *root* level with a shared `fac1`, where the cancellation is between quadrature weights and not between two separately-rounded integrals. See §3.4 — doing it right is no harder. |
| **Wait for a GPU implementation** | The host Rys path (`rys_roots_host`) is where the change belongs; the device arms follow. Staging is in §4. |

---

## 3. The algorithm, from the reference

All line numbers are `cintx/libcint-master/src`.

### 3.1 Three branches, in `CINTg0_2e` (`g2e.c:4443-4512`)

`CINTg0_2e` is shared: `g3c2e.c:131` and `g2c2e.c:104` both set
`envs->f_g0_2e = &CINTg0_2e`, with `aj = al = 0` and the unused angular momenta
zeroed. **So there is ONE place to implement this per port, not three** — which
matches cintx, where `two_electron.rs`, `center_3c2e.rs` and `center_2c2e.rs`
each carry their own copy of the same `x_rys` / `rys_roots_host` prologue.

```text
a0    = aij*akl/(aij+akl)
fac1  = sqrt(a0/(aij*akl)^3) * fac
x     = a0 * rr

omega == 0   →  rys_roots(nroots, x)                          [today's code]

omega  > 0   →  theta = ω²/(ω² + a0)                          long range
                x    *= theta
                fac1 *= sqrt(theta)
                rys_roots(nroots, x)
                for each root:  ut = u*theta;  u = ut/(u + 1 − ut)

omega  < 0   →  theta = ω²/(ω² + a0)                          short range
                if theta*x > cutoff or theta*x > EXPCUTOFF_SR(=40): return 0
                if rys_order == nroots:            → sr_rys_roots(nroots, x, sqrt(theta))
                else (rys_order <= 3, nroots = 2*rys_order):
                    rys_roots(rorder, x)          into u[0..rorder], w[0..rorder]
                    rys_roots(rorder, theta*x)    into u[rorder..],  w[rorder..]
                    for irys in rorder..nroots:
                        ut = u*theta;  u = ut/(u + 1 − ut)
                        w *= −sqrt(theta)
```

`EXPCUTOFF_SR = 40` is `rys_roots.h:46`.

### 3.2 `nrys_roots` DOUBLES for SR — and that is a control-plane change

`g2e.c:76-79`, and identically `g3c2e.c:70-77` and `g2c2e.c:61-68`:

```c
int rys_order = (li_ceil + lj_ceil + lk_ceil + ll_ceil)/2 + 1;
int nrys_roots = rys_order;
double omega = env[PTR_RANGE_OMEGA];
if (omega < 0 && rys_order <= 3) {
        nrys_roots *= 2;
}
```

`nrys_roots` sets `g_stride_i`, `g_stride_k`, `g_size` — i.e. **the workspace
size**. In cintx that is decided by
`estimate_workspace_request(descriptor, basis, shells)`
(`cintx-runtime/src/planner.rs:435-439`), whose signature **does not take
`opts`**. So SR cannot be added as a kernel-body change alone: `query_workspace`
must become omega-aware, or the query/evaluate contract (D-08, the capability
token) breaks the moment a caller asks for SR.

This is the single most under-appreciated part of the work and it is why §4
sequences the control plane first.

### 3.3 SR needs a new quadrature — but only above `rys_order = 3`

`CINTsr_rys_roots` (`rys_roots.c:145-200+`) is the lower-bounded Rys
quadrature `∫_lower^1`. It dispatches across `CINTrys_schmidt`,
`CINTqrys_jacobi`, `segment_solve`, `CINTlrys_jacobi`, `CINTlrys_laguerre` and
`CINTsr_rys_polyfits` on `(nroots, lower)`. cintx has the `lower == 0` members
of that family (`rys_wheeler.rs:711` `CINTrys_schmidt`, `:1147`
`llaguerre_moments`, `:3329` the `CINTrys_roots` dispatch) and **explicitly
none of the `lower != 0` ones**.

### 3.4 THE FINDING: the systems Phase 14 gates on never reach that branch

`rys_order = (Σ l_ceil)/2 + 1`, and the doubled-root path is taken whenever
`rys_order <= 3`. Working it out for this project's reference cells:

| integral | angular momenta | `rys_order` | SR path |
|---|---|---|---|
| `int3c2e`, He-fcc `sto-3g` + aux | `l_i = l_j = 0`, `l_k ≤ 2` | `(0+0+2)/2+1 = 2` | **doubled roots** |
| `int3c2e`, diamond `gth-szv` + aux | `l_i = l_j ≤ 1`, `l_k ≤ 2` | `(1+1+2)/2+1 = 3` | **doubled roots** |
| `int2c2e`, either auxcell | `l_i = l_k ≤ 2` | `(2+2)/2+1 = 3` | **doubled roots** |
| `int2e` (rsjk), s/p basis | all `l ≤ 1` | `(1+1+1+1)/2+1 = 3` | **doubled roots** |
| `int2e` (rsjk), d functions | all `l ≤ 2` | `(2+2+2+2)/2+1 = 5` | `sr_rys_roots` |

**So SR on every system Phase 14 gates — and on `rsjk` for an s/p basis — needs
no new quadrature at all.** It needs `rys_roots` called twice at two arguments,
a root rescaling, and a sign on half the weights. `CINTsr_rys_roots` is only
required for `rys_order > 3`, i.e. high angular momentum, and is a strictly
later stage.

That is what turns this carry-over from "unbounded" into the staged plan below.

### 3.5 Screening also reads omega, and skipping it is the conservative direction

`cint3c2e.c:108-124` and `optimizer.c:306-315` LOOSEN `expcutoff` /
`log_rr_ij` when `omega < 0`, because the SR integrand decays faster and a
distance-based bound tuned for `1/r` is too tight. A port that ignores these
keeps **more** primitives than upstream, so it is more converged, not less —
the same posture `pyscf-pbc-df` already takes toward `ExtendedMole.strip_basis`
(14-05's finding: the port keeps images upstream discards, worth 1.054e-09).

Record the omission; do not treat it as a correctness item for stages 1-3.

---

## 4. The work, staged

Each stage is independently landable and independently useful. **Do not start a
stage before its predecessor's tests are green** — the same discipline
`14-07-PLAN.md` imposed on 7a before 7b, and for the same reason: a wrong `ω`
does not fail loudly, it produces a plausible 1e-6.

### Stage 0 — measure the targets first (cintx repo, ~half a day)

Phase-9 precedent, and this project's house rule: record upstream numbers
*before* writing the code that must reproduce them.

Build a libcint oracle fixture that sweeps `env[PTR_RANGE_OMEGA]` over
`{0, +0.3, +0.8, −0.3, −0.8}` for `int2e`, `int3c2e` and `int2c2e` on shell
tuples covering `rys_order ∈ {1,2,3}` (the doubled-root regime) and
`{4,5}` (the `sr_rys_roots` regime). `cintx-oracle` already links libcint and
has the harness (`crates/cintx-oracle/tests/center_3c2e_parity.rs` is the
template). Record raw values, not just deviations.

**Acceptance:** a committed `.out` with per-tuple values, and a note stating
which tuples fall in which SR regime.

**Also record the identity** `SR(ω) + LR(ω) == full` on every tuple. It is the
one check that catches an `erf`/`erfc` swap, and `pyscf-pbc-df` already gates
its `weighted_coulG` half of it at exactly 0
(`crates/pyscf-pbc-df/tests/rsdf_builder.rs::sr_and_lr_coulg_sum_to_the_full_kernel`).

### Stage 1 — the control plane (cintx, ~1-2 days)

No numerical change. Every test must still pass bit-identically.

1. `ExecutionOptions::range_omega: Option<f64>` (`options.rs:96-133`), documented
   with the sign convention **`> 0` long-range `erf(ωr)/r`, `< 0` short-range
   `erfc(ωr)/r`, `None`/`0` full** — libcint's, and already the convention
   `pyscf_pbc_df::traits::JkOpts::omega` uses, so no second convention enters
   the workspace.
2. `OperatorEnvParams::range_omega: Option<f64>` (`planner.rs:43-58`).
3. `SessionBuilder::with_range_omega(f64)` (`builder.rs`, beside
   `f12_zeta` / `with_rinv_origin` / `with_common_origin` at `:87-115`).
4. Transfer on the safe path: `cintx-rs/src/api.rs:1079-1100` already moves
   `f12_zeta`, `rinv_orig` and `common_orig` from the options onto the plan —
   add `range_omega` in the same block.
5. Extraction on the raw path: `cintx-compat/src/raw.rs` (beside `:937`,
   `:958`, `:984`, `:1006`) reads `env[PTR_RANGE_OMEGA]` into the plan, so the
   `_env`-based callers — including `pyscf-gto`'s existing `OmegaGuard` — start
   working the moment stage 2 lands.
6. `validate_range_omega_env_params` (`validator.rs`, beside `:147`/`:182`/`:208`):
   finiteness; reject `range_omega` on operator families that have no
   `1/r₁₂` kernel (1-electron, ECP, grids) rather than silently ignoring it.
7. **`estimate_workspace_request` takes `&ExecutionOptions`** (`planner.rs:435`)
   and applies §3.2's doubling. This is the invasive edit and it is the reason
   the control plane is its own stage: it touches the query/evaluate capability
   token (D-08) and every call site.

**Acceptance:** whole cintx suite green and byte-identical with
`range_omega = None`; a new test asserts that `Some(ω)` with `ω < 0` and
`rys_order <= 3` raises the queried workspace to exactly `2×` the roots and that
`query`/`evaluate` agree on it.

### Stage 2 — LR, and SR for `rys_order <= 3` (cintx, ~3-5 days)

The whole of §3.1 except `sr_rys_roots`. Implement once per kernel family, at
the three `rys_roots_host` prologues: `two_electron.rs:452-462`,
`center_3c2e.rs` (near `:4987`) and `center_2c2e.rs:1143`. Factor the branch
into one shared helper so the three cannot drift — the mirror of what
`pyscf-pbc-df`'s `make_j3c` `Scheme` tag did for GDF and MDF.

Fail closed: `range_omega < 0` with `rys_order > 3` returns
`UnsupportedApi`, naming stage 3. **Never fall through to the full-range
kernel.**

**Acceptance:** stage 0's oracle at `rys_order ∈ {1,2,3}` for all five ω values,
byte-identical to libcint under `release-oracle`; `SR + LR == full` on every
tuple; `range_omega = None` still byte-identical to today.

**What this unblocks in `pyscf_rs`:** everything Phase 14 named. `_RSGDFBuilder`
(14-07 7b/7c), `_RSMDFBuilder`, `RSDF`, **Gate 3**, 14-07 Task 7d's
`_prefer_ccdf` flip, `rsjk` for s/p bases, and Phase 4's CAM-B3LYP/H2O RSH
assertion.

### Stage 3 — `CINTsr_rys_roots`, for `rys_order > 3` (cintx, ~1-2 weeks)

Port the lower-bounded quadrature family (`rys_roots.c:145` and its callees).
This is real numerical-analysis work — `segment_solve`, the double-double
Jacobi/Laguerre arms, and the `lower`-dependent dispatch thresholds
(`0.6 / 0.8 / 0.93 / 0.97 / 0.99`) are all accuracy-critical. cintx's
`rys_wheeler.rs` is the natural home; it already carries the `lower == 0`
members and says in its module docs that `lower != 0` was scoped out.

**Acceptance:** stage 0's oracle at `rys_order ∈ {4,5}`, and a sweep of
`lower` across every dispatch threshold.

**What this unblocks:** `rsjk` and RSH on d/f bases — i.e. real solid-state and
transition-metal work.

### Stage 4 — device arms (cintx, sizing unknown)

Stages 1-3 target the host Rys path. The device kernels carry comptime `nroots`
launch arms with a ceiling of `BASE_DEVICE_NROOTS = 5`
(`device_rys_ceiling.rs:52`; `EXTENDED_DEVICE_NROOTS = 12` at `:57` behind the
`extended-device-rys` feature). **SR doubling at `rys_order = 3` needs
`nroots = 6`, which is above the base ceiling** — see
`three_c2e_launch_nroots_ceiling()` at `center_3c2e.rs:94`. So on the device
path, SR at `rys_order = 3` requires the extended arms; at `rys_order ≤ 2`
(`nroots ≤ 4`) it does not.

Until this stage, SR must route to the host engine. Make that routing explicit
and logged, not incidental.

### Stage 5 — consume it (pyscf_rs, ~1-2 weeks, per `14-07-PLAN.md`)

Once stage 2 is in a released cintx:

1. Thread `omega` through `pyscf_pbc_df::incore::aux_e2_intor` into the
   `SessionRequest` options. **No change to
   `build_image_expanded_with_aux`** — the omega travels in the options, not in
   the basis, which is precisely why §1c is not an obstacle to the real fix.
2. `rsdf_builder` 7b/7c: `_RSGDFBuilder`'s `get_2c2e` / `outcore_auxe2` /
   `add_ft_j3c` / `solve_cderi`, and `_RSNucBuilder`. 7a is already shipped and
   gated — `crates/pyscf-pbc-df/src/rsdf_builder/omega.rs`, 10 tests at 1e-12
   against `measurements/omega.out`.
3. `mdf::_RSMDFBuilder` — and re-point Gate 2 at `measurements/mdfladder.out`,
   which was recorded on this route and which 14-06 had to replace with
   `mdfladder_cc.out`.
4. Task 7d: flip `Gdf::prefer_ccdf` to `false`, matching upstream. **This moves
   a committed reference energy** — diamond 2×2×2 goes from
   **−10.93209469510988** (CC) to **−10.93209529106394** (RS), a documented
   5.960e-07 step. Make it a one-line cited edit, as `14-07-PLAN.md` Task 7d
   requires.
5. `rsjk` in `pyscf-pbc-scf` — gated against **FFTDF, not GDF**
   (`14-08-PLAN.md` Task 5.3: gating an exact builder against a fitted one
   would hide a real error behind the 1.2e-3 fitting gap).
6. Delete the refusals and their tests: `RsGdfBuilder::build`, `Rsdf::build`,
   `density_fit(DfKind::Rsdf)`, `RangeSeparatedJkBuilder::{build, get_jk}`,
   and the `CINTX_SR_GAP` constant.

---

## 5. Gate 3, restated so it can be run the day stage 5 lands

From `measurements/builders.out` and `ccdf.py`, upstream 2.12.1:

| pair | diamond 2×2×2 | diamond gamma | He-fcc 2×2×2 |
|---|---|---|---|
| **GDF − RSDF** | **1.353e-08** | **4.566e-09** | **1.113e-10** |
| CC route − RS route | 5.960e-07 | 4.502e-06 | 5.222e-10 |

Gate 3 is `|E(GDF) − E(RSDF)|` landing on the first row within a factor of 2,
plus RSDF's own converged energy against upstream at 1e-11 on He-fcc
(target **−2.80842508705964**) and ≤3e-8 on diamond (**−10.93209530458920** /
**−10.14369691810517**).

The second row is the stronger evidence and should be asserted too: two
independent implementations of the same fitted quantity reproducing upstream's
own *disagreement* between its two routes says more than either matching alone.

---

## 6. Risks

| risk | mitigation |
|---|---|
| **A full-range substitute ships by accident** | Stage 2 fails closed with `UnsupportedApi`. The `pyscf_rs` refusal tests (`rs_gdf_builder_refuses_and_names_the_cintx_gap`, `rsjk_refuses_and_names_the_cintx_gap`, `density_fit_refuses_rsdf_and_names_the_gap`) assert the message text, so a silent substitution turns them red rather than green. |
| **Stage 1's `estimate_workspace_request` signature change is invasive** | It is the whole reason stage 1 is separate and carries a byte-identity acceptance gate. Land it with `range_omega = None` everywhere first. |
| **SR loses accuracy at small `ω·r` through cancellation** | libcint's own `EXPCUTOFF_SR = 40` early return (`g2e.c:4460`) is part of the algorithm, not an optimisation — port it. |
| **The doubled-root path is only exercised by SR** | Exactly the shape of the defect 14-06 found: `decompose_j2c`'s eigen branch had never been reached by any gate and was silently transposed, worth 6.3e6 Ha. Stage 0's oracle must cover `rys_order ∈ {1,2,3}` at `ω < 0` explicitly, not incidentally. |
| **Two ω sign conventions enter the codebase** | Fixed in stage 1.1: libcint's, which `JkOpts::omega` and `rsdf_builder::omega` already use. |
| **cintx and pyscf_rs drift on release timing** | Stages 1-3 are pure cintx and are independently valuable (RSH DFT is blocked on the same thing). Stage 5 pins a cintx version. |

---

## 7. What is already in place on the consumer side

Do not re-do these.

| shipped | where |
|---|---|
| All twelve ω estimators, gated at 1e-12 against `measurements/omega.out` | `crates/pyscf-pbc-df/src/rsdf_builder/omega.rs`; `tests/rsdf_builder.rs` (10 tests) |
| `weighted_coulG_LR` / `_SR`, and `SR + LR == full` at exactly 0 | same |
| `get_aux_chg`, equal to 14-01's monopole at 1e-14 | `crates/pyscf-pbc-df/src/rsdf.rs`; `tests/rsdf.rs` |
| One shared `density_fit` for all four upstream shims | `crates/pyscf-pbc-df/src/density_fit.rs` |
| `RangeSeparatedJkBuilder` with the ω half live and the integral half refused | `crates/pyscf-pbc-scf/src/rsjk.rs`; `tests/rsjk.rs` |
| `env[8]` set/restore with unwind safety | `crates/pyscf-gto/src/range_coulomb.rs` (Phase 4) |
| `get_coulG(omega)` with the same sign convention | `pyscf_pbc_gto::coulg`, `JkOpts::omega` |
| The pre-implementation ω measurements | `.planning/phases/14-gdf-mdf-rsdf-rsjk/measurements/omega.py` / `.out` |

## 8. Where this is recorded

* `PBC-MASTER-PLAN.md` §3 — **D-PBC-24**
* `.planning/phases/14-gdf-mdf-rsdf-rsjk/14-VERIFICATION.md` §5 (Gate 3
  unreachable) and §8 (carry-overs)
* `14-07-SUMMARY.md`, `14-08-SUMMARY.md`
* `crates/pyscf-pbc-df/src/rsdf_builder/mod.rs` — module docs and
  `CINTX_SR_GAP`
* `crates/pyscf-gto/src/range_coulomb.rs` — the same gap as Phase 4's Open
  Question A5 / cintx#11

**This is one gap with two victims.** Phase 4 CI-gated its CAM-B3LYP/H2O RSH
assertion on it in the molecular code; Phase 14 lost Gate 3 to it in the
periodic code. Stage 2 closes both.
