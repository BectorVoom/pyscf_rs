# Rust Test Governance

Use this skill to turn a Rust crate testing request into an operational test strategy with explicit tool selection, CI gates, reporting requirements, and residual-risk handling.

## Primary outcomes

Produce one or more of the following, depending on the task:

- A Rust crate test strategy or review
- A tool-selection decision with rationale
- CI gate recommendations or updates
- A testing gap analysis
- A report that distinguishes verified scope from unverified scope
- Concrete file updates for policy, CI, or templates

## Non-negotiable rules

- Do not treat passing tests as sufficient evidence of specification conformance.
- Do not treat coverage as a sufficient quality signal.
- Do not claim completeness when a required tool was not applied.
- Do not hide unverified areas, unsupported tooling, waived mutants, or resource-exhaustion outcomes.
- Assume fake implementations are possible and design checks to break them.

## Working method

1. Identify the verification target.
   - What crate or module is in scope?
   - What is the specification source?
   - Which public APIs, invariants, error contracts, feature flags, and side effects matter?

2. Classify the crate.
   Decide whether the target includes any of the following:
   - public API docs or doctests
   - compile-time usage constraints
   - stateful behavior or order-dependent workflows
   - unsafe code
   - concurrency or atomics
   - external-input parsers / decoders / formatters
   - multiple feature flags or non-default configurations
   - high-value invariants suitable for bounded verification

3. Select tools.
   Apply the mandatory baseline, then add conditional tools based on the crate classification. Use `references/tool-applicability-matrix.md` for the decision table.

4. Define acceptance criteria.
   Convert the request into explicit gates:
   - what must pass in PR CI
   - what must pass in nightly CI
   - what must pass before release
   - what is blocked vs waived
   - how waivers expire and how they are tracked

5. Produce an auditable output.
   Every recommendation or change must map to:
   - a specification item
   - a test or tool
   - a gate condition
   - a stated residual risk if not fully verified

## Mandatory baseline

Unless the request explicitly narrows scope, require this baseline for Rust crates:

- unit / integration / regression tests through `cargo test`
- property-based tests through `proptest`
- mutation testing through `cargo-mutants`
- feature-matrix checks through `cargo-hack`
- coverage reporting through `cargo-llvm-cov`
- doctests when public API docs exist
- compile-fail or UI tests through `trybuild` or `ui_test` when the crate has compile-time contracts

## Conditional tools

Require these when applicable:

- `proptest-state-machine` for stateful or order-dependent behavior
- `Kani` for bounded verification of critical invariants where applicable
- `Miri` when unsafe code or low-level memory behavior is involved
- `loom` when concurrency or atomics are involved
- `cargo-fuzz` for parsers, decoders, formatters, or hostile external input surfaces

## What to read before answering

Read the following files as needed:

- `references/operational-guideline.md` for the full policy
- `references/tool-applicability-matrix.md` for the decision matrix
- `assets/ci-gate-checklist.md` for gate language and checklist items
- `assets/test-report-template.md` for the reporting structure

## Output requirements

When producing a review, plan, or policy update, include:

- objective and scope
- specification source(s)
- selected mandatory tools
- selected conditional tools and why they apply
- CI gate conditions
- prohibited shortcuts or omissions
- unresolved or unverified areas
- residual risks
- a clear statement separating verified scope from unverified scope

## Recommended phrasing constraints

Prefer precise statements such as:

- "Verified in scope:"
- "Not yet verified:"
- "Blocked by:"
- "Waived until:"
- "Applies because the crate includes:"
- "Not applicable because the crate does not include:"

Avoid unqualified claims such as:

- "fully tested"
- "all good"
- "safe"
- "spec-complete"

## When asked to modify files

If the user asks for concrete repository changes:
- update policy and CI files consistently
- keep PR, nightly, and release gates separate
- add or update reporting templates when the policy introduces new obligations
- do not add tools without showing why they apply
- do not remove gates without documenting the risk tradeoff

## Completion standard

You are done only when the response or file change:
- identifies the applicable verification layers,
- maps them to tool choices,
- states the gate conditions,
- documents exceptions and risks,
- and avoids overstating assurance.


When working on this Rust project, always save the full Cargo output to a log file under the `log/` directory before investigating any build issues.

### Required procedure

1. Ensure the `log/` directory exists.
2. Run the relevant Cargo command and redirect **both stdout and stderr** to a log file under `log/`.
3. Do **not** rely only on terminal output. Use the saved log file as the primary source for analysis.
4. After the log file is created, investigate all **errors** first, then review **warnings**.
5. In your report:
   - identify the root cause of each error
   - identify the source location when possible
   - distinguish confirmed causes from hypotheses
   - propose concrete fixes in priority order
   - list warnings separately and state whether each warning is actionable or low priority

### Command examples

For build investigation:
```bash
mkdir -p log && cargo build > log/cargo.log 2>&1

For test investigation:

mkdir -p log && cargo test > log/cargo-test.log 2>&1
Analysis requirements
Read the log file completely before making conclusions.
Quote the relevant error and warning lines exactly when useful.
Do not guess if the log does not support the conclusion.
If multiple errors appear, determine whether later errors are cascading effects of an earlier failure.
Prioritize the earliest primary error that likely caused the rest.
After analyzing errors, summarize warnings and recommend whether they should be fixed now or later.
Output format

Provide results in this order:

Log file used
Primary error summary
Root cause analysis
Recommended fix steps
Warning review
Remaining uncertainties
=======