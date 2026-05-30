## Mandatory Manual for `cubecl` Implementation

When implementing, modifying, or generating code that uses the Rust `cubecl` crate, the agent must first read:

`/home/user/workspace/pyscf_rs/docs/manual/Cubecl`

This manual must be used as the primary reference for implementation patterns, architecture, configuration, and coding rules related to `cubecl`.

Do not write or propose `cubecl`-based code without consulting this manual first.


## Conventions

- Before creating any test code, read `\home\chemtech\workspace\libxc_rs\docs\rust_crate_test_guideline.md` and follow it when designing and implementing the tests.

When working on this Rust project, always save the full Cargo output to a log file under the `log/` directory before investigating any build issues.
