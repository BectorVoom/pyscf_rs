# Deferred items — quick task 260601-rhc

Out-of-scope discoveries during F-05 execution (NOT fixed; not caused by this task's changes):

- **Pre-existing clippy `doc_lazy_continuation` warning** in `crates/pyscf-gto/tests/spinor_intor.rs:7`
  (`//!   2e (\`int2e_spinor\`, ...)` doc list item without indentation). Unrelated to the ECP
  iprinv work; SCOPE BOUNDARY — left untouched. Trivial fix: indent the continuation line by 2.
</content>
</invoke>
