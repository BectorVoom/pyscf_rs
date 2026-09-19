#!/usr/bin/env bash
# Run one crate's named PBC oracle gates, and fail on anything that would make
# a green result vacuous.  Plan 20-03 (.planning/phases/20-pbc-python-bindings).
#
#   run.sh <crate> "<test target> ..." "<gate name> ..."
#
# The cargo line is exactly the local gate protocol (20-EXECUTION-NOTES §2):
#
#   cargo test -p <crate> --release --locked --no-fail-fast --test <t1> --test <t2> ... \
#       -- --ignored --exact --nocapture <gate1> <gate2> ...
#
# with CARGO_TARGET_DIR / CARGO_PROFILE_RELEASE_LTO / PYSCF_ORACLE_VENV taken
# from the environment.  Targets are always named (a bare `--tests` has been
# OOM-killed in this workspace); gates are selected by `--exact` name because a
# target binary can hold gates of several tiers.
#
# Checks, in order:
#   1. cargo's own exit status (a failed or panicking gate is non-zero);
#   2. the summed "N passed" over every binary equals the number of gate names
#      given -- a misspelt or renamed gate matches nothing under `--exact` and
#      would otherwise pass as "0 passed";
#   3. with PYSCF_ORACLE_VENV set, no gate printed one of the harness's
#      skip-when-unset lines (a skipped oracle gate also reports "ok").
# With PYSCF_ORACLE_VENV unset, the gates SKIP and exit 0 by contract
# (`oracle_python()` -> None); check 3 is then reported, not enforced.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <crate> \"<target> ...\" \"<gate> ...\"" >&2
  exit 2
fi
crate=$1
# Whitespace-separated lists; newlines allowed (YAML block scalars).
read -r -d '' -a targets <<<"$2" || true
read -r -d '' -a gates <<<"$3" || true
if [ "${#targets[@]}" -eq 0 ] || [ "${#gates[@]}" -eq 0 ]; then
  echo "run.sh: $crate: empty target or gate list" >&2
  exit 2
fi

: "${CARGO_TARGET_DIR:?CARGO_TARGET_DIR must be set (never under /tmp)}"
case "$CARGO_TARGET_DIR" in
  /tmp|/tmp/*) echo "run.sh: CARGO_TARGET_DIR=$CARGO_TARGET_DIR is under /tmp (RAM tmpfs); refusing" >&2; exit 2 ;;
esac

target_args=()
for t in "${targets[@]}"; do target_args+=(--test "$t"); done

log_dir="$CARGO_TARGET_DIR/pbc-oracle-logs"
mkdir -p "$log_dir"
log="$log_dir/${crate}.$(date +%Y%m%dT%H%M%S).log"

echo "run.sh: $crate: ${#targets[@]} target(s), ${#gates[@]} gate(s); PYSCF_ORACLE_VENV=${PYSCF_ORACLE_VENV-<unset>}; log $log"
set +e
cargo test -p "$crate" --release --locked --no-fail-fast "${target_args[@]}" \
  -- --ignored --exact --nocapture "${gates[@]}" 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
set -e
if [ "$rc" -ne 0 ]; then
  echo "::error title=pbc-oracle $crate::cargo test exited $rc"
  exit "$rc"
fi

passed=$(grep -Eo 'test result: ok\. [0-9]+ passed' "$log" | awk '{s += $4} END {print s + 0}')
if [ "$passed" -ne "${#gates[@]}" ]; then
  echo "::error title=pbc-oracle $crate::$passed gate(s) passed but ${#gates[@]} were named -- a gate name matched nothing"
  exit 1
fi

skips=$(grep -cE 'SKIP: |skip: set |unset — skipping' "$log" || true)
if [ -n "${PYSCF_ORACLE_VENV:-}" ]; then
  if [ "$skips" -ne 0 ]; then
    grep -nE 'SKIP: |skip: set |unset — skipping' "$log" || true
    echo "::error title=pbc-oracle $crate::$skips skip line(s) with PYSCF_ORACLE_VENV set -- a pass would be vacuous"
    exit 1
  fi
  echo "run.sh: $crate: OK -- $passed/${#gates[@]} gates passed, 0 skip lines"
else
  echo "run.sh: $crate: PYSCF_ORACLE_VENV unset -- $passed/${#gates[@]} gates reported ok, $skips skip line(s) (skip contract)"
fi
