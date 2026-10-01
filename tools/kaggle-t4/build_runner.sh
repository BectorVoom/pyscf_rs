#!/usr/bin/env bash
# Build the CUDA runner (`yta7o19_bands`) for Kaggle in ubuntu:22.04 (Kaggle's
# glibc), from a build bundle that carries the sibling crates and the
# cubecl-cpp 0.10 copy with F64 re-enabled (upstream disables it, so a stock
# build silently falls back to the CPU runtime).
#
#   BUNDLE=~/Documents/workspace/.yta_bundle tools/kaggle-t4/build_runner.sh
#
# Bundle layout: $BUNDLE/src/{pyscf_rs,cintx,cube-math,rmath,lapack_rs,libxc_rs,xcfun_rs,cubecl-cpp}
# (pyscf_rs/Cargo.toml there carries the [patch.crates-io] for cubecl-cpp),
# $BUNDLE/target-ml (build cache), $BUNDLE/cargo-home. The pyscf_rs crates are
# synced from this checkout first. Output: $BUNDLE/runner/yta7o19_bands.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
BUNDLE=${BUNDLE:?set BUNDLE to the build bundle directory}
[ -d "$BUNDLE/src/pyscf_rs" ] || { echo "no $BUNDLE/src/pyscf_rs"; exit 1; }
rsync -a --delete --exclude target "$REPO/crates/" "$BUNDLE/src/pyscf_rs/crates/"
rsync -a --delete "$REPO/pyscf/gto/" "$BUNDLE/src/pyscf_rs/pyscf/gto/"
rsync -a --delete "$REPO/pyscf/pbc/" "$BUNDLE/src/pyscf_rs/pyscf/pbc/"
T=/opt/rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin
podman run --rm --security-opt label=disable \
  -v "$BUNDLE/src:/work/src" -v "$BUNDLE/target-ml:/work/target" -v "$HOME/.rustup:/opt/rustup:ro" \
  -v "$BUNDLE/tracel:/root/.local/share/tracel" -v "$BUNDLE/cargo-home:/opt/cargo" -v "$HOME/.cargo/registry:/opt/cargo/registry" -v "$HOME/.cargo/git:/opt/cargo/git" \
  -e CARGO_HOME=/opt/cargo -e CARGO_TARGET_DIR=/work/target -e PATH=$T:/usr/local/bin:/usr/bin:/bin \
  -e DEBIAN_FRONTEND=noninteractive -w /work/src/pyscf_rs docker.io/library/ubuntu:22.04 \
  bash -c 'apt-get update -qq && apt-get install -y -qq build-essential pkg-config ca-certificates curl zlib1g-dev libzstd-dev >/dev/null && cargo build --offline --release -p pyscf-pbc-dft --features pyscf-algebra/cuda,pyscf-kernels/cuda --example yta7o19_bands -j 12' \
  > "$BUNDLE/build_runner.log" 2>&1 || { grep -E '^error' -A8 "$BUNDLE/build_runner.log" | head -40; echo "build FAILED (full log: $BUNDLE/build_runner.log)"; exit 1; }
grep -E 'Finished' "$BUNDLE/build_runner.log"
mkdir -p "$BUNDLE/runner"
cp "$BUNDLE/target-ml/release/examples/yta7o19_bands" "$BUNDLE/runner/"
strip "$BUNDLE/runner/yta7o19_bands"
sha256sum "$BUNDLE/runner/yta7o19_bands"
