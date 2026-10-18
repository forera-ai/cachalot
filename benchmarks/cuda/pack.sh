#!/usr/bin/env bash
# Mac side: pack the CUDA scaffold for a GPU host. Writes a REVISION file (commit and dirty flag) into the package,
# never into the working tree, and prints the archive path and its sha256. Nothing here touches the network.
#
#   cd /Users/hamedprooshani/Projects/deepseek-v41-mac && bash benchmarks/cuda/pack.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${CUDA_PACK_DIR:-$HOME/cachalot-private/cuda-packs}"
SHA="$(git -C "$REPO" rev-parse --short HEAD)"
DIRTY=0
if [ -n "$(git -C "$REPO" status --porcelain -- benchmarks/cuda tests/test_cuda_manifest.py tests/test_cuda_suite.py)" ]; then
  DIRTY=1
fi
if [ "$DIRTY" = 1 ]; then
  echo "WARNING: benchmarks/cuda has uncommitted changes; the package records dirty=1." >&2
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/benchmarks"
cp -R "$REPO/benchmarks/cuda" "$STAGE/benchmarks/cuda"
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} +
printf 'sha=%s\ndirty=%s\n' "$SHA" "$DIRTY" > "$STAGE/benchmarks/cuda/REVISION"

mkdir -p "$OUT_DIR"
ARCHIVE="$OUT_DIR/cuda-scaffold-$SHA.tar.gz"
tar -czf "$ARCHIVE" -C "$STAGE" benchmarks
echo "archive: $ARCHIVE"
shasum -a 256 "$ARCHIVE"
tar -tzf "$ARCHIVE"
