#!/usr/bin/env bash
# Vendor sherpa-onnx's C API for the macOS app.
#
# Sources the dylibs + header from a sherpa_onnx *Python* installation — the
# same distribution `whiz transcribe` uses (`pipx inject whiz sherpa-onnx`,
# or this repo's .venv). That is deliberate: the app then diarizes with the
# exact binary the Python pipeline runs, so cluster boundaries agree, and
# there is no second build to keep in sync or pin.
#
# If you have no Python install handy, get one first:
#   pipx inject whiz sherpa-onnx        (or)   pip install sherpa-onnx
#
# Unlike whisper.cpp there is no source build here: sherpa-onnx's cmake needs
# onnxruntime, and vendoring the wheel's prebuilt dylibs sidesteps that
# entirely. The dylibs use @rpath install names, so the app links them and
# adds vendor/sherpa-onnx/lib to its rpath; build-app.sh is responsible for
# embedding them in the .app bundle (same signing story as the roadmap's
# distribution item).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$REPO_ROOT/macos/vendor/sherpa-onnx"
VERSION_EXPECTED="1.13"

if [ -f "$DEST/lib/libsherpa-onnx-c-api.dylib" ]; then
  echo "sherpa-onnx already vendored at $DEST — remove it to re-vendor."
  exit 0
fi

# Locate a sherpa_onnx package directory: explicit --from, this repo's .venv,
# or whatever python3 can import.
SRC=""
if [ "${1:-}" = "--from" ] && [ -n "${2:-}" ]; then
  SRC="$2"
elif [ -d "$REPO_ROOT/.venv" ]; then
  for d in "$REPO_ROOT"/.venv/lib/python*/site-packages/sherpa_onnx; do
    [ -d "$d" ] && SRC="$d" && break
  done
fi
if [ -z "$SRC" ] || [ ! -d "$SRC" ]; then
  SRC="$(python3 -c 'import sherpa_onnx, os; print(os.path.dirname(sherpa_onnx.__file__))' 2>/dev/null || true)"
fi
if [ -z "$SRC" ] || [ ! -d "$SRC" ]; then
  echo "error: no sherpa_onnx package found." >&2
  echo "  pipx inject whiz sherpa-onnx  (or)  pip install sherpa-onnx  (or)  $0 --from <site-packages>/sherpa_onnx" >&2
  exit 1
fi

if [ ! -f "$SRC/lib/libsherpa-onnx-c-api.dylib" ] || [ ! -f "$SRC/lib/libonnxruntime.dylib" ]; then
  echo "error: $SRC/lib is missing libsherpa-onnx-c-api.dylib or libonnxruntime.dylib" >&2
  exit 1
fi
HEADER="$SRC/include/sherpa-onnx/c-api/c-api.h"
if [ ! -f "$HEADER" ]; then
  echo "error: $HEADER missing — this wheel predates header shipping; install a newer sherpa-onnx" >&2
  exit 1
fi

# The wheel's dylibs must agree on onnxruntime's install name; sanity-check
# the pair we are about to copy actually link to each other.
DEPS="$(otool -L "$SRC/lib/libsherpa-onnx-c-api.dylib" | grep -c 'libonnxruntime.dylib' || true)"
if [ "$DEPS" -lt 1 ]; then
  echo "error: libsherpa-onnx-c-api.dylib does not reference libonnxruntime.dylib — unexpected wheel layout" >&2
  exit 1
fi

mkdir -p "$DEST/lib" "$DEST/include/sherpa-onnx/c-api"
cp "$SRC/lib/libsherpa-onnx-c-api.dylib" "$DEST/lib/"
cp "$SRC/lib/libonnxruntime.dylib" "$DEST/lib/"
cp "$HEADER" "$DEST/include/sherpa-onnx/c-api/"

echo "vendored sherpa-onnx ($SRC):"
ls "$DEST/lib" | sed 's|^|  lib/|'
echo "  include/sherpa-onnx/c-api/c-api.h"

# The Python pipeline's diarization models live under ~/.cache/whiz/diarization
# (`whiz models download-diarization`); nothing to vendor there, but warn if
# they are absent so the app's graceful skip does not surprise anyone.
if [ ! -d "$HOME/.cache/whiz/diarization" ]; then
  echo "note: no diarization models yet — run: whiz models download-diarization"
fi