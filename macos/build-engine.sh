#!/bin/bash
#
# build-engine.sh — the transcription engine Whiz.app carries inside itself.
#
# A buyer installs nothing: no Homebrew, Python, pipx, whisper.cpp or
# ffmpeg. This script assembles, under .engine/stage/engine/:
#
#   bin/whiz          launcher: the bundled Python running the whiz package
#   bin/whisper-cli   whisper.cpp from a pinned release (Metal, static)
#   bin/whiz-media    audio extraction and video frames on AVFoundation, in
#                     place of ffmpeg (see Sources/WhizMedia)
#   python/           a relocatable CPython (python-build-standalone, via uv)
#                     with whiz and the pinned engine-requirements.txt
#   licenses/         the licence of every bundled component
#
# build-app.sh copies it into Contents/Resources/engine and signs every
# binary in it. Models are not bundled: the setup assistant downloads them.
#
# Build-time needs (the buyer's Mac needs none of them): the Xcode command
# line tools, cmake, uv and the network. Downloads and the whisper.cpp
# build are cached in .engine/ (gitignored), so a rebuild takes seconds.
#
# Apple silicon only: whisper.cpp runs on the GPU through Metal there, and
# the bundled Python and onnxruntime are arm64 builds.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MACOS_DIR="$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE="$MACOS_DIR/.engine"
STAGE="$CACHE/stage/engine"

# Pins. Changing one is a deliberate act: bump it, rebuild, run
# check-engine.sh. The whisper.cpp hash is of the release tarball.
WHISPER_VERSION="1.9.5"
WHISPER_SHA256="ff1a9053feb509ff9d7729703355541ae9690073a6b1c40eb692c962e0dc1720"
PYTHON_VERSION="3.13.13"
PY="python3.13"
REQUIREMENTS="$MACOS_DIR/bundle/engine-requirements.txt"

[ "$(uname -m)" = "arm64" ] || { echo "build-engine.sh: needs an Apple silicon Mac" >&2; exit 1; }
for tool in cmake uv; do
  command -v "$tool" >/dev/null || { echo "build-engine.sh: $tool is needed to build the engine (brew install $tool)" >&2; exit 1; }
done
mkdir -p "$CACHE"

# --- whisper.cpp -----------------------------------------------------------
# Static (no dylibs to sign or locate) with the Metal shaders embedded in
# the binary. GGML_NATIVE=OFF: -mcpu=native would emit instructions the
# build Mac has and an older M1 does not. macOS 13.3, not 13.0: ggml's
# Accelerate backend calls the cblas_sgemm that 13.3 introduced, so that
# is Whiz.app's floor too (LSMinimumSystemVersion in bundle/Info.plist).
WHISPER_SRC="$CACHE/whisper.cpp-$WHISPER_VERSION"
WHISPER_BUILD="$CACHE/whisper-build-$WHISPER_VERSION-macos13.3"
WHISPER_BIN="$WHISPER_BUILD/bin/whisper-cli"
if [ ! -x "$WHISPER_BIN" ]; then
  tarball="$CACHE/whisper.cpp-$WHISPER_VERSION.tar.gz"
  if [ ! -f "$tarball" ]; then
    echo "Downloading whisper.cpp ${WHISPER_VERSION}…"
    curl -fsSL -o "$tarball.part" \
      "https://github.com/ggml-org/whisper.cpp/archive/refs/tags/v$WHISPER_VERSION.tar.gz"
    mv "$tarball.part" "$tarball"
  fi
  echo "$WHISPER_SHA256  $tarball" | shasum -a 256 -c --quiet - \
    || { echo "build-engine.sh: whisper.cpp tarball does not match its pinned hash" >&2; rm -f "$tarball"; exit 1; }
  rm -rf "$WHISPER_SRC" && mkdir -p "$WHISPER_SRC"
  tar -xzf "$tarball" -C "$WHISPER_SRC" --strip-components 1
  echo "Building whisper.cpp $WHISPER_VERSION (a few minutes, once)…"
  cmake -S "$WHISPER_SRC" -B "$WHISPER_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=13.3 \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_SERVER=OFF \
    -DWHISPER_BUILD_EXAMPLES=ON >/dev/null
  cmake --build "$WHISPER_BUILD" --config Release \
    --target whisper-cli -j "$(sysctl -n hw.ncpu)" >/dev/null
fi

# --- whiz-media --------------------------------------------------------------
echo "Building whiz-media…"
(cd "$MACOS_DIR" && swift build -c release --disable-sandbox --product whiz-media >/dev/null)
MEDIA_BIN="$MACOS_DIR/.build/release/whiz-media"

# --- Python ------------------------------------------------------------------
PY_DOWNLOAD="$CACHE/python"
PY_SRC="$PY_DOWNLOAD/cpython-$PYTHON_VERSION-macos-aarch64-none"
if [ ! -x "$PY_SRC/bin/$PY" ]; then
  echo "Downloading Python ${PYTHON_VERSION}…"
  uv python install "$PYTHON_VERSION" --install-dir "$PY_DOWNLOAD" --no-bin >/dev/null 2>&1
fi

echo "Assembling the engine…"
rm -rf "$CACHE/stage" && mkdir -p "$STAGE/bin" "$STAGE/licenses"
cp -R "$PY_SRC" "$STAGE/python"
PYTHON="$STAGE/python/bin/$PY"
# uv marks its Pythons as managed; this copy is ours alone.
rm -f "$STAGE/python/lib/$PY/EXTERNALLY-MANAGED"

uv pip install --quiet --python "$PYTHON" --require-hashes -r "$REQUIREMENTS"
uv pip install --quiet --python "$PYTHON" --no-deps "$REPO_ROOT"

# Licences, before trimming removes the dist-info directories they sit in.
cp "$STAGE/python/lib/$PY/LICENSE.txt" "$STAGE/licenses/Python.txt"
cp "$WHISPER_SRC/LICENSE" "$STAGE/licenses/whisper.cpp.txt"
for info in "$STAGE/python/lib/$PY/site-packages/"*.dist-info; do
  name="$(basename "$info" .dist-info)"; name="${name%-*}"
  [ "$name" = "pip" ] && continue
  found="$(find "$info" \( -iname 'LICEN[CS]E*' -o -iname 'COPYING*' \) -type f | head -1)"
  [ -n "$found" ] && cp "$found" "$STAGE/licenses/$name.txt"
done

# Trim what a transcription never touches: pip (nothing installs into a
# signed bundle), Tcl/Tk and the GUI-only stdlib, tests, headers, the
# static libpython, and every console script but the interpreter itself.
SITE="$STAGE/python/lib/$PY/site-packages"
rm -rf "$SITE"/pip "$SITE"/pip-*.dist-info
rm -rf "$STAGE/python/include" "$STAGE/python/share" "$STAGE/python/lib/pkgconfig"
rm -rf "$STAGE/python/lib/"{tcl9*,tk9*,itcl*,thread*,libtcl*,libtk*}
rm -rf "$STAGE/python/lib/$PY/"{test,idlelib,tkinter,turtledemo,ensurepip,__phello__}
rm -rf "$STAGE/python/lib/$PY/config-"*
rm -f "$STAGE/python/lib/$PY/lib-dynload/"_tkinter*
find "$STAGE/python/bin" -mindepth 1 ! -name "$PY" -delete

# Bytecode compiled now, never at run time: a Python writing .pyc files
# into a signed app breaks its seal. unchecked-hash pycs are trusted as-is,
# so copying the tree (new mtimes) does not make them stale.
find "$STAGE/python" -name __pycache__ -type d -prune -exec rm -rf {} +
"$PYTHON" -I -B -m compileall -q -f -j 0 --invalidation-mode unchecked-hash "$STAGE/python/lib/$PY" >/dev/null

cp "$WHISPER_BIN" "$STAGE/bin/whisper-cli"
cp "$MEDIA_BIN" "$STAGE/bin/whiz-media"

# The launcher. -I ignores PYTHONPATH, PYTHONHOME and the user's site
# packages, so nothing installed on the Mac can leak into the engine; -B
# never writes bytecode. bin/ goes first on PATH so the bundled
# whisper-cli and whiz-media are the ones whiz finds.
cat > "$STAGE/bin/whiz" <<'EOF'
#!/bin/sh
# whiz, as bundled in Whiz.app — see macos/build-engine.sh.
here="$(cd "$(dirname "$0")" && pwd -P)"
PATH="$here:$PATH" WHIZ_BUNDLED=1 exec "$here/../python/bin/python3.13" -I -B -m whiz "$@"
EOF
chmod 755 "$STAGE/bin/whiz"

echo "Engine ready: $STAGE ($(du -sh "$STAGE" | cut -f1))"
