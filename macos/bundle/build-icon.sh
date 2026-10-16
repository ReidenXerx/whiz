#!/bin/bash
#
# build-icon.sh — render the Whiz icon into AppIcon.icns (and the README's PNG).
#
# The icon is drawn by render-icon.swift with CoreGraphics — geometry in one
# file, every size drawn rather than resampled, nothing to install (the old
# SVG path depended on qlmanage, which can hang without a GUI session).
# iconutil packs the PNGs; each is checked at its expected pixel size, since
# iconutil rejects a set whose sizes are off.
#
# Output: macos/bundle/AppIcon.icns and docs/assets/whiz-icon.png (both
# committed, so a fresh clone ships the icon without running this script).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RENDERER_SRC="$SCRIPT_DIR/render-icon.swift"
OUT_ICNS="$SCRIPT_DIR/AppIcon.icns"
OUT_PNG="$REPO_ROOT/docs/assets/whiz-icon.png"
WORK="$(mktemp -d)"
ICONSET="$WORK/AppIcon.iconset"
trap 'rm -rf "$WORK"' EXIT

for tool in swiftc iconutil sips; do
  command -v "$tool" >/dev/null 2>&1 || { echo "build-icon.sh: $tool not found" >&2; exit 1; }
done

swiftc -O -o "$WORK/render-icon" "$RENDERER_SRC"
mkdir -p "$ICONSET"

# iconset filename -> pixel size; the @2x entries are retina doublings.
pairs=(
  "icon_16x16.png 16"
  "icon_16x16@2x.png 32"
  "icon_32x32.png 32"
  "icon_32x32@2x.png 64"
  "icon_128x128.png 128"
  "icon_128x128@2x.png 256"
  "icon_256x256.png 256"
  "icon_256x256@2x.png 512"
  "icon_512x512.png 512"
  "icon_512x512@2x.png 1024"
)

for pair in "${pairs[@]}"; do
  name="${pair%% *}"
  size="${pair##* }"
  "$WORK/render-icon" "$ICONSET/$name" "$size"
  got="$(sips -g pixelWidth "$ICONSET/$name" | /usr/bin/awk '/pixelWidth/ {print $2}')"
  [ "$got" = "$size" ] || { echo "build-icon.sh: $name is ${got}px, expected ${size}px" >&2; exit 1; }
done

iconutil -c icns -o "$OUT_ICNS" "$ICONSET"
mkdir -p "$(dirname "$OUT_PNG")"
"$WORK/render-icon" "$OUT_PNG" 256
echo "Wrote $OUT_ICNS and $OUT_PNG"
