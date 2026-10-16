#!/bin/bash
#
# build-icon.sh — render the whiz mark into AppIcon.icns.
#
# No rsvg/inkscape/ImageMagick dependency: qlmanage rasterizes the SVG (the
# same renderer Finder uses for previews) and iconutil packs the PNGs. Every
# rendered PNG is verified at its expected pixel size and force-resampled
# with sips when QuickLook ignored the requested size — the sizes inside an
# iconset are a contract, iconutil rejects the set otherwise.
#
# Output: macos/bundle/AppIcon.icns (committed; small, stable, and it means
# a fresh clone ships the icon without needing this script to run first).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC="$REPO_ROOT/docs/assets/whiz-mark.svg"
OUT_ICNS="$SCRIPT_DIR/AppIcon.icns"
WORK="$(mktemp -d)"
ICONSET="$WORK/AppIcon.iconset"
trap 'rm -rf "$WORK"' EXIT

for tool in qlmanage iconutil sips; do
  command -v "$tool" >/dev/null 2>&1 || { echo "build-icon.sh: $tool not found" >&2; exit 1; }
done
[ -f "$SRC" ] || { echo "build-icon.sh: no mark at $SRC" >&2; exit 1; }

mkdir -p "$ICONSET"

# iconset filename -> pixel size. The @2x entries are the retina doublings
# of the row above them; every size the format can ask for, so the icns is
# crisp from the Dock at 16px to a 512pt About window.
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

  # qlmanage exits nonzero for trivia on some systems while still writing
  # the thumbnail, so its status is not trusted — the mv and the size
  # check below are the real failure signals.
  qlmanage -t -s "$size" -o "$WORK" "$SRC" >/dev/null 2>&1 || true
  mv "$WORK/whiz-mark.svg.png" "$ICONSET/$name"

  got="$(sips -g pixelWidth "$ICONSET/$name" | /usr/bin/awk '/pixelWidth/ {print $2}')"
  if [ "$got" != "$size" ]; then
    echo "build-icon.sh: QuickLook produced ${got}px for ${size}px, resampling with sips"
    sips -z "$size" "$size" "$ICONSET/$name" >/dev/null
  fi
done

iconutil -c icns "$ICONSET" -o "$OUT_ICNS"
echo "Wrote $OUT_ICNS"