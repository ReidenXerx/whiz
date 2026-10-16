#!/bin/bash
#
# build-dmg.sh — pack WhizUI.app into a drag-to-install DMG.
#
# The artifact a user can hand to another Mac: a volume that opens showing
# the app next to an Applications shortcut, drag once, done. Runs
# build-app.sh first so the DMG always matches the tree, and verifies the
# app bundle exists because hdiutil on a stale DMG would ship an old app
# with a fresh name.
#
# Output: dist/WhizUI-<version>.dmg (dist/ is gitignored).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MACOS_DIR="$SCRIPT_DIR"
DIST="$MACOS_DIR/dist"

# Build first — the DMG must never be older than the .app it claims to hold.
"$SCRIPT_DIR/build-app.sh"

VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' \
  "$DIST/WhizUI.app/Contents/Info.plist")"
DMG="$DIST/WhizUI-$VERSION.dmg"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# The Applications symlink is what makes the volume "drag to install" —
# Finder resolves it in the copy, so dragging the app lands in /Applications.
cp -R "$DIST/WhizUI.app" "$STAGE/"
ln -s /Applications "$STAGE/Applications"

rm -f "$DMG"
hdiutil create -volname "WhizUI" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
hdiutil verify "$DMG" >/dev/null

echo "Wrote $DMG"