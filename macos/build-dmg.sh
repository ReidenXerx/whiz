#!/bin/bash
#
# build-dmg.sh — pack WhizUI.app into a drag-to-install DMG.
#
# A volume that opens showing the app next to an Applications shortcut,
# drag once, done.
#
# FOR THIS MAC ONLY. The app inside is ad-hoc signed by build-app.sh — no
# Developer ID signature, no notarization, and the DMG itself is unsigned.
# Opened on another Mac, Gatekeeper quarantines it and refuses to launch the
# app. Shipping to other machines needs Developer ID signing + notarytool +
# stapling, which these scripts deliberately do not do. Runs
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

echo "Wrote $DMG — for THIS Mac only: ad-hoc signed, not notarized; Gatekeeper blocks it elsewhere."