#!/bin/bash
#
# build-dmg.sh — pack WhizUI.app into a drag-to-install DMG.
#
# A volume that opens showing the app next to an Applications shortcut,
# drag once, done.
#
# By default the app is ad-hoc signed and the DMG is not: FOR THIS MAC
# ONLY — opened on another Mac, Gatekeeper refuses to launch the app.
#
# To ship to other Macs (needs Apple Developer Program membership):
#   1. a "Developer ID Application" certificate in the keychain — an "Apple
#      Development" one signs, but Apple will not notarize it;
#   2. notary credentials, stored once in the keychain:
#        xcrun notarytool store-credentials whiz-notary \
#          --apple-id <email> --team-id <TEAMID> --password <app-specific>
#   3. WHIZ_SIGN_IDENTITY="Developer ID Application: Name (TEAMID)" \
#      WHIZ_NOTARY_PROFILE=whiz-notary ./build-dmg.sh
# The app and the DMG are signed, the DMG is notarized and its ticket
# stapled, and Gatekeeper's verdict is checked before the script succeeds.
#
# Runs build-app.sh first (it reads WHIZ_SIGN_IDENTITY too) so the DMG
# always matches the tree.
#
# Output: dist/WhizUI-<version>.dmg (dist/ is gitignored).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MACOS_DIR="$SCRIPT_DIR"
DIST="$MACOS_DIR/dist"
IDENTITY="${WHIZ_SIGN_IDENTITY:-}"
PROFILE="${WHIZ_NOTARY_PROFILE:-}"

# Checked before building: a notarization bound to fail should not cost a
# release build first.
if [ -n "$PROFILE" ]; then
  case "$IDENTITY" in
    "Developer ID Application:"*) ;;
    "") echo "build-dmg.sh: WHIZ_NOTARY_PROFILE needs WHIZ_SIGN_IDENTITY (a Developer ID Application identity)" >&2
        exit 1 ;;
    *) echo "build-dmg.sh: Apple notarizes only Developer ID signatures, not \"$IDENTITY\"" >&2
       exit 1 ;;
  esac
fi

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

if [ -z "$IDENTITY" ]; then
  echo "Wrote $DMG — for THIS Mac only: ad-hoc signed, not notarized; Gatekeeper blocks it elsewhere."
  exit 0
fi

codesign --force --timestamp --sign "$IDENTITY" "$DMG"
codesign --verify "$DMG"
if [ -z "$PROFILE" ]; then
  echo "Wrote $DMG — signed by \"$IDENTITY\", not notarized (set WHIZ_NOTARY_PROFILE to notarize)."
  exit 0
fi

echo "Notarizing $DMG (waits for Apple, usually a few minutes)…"
xcrun notarytool submit "$DMG" --keychain-profile "$PROFILE" --wait
xcrun stapler staple "$DMG"
spctl --assess --type open --context context:primary-signature --verbose "$DMG"
echo "Wrote $DMG — signed, notarized and stapled; opens on any Mac."