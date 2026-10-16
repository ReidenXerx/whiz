#!/bin/bash
#
# build-app.sh — assemble WhizUI.app from a release build.
#
# One command between "the tests pass" and "the app is in /Applications":
# builds the release binary, copies the bundle directory around it, stamps
# the version from pyproject.toml (one version for the project, not two),
# and codesigns.
#
# Signing:
#   default             ad-hoc (`codesign --sign -`), no identity needed.
#                       FOR THIS MAC ONLY: copied to another Mac (AirDrop, a
#                       download) the app is quarantined and Gatekeeper
#                       blocks it as "damaged" or "unidentified developer".
#   WHIZ_SIGN_IDENTITY  a keychain identity, e.g. "Developer ID Application:
#                       Name (TEAMID)": signs with the hardened runtime and a
#                       secure timestamp, as notarization requires.
#                       Notarizing is build-dmg.sh's job — the DMG ships.
#
# Output: dist/WhizUI.app (dist/ is gitignored — artifacts, not sources).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MACOS_DIR="$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST="$MACOS_DIR/dist"

APP_NAME="WhizUI"
APP="$DIST/$APP_NAME.app"
CONTENTS="$APP/Contents"

VERSION="$(/usr/bin/awk -F '"' '/^version = / {print $2}' "$REPO_ROOT/pyproject.toml")"
[ -n "$VERSION" ] || { echo "build-app.sh: could not read version from pyproject.toml" >&2; exit 1; }

echo "Building $APP_NAME $VERSION (release)…"
cd "$MACOS_DIR"
swift build -c release --disable-sandbox >/dev/null
BINARY="$MACOS_DIR/.build/release/$APP_NAME"
[ -f "$BINARY" ] || { echo "build-app.sh: no release binary at $BINARY" >&2; exit 1; }

rm -rf "$APP"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources"

cp "$BINARY" "$CONTENTS/MacOS/$APP_NAME"
cp "$MACOS_DIR/bundle/Info.plist" "$CONTENTS/Info.plist"

# The icon ships in the bundle directory so a fresh clone has it; a missing
# icns is a warning, not a failure — the app still runs, it just looks
# generic (and the README points at build-icon.sh).
if [ -f "$MACOS_DIR/bundle/AppIcon.icns" ]; then
  cp "$MACOS_DIR/bundle/AppIcon.icns" "$CONTENTS/Resources/AppIcon.icns"
else
  echo "build-app.sh: warning — no AppIcon.icns in bundle/; run build-icon.sh" >&2
fi

# Stamp the version Info.plist references. CFBundleVersion stays a build
# counter the script does not manage — for a local artifact "1" is honest.
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" "$CONTENTS/Info.plist"

# --force because a previous build may have left a stale signature on a
# copied bundle. No entitlements: the app only runs whiz as a child process
# and reads the files whiz writes, which the hardened runtime allows.
IDENTITY="${WHIZ_SIGN_IDENTITY:-}"
if [ -n "$IDENTITY" ]; then
  codesign --force --options runtime --timestamp --sign "$IDENTITY" "$APP"
  codesign --verify --strict "$APP"
  echo "Wrote $APP ($VERSION) — signed by \"$IDENTITY\" (hardened runtime)."
else
  codesign --force --sign - "$APP"
  codesign --verify "$APP"
  echo "Wrote $APP ($VERSION) — ad-hoc signed, runs on THIS Mac only (not notarized)."
fi