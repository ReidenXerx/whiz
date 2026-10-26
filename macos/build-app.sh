#!/bin/bash
#
# build-app.sh — assemble Whiz.app from a release build.
#
# One command between "the tests pass" and "the app is in /Applications":
# builds the release binary, copies the bundle directory around it, puts the
# transcription engine inside (build-engine.sh: Python + whiz, whisper.cpp,
# whiz-media — so the app needs nothing installed) and Sparkle (in-app
# updates), stamps the version from pyproject.toml (one version for the
# project, not two), and codesigns every binary in it.
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
# Output: dist/Whiz.app (dist/ is gitignored — artifacts, not sources).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MACOS_DIR="$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST="$MACOS_DIR/dist"

# The Swift product (and module) stays WhizUI; what macOS shows — the
# bundle, its executable, Activity Monitor, the Dock — is "Whiz". The
# bundle identifier is unchanged, so settings and run history carry over.
PRODUCT="WhizUI"
APP_NAME="Whiz"
APP="$DIST/$APP_NAME.app"
CONTENTS="$APP/Contents"

VERSION="$(/usr/bin/awk -F '"' '/^version = / {print $2}' "$REPO_ROOT/pyproject.toml")"
[ -n "$VERSION" ] || { echo "build-app.sh: could not read version from pyproject.toml" >&2; exit 1; }

"$SCRIPT_DIR/build-engine.sh"
ENGINE="$MACOS_DIR/.engine/stage/engine"

echo "Building $APP_NAME $VERSION (release)…"
cd "$MACOS_DIR"
swift build -c release --disable-sandbox >/dev/null
BINARY="$MACOS_DIR/.build/release/$PRODUCT"
[ -f "$BINARY" ] || { echo "build-app.sh: no release binary at $BINARY" >&2; exit 1; }

rm -rf "$APP"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources" "$CONTENTS/Frameworks"

cp "$BINARY" "$CONTENTS/MacOS/$APP_NAME"
# The binary finds Sparkle through its rpaths; SwiftPM's point into .build
# (@loader_path, and an absolute path on the build Mac). In the bundle it
# lives in Frameworks/, and no rpath may point outside the app.
for rpath in $(otool -l "$CONTENTS/MacOS/$APP_NAME" | awk '/LC_RPATH/ {getline; getline; print $2}'); do
  case "$rpath" in /usr/lib/swift) ;; *) install_name_tool -delete_rpath "$rpath" "$CONTENTS/MacOS/$APP_NAME" ;; esac
done
install_name_tool -add_rpath @executable_path/../Frameworks "$CONTENTS/MacOS/$APP_NAME"
# ditto, not cp -R: a framework is symlinks (Versions/Current) that must
# stay symlinks for its signature to verify. The XPC services are for
# sandboxed apps only (Sparkle's docs); Whiz is not sandboxed.
SPARKLE="$CONTENTS/Frameworks/Sparkle.framework"
ditto "$MACOS_DIR/.build/release/Sparkle.framework" "$SPARKLE"
rm -rf "$SPARKLE/XPCServices" "$SPARKLE/Versions/B/XPCServices"
cp "$MACOS_DIR/bundle/Info.plist" "$CONTENTS/Info.plist"
# Resources/engine, not MacOS/: on a case-insensitive disk MacOS/whiz and
# MacOS/Whiz are one file. The app finds it there (WhizLocator.bundledEngine).
cp -R "$ENGINE" "$CONTENTS/Resources/engine"
# Whiz > Acknowledgements (build-engine.sh writes it beside the engine).
cp "$ENGINE/../Acknowledgements.json" "$CONTENTS/Resources/Acknowledgements.json"

# The icon ships in the bundle directory so a fresh clone has it; a missing
# icns is a warning, not a failure — the app still runs, it just looks
# generic (and the README points at build-icon.sh).
if [ -f "$MACOS_DIR/bundle/AppIcon.icns" ]; then
  cp "$MACOS_DIR/bundle/AppIcon.icns" "$CONTENTS/Resources/AppIcon.icns"
else
  echo "build-app.sh: warning — no AppIcon.icns in bundle/; run build-icon.sh" >&2
fi

# Stamp the version Info.plist references, into both keys: Sparkle decides
# "newer" by CFBundleVersion, so it must rise with every release, and the
# project version already does.
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" "$CONTENTS/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $VERSION" "$CONTENTS/Info.plist"

# --force because a previous build may have left a stale signature on a
# copied bundle. The engine needs no entitlements: the app runs it as child
# processes and reads the files they write, which the hardened runtime
# allows, and every library the bundled Python loads carries the same
# signature, which library validation requires. The app itself has one,
# the microphone (bundle/Whiz.entitlements), for Record.
#
# Inside out: each Mach-O in the engine first (the interpreter, its
# extension modules, onnxruntime, whisper-cli, whiz-media), the app last,
# sealing them. Notarization rejects any binary signed without the
# hardened runtime and a timestamp, so they get the same flags as the app.
IDENTITY="${WHIZ_SIGN_IDENTITY:-}"
if [ -n "$IDENTITY" ]; then
  SIGN=(codesign --force --options runtime --timestamp --sign "$IDENTITY")
else
  SIGN=(codesign --force --sign -)
fi
count=0
while IFS= read -r -d '' file; do
  if file -b "$file" | grep -q '^Mach-O'; then
    "${SIGN[@]}" "$file" 2>/dev/null || { echo "build-app.sh: could not sign $file" >&2; exit 1; }
    count=$((count + 1))
  fi
done < <(find "$CONTENTS/Resources/engine" -type f \( -perm -u+x -o -name '*.so' -o -name '*.dylib' \) -print0)
echo "Signed $count engine binaries."

# Sparkle, inside out as its docs order it: the installer tool and the
# Updater.app it launches, then the framework that seals them.
for part in "$SPARKLE/Versions/B/Autoupdate" "$SPARKLE/Versions/B/Updater.app" "$SPARKLE"; do
  "${SIGN[@]}" "$part" 2>/dev/null || { echo "build-app.sh: could not sign $part" >&2; exit 1; }
done

if [ -n "$IDENTITY" ]; then
  codesign --force --options runtime --timestamp --entitlements "$SCRIPT_DIR/bundle/Whiz.entitlements" --sign "$IDENTITY" "$APP"
  codesign --verify --strict "$APP"
  echo "Wrote $APP ($VERSION) — signed by \"$IDENTITY\" (hardened runtime)."
else
  codesign --force --entitlements "$SCRIPT_DIR/bundle/Whiz.entitlements" --sign - "$APP"
  codesign --verify "$APP"
  echo "Wrote $APP ($VERSION) — ad-hoc signed, runs on THIS Mac only (not notarized)."
fi