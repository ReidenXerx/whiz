#!/bin/bash
#
# publish-update.sh — offer a built DMG to every installed Whiz.
#
# Signs the DMG with the update key (Sparkle's EdDSA; the app refuses an
# update without a valid signature) and adds it to the appcast the app
# reads, docs/updates/v1.xml, served at https://duduphudu.app/whiz/updates/v1.xml.
#
#   ./build-dmg.sh                                # notarized (see build-dmg.sh)
#   ./publish-update.sh dist/Whiz-0.15.0.dmg --notes notes.html
#   gh release create v0.15.0 dist/Whiz-0.15.0.dmg   # the URL the item points at
#   commit docs/updates/v1.xml and merge: GitHub Pages publishes it, and
#   copies of Whiz find the update on their next check
#
# The update key lives in the login keychain (account dev.reidenxerx.whiz),
# with a backup at ~/.config/whiz-sparkle/update-signing.key. Lose both and
# installed copies can no longer be updated. Restore the backup with
#   .build/artifacts/sparkle/Sparkle/bin/generate_keys --account dev.reidenxerx.whiz -f <file>
#
# Options:
#   --notes FILE   HTML shown in the update dialog (a fragment: <ul><li>…)
#   --feed FILE    the appcast to add to (default docs/updates/v1.xml)
#   --url URL      where the DMG will be downloaded from (default: the
#                  GitHub release asset for the version's tag)
#
# Only a notarized DMG is published: an update a buyer's Mac then refuses to
# open would strand every copy. WHIZ_ALLOW_UNNOTARIZED=1 lifts that, for a
# test feed only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TOOLS="$SCRIPT_DIR/.build/artifacts/sparkle/Sparkle/bin"
ACCOUNT="dev.reidenxerx.whiz"

FEED="$REPO_ROOT/docs/updates/v1.xml"
URL=""
NOTES=""
DMG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --feed) FEED="$2"; shift 2 ;;
    --url) URL="$2"; shift 2 ;;
    --notes) NOTES="$2"; shift 2 ;;
    -*) echo "publish-update.sh: unknown option $1" >&2; exit 2 ;;
    *) DMG="$1"; shift ;;
  esac
done
[ -n "$DMG" ] && [ -f "$DMG" ] || { echo "usage: publish-update.sh <Whiz-x.y.z.dmg> [--notes FILE] [--feed FILE] [--url URL]" >&2; exit 2; }
[ -f "$FEED" ] || { echo "publish-update.sh: no appcast at $FEED" >&2; exit 1; }
[ -z "$NOTES" ] || [ -f "$NOTES" ] || { echo "publish-update.sh: no notes file at $NOTES" >&2; exit 1; }
[ -x "$TOOLS/sign_update" ] || (cd "$SCRIPT_DIR" && swift package resolve --disable-sandbox >/dev/null)

if [ "${WHIZ_ALLOW_UNNOTARIZED:-}" != "1" ]; then
  xcrun stapler validate -q "$DMG" \
    || { echo "publish-update.sh: $DMG is not notarized (build it with WHIZ_NOTARY_PROFILE)" >&2; exit 1; }
fi

# What the DMG holds, read from the app inside it: the version the feed
# announces must be the version that installs.
MOUNT="$(mktemp -d)"
trap 'hdiutil detach -quiet "$MOUNT" 2>/dev/null || true; rmdir "$MOUNT" 2>/dev/null || true' EXIT
hdiutil attach -quiet -nobrowse -readonly -mountpoint "$MOUNT" "$DMG"
PLIST="$MOUNT/Whiz.app/Contents/Info.plist"
[ -f "$PLIST" ] || { echo "publish-update.sh: no Whiz.app in $DMG" >&2; exit 1; }
read_key() { /usr/libexec/PlistBuddy -c "Print :$1" "$PLIST"; }
VERSION="$(read_key CFBundleShortVersionString)"
BUILD="$(read_key CFBundleVersion)"
MIN_SYSTEM="$(read_key LSMinimumSystemVersion)"
APP_FEED="$(read_key SUFeedURL)"
hdiutil detach -quiet "$MOUNT"

case "$APP_FEED" in
  */updates/v1.xml) ;;
  *) echo "publish-update.sh: this Whiz reads $APP_FEED, not the v1 feed — a new major version gets its own feed" >&2; exit 1 ;;
esac
if grep -q "<sparkle:version>$BUILD</sparkle:version>" "$FEED"; then
  echo "publish-update.sh: $FEED already offers $BUILD" >&2; exit 1
fi

URL="${URL:-https://github.com/ReidenXerx/whiz/releases/download/v$VERSION/$(basename "$DMG")}"
SIGNATURE="$("$TOOLS/sign_update" --account "$ACCOUNT" "$DMG")"   # sparkle:edSignature="…" length="…"
case "$SIGNATURE" in
  sparkle:edSignature=*) ;;
  *) echo "publish-update.sh: sign_update failed: $SIGNATURE" >&2; exit 1 ;;
esac

ITEM="$(mktemp)"
trap 'rm -f "$ITEM"; hdiutil detach -quiet "$MOUNT" 2>/dev/null || true; rmdir "$MOUNT" 2>/dev/null || true' EXIT
{
  echo "    <item>"
  echo "      <title>Whiz $VERSION</title>"
  echo "      <pubDate>$(LC_ALL=C date -u '+%a, %d %b %Y %H:%M:%S +0000')</pubDate>"
  echo "      <sparkle:version>$BUILD</sparkle:version>"
  echo "      <sparkle:shortVersionString>$VERSION</sparkle:shortVersionString>"
  echo "      <sparkle:minimumSystemVersion>$MIN_SYSTEM</sparkle:minimumSystemVersion>"
  if [ -n "$NOTES" ]; then
    echo "      <description><![CDATA["
    cat "$NOTES"
    echo "      ]]></description>"
  fi
  echo "      <enclosure url=\"$URL\" type=\"application/x-apple-diskimage\" $SIGNATURE />"
  echo "    </item>"
} > "$ITEM"

# Newest first, right under the marker; Sparkle picks the highest version
# wherever it sits, people reading the file do not.
grep -q '<!-- releases, newest first -->' "$FEED" \
  || { echo "publish-update.sh: $FEED has no '<!-- releases, newest first -->' marker" >&2; exit 1; }
awk -v item="$ITEM" '
  { print }
  /<!-- releases, newest first -->/ { while ((getline line < item) > 0) print line }
' "$FEED" > "$FEED.new"
xmllint --noout "$FEED.new" || { rm -f "$FEED.new"; echo "publish-update.sh: the appcast would not be valid XML" >&2; exit 1; }
mv "$FEED.new" "$FEED"

echo "Added Whiz $VERSION to $FEED"
echo "  download: $URL"
echo "Next: upload $(basename "$DMG") to that URL (gh release create v$VERSION $DMG), then commit and merge the feed."
