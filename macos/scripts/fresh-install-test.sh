#!/usr/bin/env bash
# Test Whiz.app as if it were a brand-new install, then put everything back.
#
# A fresh machine differs from a development one in four places: the whisper
# models, the diarization models, the config file, and the TCC permission
# grants. This moves the first three aside and optionally resets the fourth.
#
# Nothing is ever deleted. State is MOVED to a stash directory and `restore`
# moves it back, because ~/.cache/whisper alone is a multi-gigabyte download
# that takes a long while to re-fetch.
#
#   fresh-install-test.sh status    what is stashed right now
#   fresh-install-test.sh stash     hide state, reset permissions, launch
#   fresh-install-test.sh restore   put everything back
set -euo pipefail

STASH="$HOME/.whiz-fresh-test-stash"
BUNDLE_ID="com.reidenxerx.whiz"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="$ROOT/build/Whiz.app"

PATHS=(
  "$HOME/.cache/whisper"      # speech + VAD models
  "$HOME/.cache/whiz"         # diarization models
  "$HOME/.config/whiz"        # config.toml and speaker profiles
)

# Stash entries are named from the whole path, not the basename. ~/.cache/whiz
# and ~/.config/whiz are both "whiz", so basenames collided: the second stash
# clobbered the first and restore silently put back only one of them.
slug() {
  printf '%s' "${1#"$HOME"/}" | tr '/' '_'
}

quit_app() {
  pkill -f "Whiz.app/Contents/MacOS/WhizApp" 2>/dev/null || true
}

case "${1:-status}" in
  status)
    echo "stash: $STASH"
    if [ -d "$STASH" ]; then
      echo "  ACTIVE — a fresh-install test is in progress"
      for p in "$STASH"/*; do
        [ -e "$p" ] && echo "    stashed: $(basename "$p") ($(du -sh "$p" | cut -f1))"
      done
      echo "  run 'restore' to put it back"
    else
      echo "  empty — live state is in place"
    fi
    echo
    echo "live state:"
    for p in "${PATHS[@]}"; do
      [ -e "$p" ] && echo "  $p ($(du -sh "$p" 2>/dev/null | cut -f1))" || echo "  $p (absent)"
    done
    ;;

  stash)
    if [ -d "$STASH" ]; then
      echo "error: a stash already exists at $STASH" >&2
      echo "       run 'restore' first — refusing to stack stashes and lose the older one." >&2
      exit 1
    fi
    quit_app
    mkdir -p "$STASH"
    for p in "${PATHS[@]}"; do
      if [ -e "$p" ]; then
        mv "$p" "$STASH/$(slug "$p")"
        echo "  stashed $p"
      fi
    done

    # TCC grants are keyed to the bundle id, so a "fresh" app still holds the
    # permissions the previous build was granted. Resetting them is what makes
    # the first-run permission flow testable.
    for service in Accessibility Microphone; do
      tccutil reset "$service" "$BUNDLE_ID" >/dev/null 2>&1 \
        && echo "  reset $service" \
        || echo "  could not reset $service (grant it manually to retest)"
    done

    echo
    echo "State hidden. Launching a 'first run'…"
    [ -d "$APP" ] && open "$APP" || echo "  (no build at $APP — run scripts/build-app.sh)"
    echo
    echo "Expect: no speech model, no diarization models, default settings,"
    echo "and permission prompts. When done: $0 restore"
    ;;

  restore)
    if [ ! -d "$STASH" ]; then
      echo "nothing stashed — live state is already in place"
      exit 0
    fi
    quit_app
    for p in "${PATHS[@]}"; do
      name="$(slug "$p")"
      if [ -e "$STASH/$name" ]; then
        # Anything the test run created is discarded in favour of the real
        # state; the point of the test was to observe, not to keep.
        [ -e "$p" ] && rm -rf "$p"
        mkdir -p "$(dirname "$p")"
        mv "$STASH/$name" "$p"
        echo "  restored $p"
      fi
    done
    rmdir "$STASH" 2>/dev/null || echo "  note: $STASH not empty, left in place"
    echo
    echo "Restored. Re-grant Accessibility if you reset it:"
    echo "  System Settings -> Privacy & Security -> Accessibility"
    ;;

  *)
    echo "usage: $0 [status|stash|restore]" >&2
    exit 1
    ;;
esac
