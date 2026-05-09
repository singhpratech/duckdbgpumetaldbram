#!/usr/bin/env bash
# sync.sh — coordinate parallel Claude Code instances via .sync/INSTANCE.md.
#
# Usage:
#   ./scripts/sync.sh status "what you're doing now"
#   ./scripts/sync.sh status --done           # mark current section done
#   ./scripts/sync.sh status --blocked-on <peer>
#   ./scripts/sync.sh check                   # show all peers' status
#   ./scripts/sync.sh whoami                  # print this instance's name
#
# Instance name = basename of the current working directory (the worktree).

set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/..")"

INSTANCE="${CLAUDE_INSTANCE:-$(basename "$PWD")}"
SYNC_DIR=".sync"
FILE="$SYNC_DIR/${INSTANCE}.md"
mkdir -p "$SYNC_DIR"

now() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
branch() { git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "(no branch)"; }

cmd="${1:-check}"; shift || true

case "$cmd" in
  whoami)
    echo "$INSTANCE"
    ;;

  status)
    if [ "${1:-}" = "--done" ]; then
        st="done"; what="(work complete)"; blocker="nothing"
        shift
    elif [ "${1:-}" = "--blocked-on" ]; then
        st="blocked"
        blocker="${2:-unspecified}"
        shift 2
        what="${1:-(blocked, see blocker)}"
    else
        st="in_progress"
        what="${*:-(no description)}"
        blocker="nothing"
    fi
    {
      echo "# ${INSTANCE}"
      echo "- branch: $(branch)"
      echo "- working on: ${what}"
      echo "- status: ${st}"
      echo "- blocked on: ${blocker}"
      echo "- last update: $(now)"
    } > "$FILE"
    echo "==> wrote $FILE"
    cat "$FILE"
    ;;

  check)
    echo "==> .sync/ snapshot ($(now))"
    if ! ls "$SYNC_DIR"/*.md >/dev/null 2>&1; then
        echo "  (no peer status files yet)"
        exit 0
    fi
    for f in "$SYNC_DIR"/*.md; do
        [ "$(basename "$f")" = "README.md" ] && continue
        echo
        echo "----- $f -----"
        cat "$f"
    done
    ;;

  *)
    echo "usage: $0 {status [--done | --blocked-on PEER | <text>] | check | whoami}" >&2
    exit 1
    ;;
esac
