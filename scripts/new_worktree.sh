#!/usr/bin/env bash
# new_worktree.sh — spin up a new git worktree on a fresh branch.
#
# Usage:
#   ./scripts/new_worktree.sh feat/cuda-groupby
#   ./scripts/new_worktree.sh feat/core-bench-skew    # base on main by default
#   ./scripts/new_worktree.sh feat/cuda-foo --base feat/cuda-groupby
#
# Worktree path: ../worktrees/<branch-with-slashes-as-dashes>
# Each worktree is its own directory; you can `cd` into it and launch a
# separate Claude Code instance there. Instances coordinate via .sync/.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ $# -lt 1 ]; then
    echo "usage: $0 <branch-name> [--base <existing-branch>]" >&2
    exit 1
fi

BRANCH="$1"; shift
BASE="main"
while [ $# -gt 0 ]; do
    case "$1" in
        --base) BASE="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Path: place worktrees as siblings of the main checkout, not inside it
# (otherwise CMake build dirs would be visible from parent listings).
PARENT_DIR="$(dirname "$PWD")"
WT_BASE="$PARENT_DIR/worktrees"
mkdir -p "$WT_BASE"
SAFE_NAME="${BRANCH//\//-}"
WT_PATH="$WT_BASE/$SAFE_NAME"

if [ -d "$WT_PATH" ]; then
    echo "worktree already exists at $WT_PATH" >&2
    exit 1
fi

# Update remote refs (no-op if no remote)
git fetch --all --prune 2>/dev/null || true

# Create the branch + worktree in one shot
git worktree add -b "$BRANCH" "$WT_PATH" "$BASE"

echo
echo "==> worktree ready"
echo "    branch: $BRANCH (based on $BASE)"
echo "    path:   $WT_PATH"
echo
echo "Next:"
echo "  cd \"$WT_PATH\""
echo "  ./scripts/sync.sh status \"starting work on $BRANCH\""
echo "  # then launch claude code from this directory"
echo
echo "Worktrees managed:"
git worktree list
