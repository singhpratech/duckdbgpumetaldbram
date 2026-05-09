# .sync — coordination between parallel Claude Code instances

Each running Claude Code instance writes to `<instance>.md` here.

## Convention
- One file per worktree: `.sync/<basename-of-worktree>.md`
- Each instance updates ONLY its own file (no merge conflicts ever)
- Each instance reads ALL files in this directory at start of work
- Commit your status updates frequently to your feature branch; rebase on main often

## File template
```
# <instance-name>
- branch: feat/<scope>-<topic>
- working on: <one-sentence what>
- status: in_progress | blocked | done
- blocked on: <other-instance> needs <thing>, or "nothing"
- last update: <ISO-8601 timestamp>
- notes: <free-form, multi-line ok>
```

## Usage from a Claude Code instance
```
./scripts/sync.sh status "implementing CUDA group-by hash table"
./scripts/sync.sh check       # see all peers
```

## When to read this directory
- At the start of every conversation
- Before touching shared code (anything outside your platform's backend dir)
- Before opening a PR — make sure no peer is mid-edit on the same files
