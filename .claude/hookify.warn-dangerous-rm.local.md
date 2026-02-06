---
name: warn-dangerous-rm
enabled: true
event: bash
pattern: rm\s+(-[a-zA-Z]*r[a-zA-Z]*\s+-[a-zA-Z]*f[a-zA-Z]*|(-[a-zA-Z]*f[a-zA-Z]*\s+)?-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*|-rf)
action: block
---

**Dangerous rm command detected!**

You are about to run an `rm -rf` command which recursively and forcefully deletes files.

**Before proceeding, please verify:**
- The path is correct
- No important files will be deleted
- The directory is not a critical system or project directory

**User requested to be warned before using rm -rf commands.**

Consider alternatives:
- `rm -ri` for interactive removal
- `trash-cli` to move to trash instead
- Preview with `ls` first
