---
description: Triage and address PR review feedback without additive patch spirals
argument-hint: "[<PR URL or number>]"
---

Address the review feedback on a pull request. Arguments: $ARGUMENTS

Follow [`.claude/skills/address-the-review/SKILL.md`](../skills/address-the-review/SKILL.md) and its critic pass in [`.claude/skills/address-the-review/references/critic.md`](../skills/address-the-review/references/critic.md).

1. Resolve the target PR from `$ARGUMENTS` or the current branch. Preserve unrelated working-tree changes; use the branch's existing worktree when one is checked out.
2. Read the PR description, top-level reviews (`## Claude review`, `From Grok`, `From Muse`, Copilot, and human reviews), and unresolved inline review threads before editing any file. When the PR already has follow-up commits after an earlier review, inspect the diff since the earliest reviewed commit as well as the full diff against the merge base.
3. Cluster comments by mechanism and run the first pass of `references/critic.md` on the triage plan (`Revert`, `Redesign`, `Fix`, `Dismiss`, `Ask`). Prefer deleting or simplifying code introduced by the PR over adding flags, wrappers, or schema branches.
4. Apply changes in footprint order (deletions first, replacements second, root-cause fixes last), run the relevant tests and static checks (`make format`, `make check`), and run the second pass of `references/critic.md` on the resulting diff.
5. Present the footprint delta, resolution ledger, and concise draft replies. Do not commit, push, or post replies on GitHub unless the user explicitly asks.
