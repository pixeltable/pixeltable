---
description: Address review comments on a Claude CI-fix PR, then commit and push one round of fixes
argument-hint: <PR URL>
allowed-tools: Read, Grep, Glob, Edit, Write, Bash(git:*), Bash(gh:*), Bash(python:*), Bash(python3:*), Bash(pytest:*), Bash(make:*), Bash(ruff:*), Bash(mypy:*), Bash(uv:*)
---

Address the review feedback on the CI-fix pull request. Arguments: $ARGUMENTS

The first argument is the PR's full URL. Follow
[`.claude/skills/address-the-review/SKILL.md`](../skills/address-the-review/SKILL.md) and its critic pass in
[`.claude/skills/address-the-review/references/critic.md`](../skills/address-the-review/references/critic.md).

This command runs unattended on a CI-fix PR. Where the skill says not to commit, this command wins: commit, push,
and reply. Do not merge. One round only.

## Policies

- Human actions always win. If a person (a GitHub `User` whose login does not end with `[bot]`) has commented,
  reviewed, closed, converted, or pushed to this PR, stop. Do not push over their work and do not re-open anything
  they closed.
- Do not push to a branch that is not this PR's head, and do not open a replacement PR.
- Do not invent Jira secrets or PXT tickets.
- Prefer deleting or simplifying code this PR introduced over adding flags, wrappers, or schema branches.

## Tools

- **Files:** Read, Grep, Glob, Edit, and Write in the working directory.
- **Shell:** `git`, `gh`, `python` / `python3`, `pytest`, `make`, `ruff`, `mypy`, and `uv`. No other programs.

The working directory is the default branch until you check out the PR's head with `gh pr checkout`. Fetch review
threads with `gh`. Do not run untrusted installers from the PR.

## Steps

1. Resolve the PR from `$ARGUMENTS`. If a person has already participated as above, stop with `skipped`.
2. Read the PR description, top-level reviews (`## Claude review`, `From Grok`, `From Muse`, Copilot, and human
   reviews), and unresolved inline threads. Cluster comments and run the first critic pass (`Revert`, `Redesign`,
   `Fix`, `Dismiss`, `Ask`).
3. Check out the PR branch. Apply changes in footprint order. Run the relevant tests and `make format` / `make check`
   on the changed files. Run the second critic pass.
4. Commit, push the PR branch, and reply to each thread with one sentence that cites `file:line`. Do not merge.
   Do not comment `agent review` again.

## Final message

End your turn with one fenced `json` block and nothing after it:

```json
{"action": "addressed", "pr": 123, "reason": "<one sentence>"}
```

`action` is `addressed` or `skipped`. `pr` is the PR number.
