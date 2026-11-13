---
description: Investigate a scheduled CI failure on main and open or update a fix PR when the standing policies allow it
argument-hint: <ci-fix-context.json>
allowed-tools: Read, Grep, Glob, Edit, Write, Bash(git:*), Bash(gh:*), Bash(python:*), Bash(python3:*), Bash(pytest:*), Bash(make:*), Bash(ruff:*), Bash(mypy:*), Bash(uv:*)
---

Investigate a scheduled CI failure and, when the policies below allow it, open a fix PR. The argument is
the path of a JSON context file written by the workflow: `$ARGUMENTS`

Read that file first. It names the failed run, the workflow (`tests`, `stress-tests`, or `nightly`), the ref the
run tested (`main` or `release`), the failed jobs, the branch to use, and closed unmerged CI-fix PRs for this
workflow. The workflow has already checked that no CI-fix PR for it is open.

## Method

The run ends when your turn ends. Launch every agent in the foreground and wait for its result. End your turn only
with the message that "Final message" describes. Do not merge. Do not re-file a ticket or PR a person closed.

Human actions always win. If a person closed, converted, or took over a PR or issue for this failure, leave it
alone. Aaron Siegel closed #1925 in favor of his #1927, and closed other agent fix PRs as won't-fix; do not
recreate those.

## Policies

- Investigate every failure. Do not dismiss a failure as flaky, intermittent, or a one-off, and do not re-run CI
  as a substitute for a root cause.
- `nightly` tests the `release` tag, not `main`. A fix merged to `main` does not clear nightly until the next
  release. If `main` already contains the fix, do not open a PR; say that a release is needed.
- For a root cause in catalog metadata, table versions, transactions, or concurrency control, including code under
  `pixeltable/catalog/` and `pixeltable/metadata/` and races that `tests/test_concurrent.py` exercises, investigate
  and flag for a human. Do not open a fix PR. Marcel closed #1906 for this reason.
- Do not push to a branch a person owns. Only push `branch` from the context file.
- Do not invent Jira secrets, Jira API calls, or PXT ticket numbers. There is no in-repo Jira integration. Flag
  with a GitHub issue when a human must take over, as step 3 describes.
- Keep the diff surgical. Prefer a one-line pin, env var, or skipped retired model over a compatibility shim.
  Follow `AGENTS.md`. Built-in UDFs must not mutate their arguments.

## Tools

- **Files:** Read, Grep, Glob, Edit, and Write in the working directory (the default branch).
- **Shell:** `git`, `gh`, `python` / `python3`, `pytest`, `make`, `ruff`, `mypy`, and `uv`. No other programs.

The working directory is the default branch. Nothing from a pull request runs here. Fetch logs with `gh`; do not
guess. Use `gh run view <id> --log-failed` and `gh api repos/<owner>/<repo>/actions/runs/<id>/jobs`.

## Steps

1. **Context.** Read the JSON file. Fetch the failed jobs and the failed log. List open and closed PRs and issues
   that name this run URL, the failed tests, or the closed PRs in the context file. Stop with `skipped` when a
   person already closed or took over the same failure, or when `main` already has the fix and this is nightly.
2. **Cause.** Name the failing tests or jobs, the first place the state goes wrong, and whether the cause is a
   product bug, a dependency change, a release lag, infra, or core catalog/transaction/concurrency.
3. **Decide.** Choose `opened`, `flagged`, or `skipped` before you edit. Flag means: do not open a fix PR, and put
   the investigation in a GitHub issue. When an open issue already covers the same failing test or job, comment on it
   with this run instead of opening another, unless a person took that issue over.
4. **Fix**, only for `opened`. Branch from the default branch onto `branch`. Make the smallest change
   that addresses the root cause. Add a regression test when the failure is in library code. Run the failing tests
   and `make format` / `make check` on the changed files.
5. **Publish.** Commit, push `branch`, and open a PR against the default branch. The PR body must include the run URL
   and, when this is nightly, that nightly tests `release`. A later run finds the PR by that run id. Do not comment
   on the PR: the workflow comments `agent review` and reviews it. Do not merge.

## Final message

End your turn with one fenced `json` block and nothing after it:

```json
{"action": "opened", "pr": 123, "reason": "<one sentence>"}
```

`action` is `opened`, `flagged`, or `skipped`. `pr` is the fix PR number, or `null` when there is none. The workflow
finds the PR by `branch`, not by this number.
