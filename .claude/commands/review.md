---
description: Review a pull request for bugs and CLAUDE.md violations, then post inline comments
argument-hint: <owner>/<repo>/pull/<number>
allowed-tools: Bash(gh pr view:*), Bash(gh pr diff:*), Bash(gh pr comment:*), mcp__github_inline_comment__create_inline_comment
---

Review pull request $ARGUMENTS and post the result on GitHub.

## Where the code is

- The working directory holds the base branch: the code before this PR.
- `pr-head/` holds the PR at its head commit. Read it for the full content of changed files. Never run, install, or
  build anything from it.
- `gh pr diff` shows which lines the PR changes.

If `pr-head/` does not exist, the working directory may already be the PR branch. Use `gh pr diff` to tell changed
lines from unchanged ones.

## Rules for every agent

Give each agent you launch the PR title, the PR description, and these rules:

- Every tool works. Do not probe tools, and make no call without a purpose.
- Flag an issue only when you are certain it is real. A false positive costs more than a missed issue.

## Steps

Create a todo list before you start.

1. **Gate.** Launch a haiku agent to read `gh pr view <PR> --comments`. Stop without posting anything if the PR is
   closed, is a draft, already has a review comment from Claude, or changes only content that needs no review (a
   version bump, a regenerated lockfile). Review PRs that Claude wrote.

2. **Rules.** Launch a haiku agent to return the paths, not the contents, of:
   - the root `CLAUDE.md`;
   - each `CLAUDE.md` in the directory of a changed file or in one of its parents;
   - each file those `CLAUDE.md` files import with `@path`, such as `@AGENTS.md`.

   Read these from the working directory, not from `pr-head/`, so the rules come from the base branch, which the PR
   cannot change.

3. **Summary.** Launch a sonnet agent to read `gh pr view` and `gh pr diff` and return what the PR changes and why.

4. **Review.** Launch four agents in parallel. Each returns a list of issues. Each issue names the file, the line
   range at the PR head, what is wrong, and its category: `rules` or `bug`.
   - Agents 1 and 2 (sonnet), independently: check the changed lines against the rules from step 2. A rule applies to a
     file only when its `CLAUDE.md` is in that file's directory or a parent. Quote the exact rule each issue breaks.
   - Agent 3 (opus): read only the diff. Flag bugs the diff alone proves: code that will not parse or type-check, a
     missing import, an undefined name, logic that is wrong for every input.
   - Agent 4 (opus): read the changed code in `pr-head/` with its surrounding context. Flag wrong logic, broken error
     handling, and security holes that this PR introduces.

5. **Validate.** For each issue, launch a subagent in parallel (opus for `bug`, sonnet for `rules`). Give it the issue
   and have it check the issue against `pr-head/` and the diff. It confirms the issue only when the code shows the
   issue is real and the PR changed those lines. For `rules`, the quoted rule must also exist and apply to that file.
   Drop every issue it does not confirm.

6. **Post.**
   - No confirmed issues: post one comment with `gh pr comment`, then stop:

     ```
     ## Claude review

     No issues found. Checked the diff for bugs and for `CLAUDE.md` rule violations.
     ```

   - Confirmed issues: post one inline comment per issue with `mcp__github_inline_comment__create_inline_comment` and
     `confirmed: true`. Say what is wrong and why. Add a suggestion block only when committing it fixes the issue
     completely and it spans fewer than 6 lines; otherwise describe the fix. Never post the same issue twice.

## Do not flag

- Problems that existed before this PR
- Style, naming, or formatting, or anything a linter or type checker catches (do not run them to check)
- Missing tests, unless a `CLAUDE.md` rule requires them
- Problems that depend on particular inputs or state you cannot show
- Code that breaks a rule but is explicitly exempted, for example by a lint ignore comment
- Nitpicks a senior engineer would skip

## Links

Cite each issue with a link to the code at the PR head commit. Get the full SHA from
`gh pr view <PR> --json headRefOid`; a shortened SHA or a shell expression does not render as a code preview.

`https://github.com/<owner>/<repo>/blob/<full-sha>/<path>#L<start>-L<end>`

Include one line of context on each side of the flagged lines. When an issue breaks a rule, link to its `CLAUDE.md`
the same way.

Use the `gh` CLI for all GitHub access. Do not use web fetch.
