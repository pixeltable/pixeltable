---
name: address-the-review
description: >
  Digest human and automated pull request review feedback (Claude, Grok, Muse,
  Copilot, and maintainer reviews) as an ensemble before editing code. Use when
  addressing PR review comments, bot findings, or requested changes to reassess
  the design, delete or simplify flawed abstractions, dismiss false positives
  with source evidence, and shrink the diff footprint instead of layering
  patch-on-patch fixes.
argument-hint: "[<PR URL or number>]"
---

# Address the review

Digest all open review feedback as a single architectural signal, verify every reviewer claim against the code, and converge on the smallest change that satisfies the requirement. Pair this workflow with [`.claude/skills/challenge-the-design/SKILL.md`](../challenge-the-design/SKILL.md) and [`.claude/skills/structure-review/SKILL.md`](../structure-review/SKILL.md).

Never treat review threads as a line-by-line coding checklist. Multiple comments on the same component are evidence that the current abstraction is wrong, not a mandate to add defensive branches, helpers, or configuration flags. Agreement is not the default. Prefer candor and deletion over sycophancy.

Preserve unrelated work. Group the verified code changes so they can be one commit. Do not commit, push, or post thread replies on GitHub unless the user asks.

## 1. Establish the original contract and review state

- Resolve the target PR (from `$ARGUMENTS` or the current branch) and record the merge base, any earlier reviewed head SHAs, the current `HEAD`, and the diff footprint (`git diff --stat <merge-base>...HEAD`).
- Read the PR title, description, and every review summary and unresolved inline thread before editing any file:
  - Top-level reviews and summary comments (`## Claude review`, `From Grok`, `From Muse`, Copilot, and human reviews): `gh pr view <PR> --json title,body,baseRefOid,headRefOid,reviews,comments`
  - Inline review threads and their resolution status (`isResolved`, `path`, `line`, `diffHunk`, `comments`): query `reviewThreads` via `gh api graphql` or `gh api repos/<owner>/<repo>/pulls/<pr>/comments`.
- A `## Claude review` is the model's review as GitHub stored it. There is no separate findings record to parse.
- If the PR has already gone through one or more review-fix commits, inspect `git log --oneline <first-reviewed-sha>..HEAD` and `git diff <first-reviewed-sha>..HEAD` to see where previous fix rounds added complexity.
- Separate the PR's core requirement from reviewer preferences, speculative edge cases, and out-of-scope requests. Hold all edits until every summary point and inline thread is mapped.

## 2. Cluster feedback and attack the premise

Group review points (both summary findings and inline threads) by the underlying data flow, state boundary, or abstraction they touch rather than by file order or reviewer.

- **Attack the premise.** When two or more comments, or a second round of review fixes, target the same helper, class, schema, or flow, write down the single premise that code assumes. Test whether that premise is needed at all before writing another patch to preserve it.
- **Treat reviewer claims as hypotheses to verify.** Automated reviewers (Claude, Grok, Muse, Copilot) and fast human passes can misread callers, miss upstream invariants, or suggest local patches that violate repository rules (`AGENTS.md`). Trace the actual callers and invariants first.
- **Challenge necessity.** Another block calling a new helper in your diff does not establish that the helper is necessary. If a flagged mechanism only exists to serve scaffolding introduced in this PR, evaluate deleting the mechanism and inlining the required guarantee.

## 3. Triage the batch: Revert, Redesign, Fix, Dismiss, or Ask

Classify every comment cluster into one of five dispositions before editing code. Apply [references/critic.md](references/critic.md) to challenge any disposition that adds lines or indirection.

- **Revert or delete.** The flagged code is dead weight, a speculative guard, a one-caller wrapper, or an unrequested implementation of something outside the ticket scope. Remove it completely.
- **Redesign from first principles.** The review exposed design-induced complexity, the wrong boundary, or scattered conditionals. Do not bolt a patch onto the broken shape. Revert the flawed slice and write the shape you would have built from scratch had this constraint been known on day one.
- **Fix root cause in place.** The comment identifies a reproducible defect or missing guarantee. Fix it at the first place the state goes wrong. A defensive guard is a valid fix only when the state it handles is legitimate at that boundary.
- **Dismiss with evidence.** The finding is a false positive, is already guaranteed by an upstream boundary, increases reader load without fixing a defect, or expands scope beyond the PR. Draft a concise technical reply citing the exact `file:line` and invariant.
- **Ask.** Human feedback is ambiguous, or reviewers conflict on a product or API contract. State the tradeoff in two sentences and ask before writing code that depends on the answer.

## 4. Subtract before you add

Execute changes in strict footprint order so deletions dissolve downstream comments automatically:

1. **Apply reverts and deletions first.** Strip rejected helpers, redundant validators, compatibility shims, and speculative branches. Mark remaining comments on deleted code as resolved by removal.
2. **Apply first-principles replacements second.** Use existing repository primitives and data structures instead of adding new layers, state fields, or flags.
3. **Apply local root-cause fixes last.** Keep the diff surgical and follow `AGENTS.md` and [`.claude/skills/structure-review/references/pixeltable-rules.md`](../structure-review/references/pixeltable-rules.md).
4. **Enforce reader-load compression.** Apply the central act of `structure-review`: after your fix, every modified class and function must still be statable in one or two crisp sentences. If a fix does not lower reader load or fix a verified defect, revert it.

## 5. Prove it works and audit the footprint

- Run targeted tests for the changed behavior first (starting with a failing reproduction for genuine defects), then run the repository's static checks (`make format`, `make check`, and relevant `pytest` or `unittest` suites per `AGENTS.md`).
- Compare the pre-fix diff against the post-fix diff (`git diff --stat <merge-base>...HEAD` and `git diff --stat <pre-fix-head>...HEAD`).
- Run the second pass of [references/critic.md](references/critic.md) on the resulting diff and replies.

## 6. Deliver the synthesis and thread replies

Present the review resolution in three parts:

- **Footprint delta.** Merge base SHA, previous head SHA, current state, and the net line/file delta (`before -> after`), highlighting what was deleted or collapsed.
- **Resolution ledger.** One row per comment cluster stating the source (reviewer/thread ID), disposition (`Reverted`, `Redesigned`, `Fixed`, `Dismissed`, `Question`), root cause, and exact `file:line` evidence.
- **Draft replies.** One-sentence declarative replies for each thread and summary point, stating what changed or the `file:line` disproof without performative agreement. When the user asks to post replies on GitHub, pass reply bodies as JSON via `gh api ... --input -` rather than shell interpolation.
