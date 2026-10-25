# Anti-Spiral & Footprint Critic

Run this audit twice: once on the proposed triage plan (after Step 3) and once on the actual git diff (in Step 5). This critic strips code and prose slop; it never adds features.

## 1. Bot & Reviewer Skepticism Audit (Fix / Dismiss / Ask)

Check every item marked `Fix` or `Redesign` against these traps:

- **The "Implement Properly" Trap.** Did a reviewer or bot complain that a helper, fallback, state machine, or config path has an edge case? Check whether that mechanism is needed at all. If no hard requirement demands it, change the disposition from `Fix` to `Revert/Delete` rather than expanding the mechanism to handle the edge case.
- **The Unverified Bot Premise.** Did an automated reviewer (Claude, Grok, Muse, Copilot) claim a null dereference, race, or contract mismatch without a concrete trigger? Trace the caller boundary. If upstream validation or types already make that state unrepresentable, change the disposition to `Dismiss` and cite the boundary `file:line`.
- **Boundary Violation.** Does the proposed fix add a `try/except`, `isinstance` check, or null guard deep inside internal logic? Validate untrusted model JSON once at the record boundary in `claude_review_contract.py`, use `assert` for internal invariants, or delete the guard if the invariant already holds.
- **High-Risk Escalation.** Never auto-dismiss findings touching security, authentication, tenant isolation, billing, or the review history marker and resolution proof without either a concrete `file:line` disproof or asking the user (`Ask`).

## 2. Diff Footprint & Reader-Load Gate

Inspect `git diff <merge-base>...HEAD` and `git diff <pre-fix-head>...HEAD`:

- **Abstraction Check.** For every function or class touched by the review fixes, state what it does and owns in one sentence. If the fix added a mode flag, special-case branch, or pass-through parameter that muddies that sentence, redesign or revert.
- **Layer Count Check.** Did the review fixes introduce a new class, single-caller helper, configuration key, or compatibility shim? Inline or delete it.
- **Additive Spiral Circuit Breaker.** Did the fix round grow the PR by bolting new fields, schema rules, or fallback branches onto code introduced earlier in the same PR? Stop and run a forced-deletion pass (`challenge-the-design` Part 2): ask what part of the PR's new mechanism can be deleted so both the original bug and the review comment vanish.
- **Non-Convergence Gate.** Is this the second time this PR is patching the same function, schema, or invariant in response to review feedback? Stop writing patches. Present a two-option reset: (A) delete or revert the contested mechanism so the core change ships cleanly, or (B) replace the mechanism from first principles.

## 3. Code & Prose Deslop Pass

- Strip defensive checks, casts, or explanatory comments added only to appease a review thread. Keep comments only for constraints, non-obvious choices, or invariants (`AGENTS.md`).
- Audit all draft thread replies and summary prose against [`.claude/skills/articulate-the-abstraction/references/prose.md`](../../articulate-the-abstraction/references/prose.md):
  - Cut sycophancy ("Good catch", "You're right", "Thanks for pointing this out").
  - Use ASCII typography only (no em-dashes).
  - Lead with the action or disproof and cite `file:line` ("Required a prior evidence path in `validate_resolution()` at `.github/scripts/claude_review_contract.py:241`." or "Not applicable: `verify_sources()` rejects a symlink at `.github/scripts/claude_review_contract.py:186`.").
