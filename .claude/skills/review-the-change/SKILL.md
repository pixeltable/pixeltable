---
name: review-the-change
description: >
  Review a pull request, diff, or proposed code change through the affected
  person's job, code guarantees, necessity, and verification evidence. Use for
  code reviews, in-depth PR reviews, and review of related client/server changes.
  Produces supported findings, maintenance suggestions, and explicit questions.
  A standalone UX critique or implementation request does not require this workflow.
---

# Review the change

Determine whether the change lets someone finish the intended job and preserves
the relevant guarantees. Combine a concrete walk through the product with a trace
through the code that makes it work. Review author explanations as claims to verify.

Adapted from `walk-the-job` and the user's six-part review workflow. This skill is
self-contained; the original skill is not a runtime dependency. Keep the primary
job concrete without limiting review of other callers, compatibility or data boundaries.

Explicit user instructions and the environment's permissions take precedence.
A review alone does not authorize product fixes, posted comments, account changes
or deployments. Preserve unrelated work; put experiments in an isolated environment.
Honor separate authorization when the user also requests fixes or publication.

## 1. Establish the requirement and reviewed scope

- Identify the problem, intended beneficiary, observable acceptance criteria and
  compatibility constraints. Read applicable repository instructions, relevant
  documentation and existing implementations before drawing conclusions.
- Record repository, intended base, base commit, head commit and merge base. Compare
  against the intended base; distinguish upstream work from the proposed change.
  If revisions are unavailable, say so. Recheck the head before delivery; label a
  review of an older revision explicitly rather than implying current coverage.
- For related PRs, record each revision and the actual dependency pins/protocols.
  Distinguish testing the proposed pair from testing the pair actually pinned.
- Separate requirements from inferred intent, conventions, implementation choices,
  explicit tradeoffs and temporary workarounds. State material unknowns. Ask only
  when an answer changes the review or its conclusion; continue independent work.

In review notes, make a compact job card from evidence:

| Person/role | Situation | Job | Done | Leave/failure | Entry/path | Evidence |
| --- | --- | --- | --- | --- | --- | --- |

Use the person the product names or the narrowest role supported by its API, docs
or source. Label inference. Invent neither a named persona nor feelings. Reuse an
existing job card when it matches; do not overwrite a card for a different job.

## 2. Build a change map

Inspect every changed file and substantive block, including removed code and the
context needed to understand it. Trace important paths through callers,
dependencies, persistence and public APIs. For each block, record:

| File/block | Intended behavior | Requirement/guarantee | Evidence | Review depth |
| --- | --- | --- | --- | --- |

Distinguish functional changes, conventions, organizational choices, temporary
workarounds and generated changes. Identify changes outside the stated scope.
Another block calling a helper does not establish that the helper is necessary.

Keep the map proportional: a few notes for a small diff; a navigable ledger for a
large one. A generated file/function inventory is navigation, not evidence that
the code was inspected. Track unread regions and resolve them; if access or other
constraints prevent completion, deliver a clearly bounded review and identify the
remaining regions. Never claim complete review from a path count or symbol scan.

## 3. Walk the affected job and challenge correctness

Follow the person's actual commands, screens, API calls or library example through
entry, first result, repeat use, relevant trouble, leaving and returning. Quote the
product's labels, output and errors with retrieval locations. Inspect backend-only
changes through their caller's observable contract; do not manufacture UI stages,
acquisition research or market claims.

Use an already available runnable product when practical. When an account, secret,
deployment or unavailable dependency prevents execution, trace the source and
label the walk accordingly. A source-based expected success is not an observed
successful job. After actual or expected success, name consequential new jobs the
change creates: follow progress, verify delivery, retry, cancel, clean up or return.
Show where the product helps and where the answer remains open.

At a consequential step, determine whether the caller can find the action, connect
it to the result, tell it worked and recover when it did not. Examine trouble the
code or documented use can actually produce. A full job walk includes relevant
failure/return paths or names what could not be checked and why; do not invent
trouble to meet a quota.

Challenge the behaviors relevant to this change:

- Compatibility, defaults and optional dependencies.
- Input validation, permissions, tenant/data boundaries and side effects.
- Error handling, retries, cancellation and cleanup.
- Concurrency, atomicity, replay and interrupted operations.
- Bounds, ordering, serialization and resource use.

For a suspected defect, establish its triggering input/state and trace the result
to concrete impact. Use a focused reproduction when practical; conclusive code
evidence is also sufficient. Test the actual changed path. Distinguish a mock's
assumption from behavior demonstrated by the system. State which consequences
were observed and which follow by inspection or remain inference.

Attribute the issue against the base. An unchanged defect is existing debt unless
the change introduces a reachable trigger or promises a guarantee it now violates;
explain that connection. Retained unsafe code is not newly added code. An explicit
tradeoff is not automatically a defect: assess it against acceptance criteria and
the caller's ability to understand its consequences.

## 4. Challenge necessity and complexity

For each substantial abstraction, helper, dependency or defensive branch, ask:

- What requirement or observable guarantee does it satisfy?
- What would fail if it were removed?
- Can an existing repository primitive preserve that behavior?
- Is the complexity inherent in the guarantee or introduced by this design?

Recommend deletion or consolidation only after naming the behavior to preserve,
the expected benefit and any lost guarantee or tradeoff. Separate these maintenance
suggestions from defects. Do not substitute an architectural preference for a
correctness finding or claim one representation is uniquely necessary.

Review generated changes separately: relevant metadata, ownership/generation rules,
unrelated dependency upgrades and formatting churn. Check examples and quickstarts
against the changed API.

## 5. Assess verification and challenge the draft

Determine what the tests actually prove. Check distinct behaviors and consequential
failures; identify missing cases, duplicate coverage, implementation-coupled
assertions and tests that do not establish their stated guarantee. Passing CI is
supporting evidence with a coverage boundary. A failed CI job is not automatically
a defect introduced by the PR; establish attribution or leave a precise question.

Associate results with the commit, dependency pair and environment tested. Preserve
exact invocations, relevant output and CI links when material. Keep live execution,
mocked execution, source inspection, static checks, unexecuted examples and
author-reported results separate. Report fixture/setup failures and excluded cases
as such; they are not passes. Use targeted checks first and broaden when unresolved
behavior warrants it.

Before delivery, apply [references/critic.md](references/critic.md) to the draft.
Read it for the final evidence/attribution audit. If an independent critic is already
authorized and available, it may challenge the draft; otherwise do this pass yourself.
The critic edits review claims, not product code. Remove disproved findings and
keep unverified concerns as explicit questions.

## 6. Deliver actionable feedback

Lead with supported findings in descending consequence. Each contains:

- A concise severity/title and precise file/line at the reviewed revision.
- Triggering conditions, causal path and concrete impact.
- Supporting evidence and verification status, including reproduction limits.
- The smallest suitable correction and any guarantee it must preserve.

Calibrate severity to demonstrated impact and plausible triggering conditions;
do not promote a possible charge or failure into observed loss. Avoid duplicate
comments, vague recommendations and unsupported stylistic preferences. A review
may have no findings; report what was checked without manufacturing defects.

Present maintenance suggestions separately with benefit and preservation conditions.
Finish with a concise scope summary, reviewed revisions, verification limits and
unresolved questions. Provide the expanded file-by-file walkthrough when requested.
If a required output schema exists, follow it and keep this evidence in the permitted
fields/artifact. Do not post feedback externally without the user's authorization.

For a dedicated agent whose instructions must stand alone, use the portable prompt
in [references/system-prompt.md](references/system-prompt.md). Read it only when
configuring/exporting an agent; the review workflow above is canonical for this skill.
