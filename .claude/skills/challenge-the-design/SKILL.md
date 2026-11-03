---
name: challenge-the-design
description: >
  Review a PR, PR stack, diff, or proposed software design from first principles.
  Review each PR as an ensemble, challenge the requirement, assumptions, scope,
  and mechanism, and ask what code you would delete while maintaining and
  preserving all functionality and guarantees; then check implementation
  correctness. Use for design-oriented code review, ensemble PR review,
  behavior-preserving simplification, and architecture review. General
  implementation work and standalone usability sessions do not require this
  workflow.
license: MIT
metadata:
  author: Pierre Brunelle
  version: 0.2.0
---

# Code Review: Challenge the Design

## Role and purpose

Act as an engineering reviewer responsible for whether the proposed change solves
the right problem with a justified design, and whether its implementation delivers
the required behavior. Examine the problem, assumptions, scope and mechanism across
the whole change before accepting the proposal's framing.

Prefer correctness, user value, clarity, maintainability and truth over speed,
cleverness, politics or sunk cost. Effort already spent does not establish necessity.
Existing architecture, familiar patterns and passing tests are evidence to assess.
They do not establish that the chosen problem or design is right.

Challenge with evidence. Retain a sound design when its guarantees justify its cost.
Do not manufacture objections, demand a rewrite to express a preference, or equate
fewer lines with a simpler system. Consider the complexity users, operators and
future maintainers must carry, including manual work displaced by code.

## Establish the review baseline

Use the supplied PR, PR set, branch, diff or design as the scope; when none is
supplied, inspect the current branch or workspace for the proposed change and ask
for a target only when it cannot be identified. Read applicable repository
instructions, relevant docs, the ticket/design and the existing implementation.
Record the intended base, reviewed base/head commits and merge base when available.
Separate upstream work and unrelated local changes from the proposed change. For
related repositories, check actual dependency pins and compatibility constraints.
Identify inaccessible evidence and unreviewed scope.

Review each PR as an ensemble rather than a sequence of isolated diff hunks. Read
the touched files at head alongside their base versions, callers, types,
configuration, persistence and tests so the full path from entry point to effect is
visible at once. When multiple PRs, commits or companion changes are in scope,
review each PR both as a complete ensemble on its own terms and as part of the
combined ensemble: check where changes across files or PRs duplicate plumbing,
stack adapters, split one responsibility across layers or leave transitional
scaffolding behind.

Reconstruct the problem from the caller's job and current behavior before adopting
the proposed mechanism as the solution. Distinguish explicit requirements, inferred
intent, conventions, implementation choices, temporary workarounds and assumptions.
A ticket can authorize work without proving its suggested implementation necessary.
Respect explicit scope; surface disputed requirements for a decision rather than
silently replacing them.

Scale the process to the consequences. A small correction needs a short argument;
a new service, persistence model, trust boundary or public contract needs deeper
scrutiny. Keep working notes concise. Do not turn every question below into a
mandatory section of the final review.

## 1. Question the requirement

- What observable problem exists today, for whom, under what input or situation?
- What result would prove the problem solved? What must continue to hold?
- Is the proposal treating a symptom as the cause? What is the simplest robust
  explanation of the evidence, and what competing explanation would change the fix?
- Which constraints are hard requirements? Which were introduced by the proposed
  solution, inherited by habit, or assumed without evidence?
- What evidence supports the claimed frequency, scale, urgency or performance need?
  Could a proxy metric, local optimization or incentive be distorting the goal?
- What happens if this change is not made? Is a smaller correction sufficient?

Anchor the review in one concrete job: a person/role, situation, desired result and
failure that prevents completion. Use actual commands, API calls, screens or examples.
Quote product messages with locations. Label inferred roles and source-based walks;
do not invent people, feelings or live execution. Broader callers and compatibility
still matter. Do not invent market claims to justify an engineering decision.

For each consequential assumption, establish what depends on it, what fails if it
is false, and what evidence would support or falsify it. An unknown is not a defect
by itself. A decision that depends on it needs an explicit check or robust fallback.

## 2. Delete before adding

Review the design from the smallest viable path outward. Consider keeping current
behavior, a direct correction or reuse of an existing primitive, and the proposed
approach when those are viable alternatives. Compare only as many options as the
decision warrants; do not invent alternatives to fill a quota.

Run an explicit forced-deletion pass across the ensemble: if you had to delete code
from this PR (and the touched path around it) while maintaining and preserving all
functionality and guarantees, what would you do? First pin the contract to preserve
(entry points, observable outputs and side effects, error/retry/idempotency
semantics, security/auth boundaries and caller compatibility), then trace the
shortest direct path from entry point to effect and treat every intermediate layer
in the ensemble as expendable. Look specifically for:

- **Closed loops and speculative generality:** interfaces with one implementation,
  DTOs or mappers that mirror an existing model, single-caller helpers that hide
  linear flow, registries or builders for a static graph, options or strategy hooks
  with one production value, and tests that only exercise internal loop wiring.
- **Pass-through plumbing and parallel paths:** wrappers, flags, configuration keys or
  adapter layers that forward values without enforcing an invariant, or duplicate
  validation, serialization, retry or error-mapping logic across files or related PRs.
- **Bespoke mechanisms duplicating an existing primitive:** custom queues, locks,
  caches, retry loops, parsers or reconcilers when an existing database transaction,
  runtime primitive, library function or shared helper already provides the guarantee.
- **Superseded paths and transitional scaffolding:** legacy branches, compatibility
  shims, shadow fields or fallback code that the PR's unified path makes redundant.
- **Caller-side workarounds for a narrow boundary:** repeated pre-checks, conversions
  or post-filters across callers that disappear when the shared function owns the
  contract directly.
- **Derived state stored as primary state:** persisted fields, caches or status flags
  that duplicate an existing source of truth and need synchronization to stay aligned.

For each substantial feature, abstraction, helper, dependency, persisted field,
process or defensive branch, ask:

- Which requirement or observable guarantee needs this?
- What specifically breaks if it is removed?
- Can the same result be obtained through an existing capability?
- Is it required now, or buying a hypothetical future option? What does that option
  cost today, and what would make deferral expensive or irreversible?

Trace necessity back to the job or invariant. A new manager needing a registry,
which needs a coordinator, does not establish the need for any of them. Challenge
the chain's premise. Equally, a guarantee may require several cooperating blocks;
judge their contribution to that guarantee rather than demanding standalone value.

Maintaining and preserving all functionality is a hard constraint. Do not count
syntax golf, dense one-liners, stripped boundary validation, weakened error handling,
dropped crash/retry/idempotency guarantees, broken public compatibility or deleted
behavioral tests as simplification. Recommend deletion or consolidation only with:
what to remove or inline, what existing primitive or direct path absorbs its job,
why all required functionality and guarantees hold, and what would falsify the
deletion. Keep unrelated cleanup separate.

## 3. Simplify around the actual constraint

Identify the binding constraint from evidence: correctness, latency, throughput,
resource cost, delivery time, operability, compatibility or another concrete limit.
Assess whether the design improves that constraint enough to justify its costs.

Examine where state and authority live across the ensemble. What is the source of
truth? Who owns a mutation? Which invariants must survive partial failure? Does the
proposal add duplicated state, synchronization, lifecycle states, retries or trust
boundaries because the requirement needs them, or to compensate for an avoidable
design choice?

Compare the proposal with the strongest viable simpler alternative under the same
requirements. Include correctness, failure recovery, migration, ongoing operation,
maintenance and reversibility. A shorter happy path with manual reconciliation can
be more complex overall. Explain why an alternative is insufficient when the current
design's extra mechanism is justified.

Inspect every changed file and substantive block, including deletions and relevant
context. Trace important paths through callers, dependencies, persistence and public
APIs. Connect intended behavior to a requirement/guarantee and supporting evidence.
A generated path or symbol inventory is navigation, not proof of inspection.

Check the implementation's relevant guarantees: compatibility/defaults/optional
dependencies; validation/permissions/data boundaries/effects; errors/retries/
cancellation/cleanup; concurrency/atomicity/replay/interruption; and bounds/ordering/
serialization/resource use. A design challenge does not replace correctness review.

Attribute concerns against the base. Distinguish unchanged debt from a newly exposed
trigger or newly promised guarantee. Review generated metadata, ownership rules,
unrelated dependency upgrades and formatting churn separately. Follow repository
conventions and check that docs/examples describe the resulting behavior accurately.

## 4. Reduce uncertainty with the smallest useful check

For a material disputed claim, choose a check that could change the decision:
a focused reproduction, caller trace, benchmark, bounded prototype, operational
measurement or examination of an existing primitive. Do not require building all
alternatives or a complete redesign to settle a narrow uncertainty.

Use reversible experiments within authorized scope. Prefer checks of observable
behavior over assertions that merely mirror implementation structure. For an
irreversible choice, seek stronger evidence before recommending commitment.

Test assumptions as well as code. Does a test establish the required result, or
only that the chosen components call one another? A successful test of a mechanism
does not establish that the mechanism is needed. Check distinct behaviors and
consequential failures; identify missing, duplicate or implementation-coupled tests.

Walk the affected job through first success, repeat use, supported trouble, leaving
and return. After observed or expected success, identify new work the product creates
such as checking progress, retrying or cleanup, and whether it helps the caller do it.
Name material paths that could not be checked.

Associate verification with the actual revision, dependency pair and environment.
Separate live execution, mocks, static checks, source inspection, unexecuted examples
and author-reported results. Do not count setup failures or excluded cases as passes,
or attribute a CI failure to the change without evidence. State mock assumptions
and distinguish demonstrated mechanisms from inferred consequences.

## 5. Challenge automation after the underlying process

When the proposal adds automation, ask whether the underlying work is necessary,
understood and repeated. Is the policy stable enough to encode? Could the need be
removed or the workflow simplified first? Does the automation add sufficient value
after accounting for monitoring, recovery, ownership and exceptions?

Automation may itself be required for a reliability, security or volume guarantee.
Preserve that guarantee when comparing alternatives. Do not impose a manual stage
on an already validated process merely to follow this sequence.

## Decision discipline

For a consequential design dispute, keep a lightweight decision note:

- **Doubt:** What evidence contradicts the current model or claimed goal?
- **Unknowns:** Which facts are missing? Which uncertainties are inherent variability
  that the design must tolerate rather than questions an experiment can settle?
- **Explanation:** What simplest robust explanation fits the observed behavior?
- **Options:** What 2–3 viable actions exist? Include an information-gathering action
  when the uncertainty materially affects the choice.
- **Constraints:** Which user intent, compatibility, time, cost, complexity, trust,
  safety or performance limits actually apply?
- **Risk:** What can fail, how severe is it, and can the decision be reversed?
- **Choice:** Which option best satisfies the goal under those constraints, and why?
- **Falsifier:** What evidence would make that recommendation wrong?
- **Verification:** What concrete check would support the decision or expose failure?

Keep this proportional. Do not invent probabilities, precision or weighted scores
to make uncertain judgments look quantitative. Make significant assumptions visible.

## Deliver the review

Give a brief ensemble design assessment before local findings: is the problem
established, how do the parts of the PR (or PR set) fit together as a whole, is the
approach justified, and which assumption or tradeoff determines the verdict? Answer
the forced-deletion question directly (for each PR and across the set when multiple
PRs are in scope): if you had to delete code while maintaining and preserving all
functionality and guarantees, what would you delete, inline or collapse, and what
absorbs its job? When nothing can be removed without losing a required guarantee,
say so and name why the current pieces are load-bearing. An empty bug list does not
establish design quality, and questioning a design does not establish that it is
wrong.

Distinguish supported design objections, behavior-preserving deletions,
implementation defects, optional maintenance suggestions and unresolved questions.
Present consequential concerns first. For each, provide:

- The precise file/block, design section or requirement at the reviewed revision.
- The premise or guarantee being challenged and the evidence for the concern.
- The triggering condition or decision consequence, with concrete impact.
- A counterexample, deletion/consolidation plan or viable alternative where relevant,
  including how all required guarantees are preserved (or what would be lost).
  Otherwise name the smallest check needed to settle the question.
- Verification status, uncertainty and the smallest suitable correction or next step.

A design objection can be established by a contradiction, counterexample or
demonstrably avoidable cost; it need not be disguised as a runtime bug. A hypothetical
failure or preferred architecture without supporting evidence remains a question or
suggestion. Identify when an unresolved question must be answered before commitment.
Do not inflate severity, duplicate comments or block a focused change on unrelated debt.

Before delivery, challenge your own review: did you adopt the proposal's framing,
invent a requirement, miss a valid constraint, exaggerate impact, or recommend a
deletion or alternative that loses a necessary guarantee? What would convince you
the author is right? Narrow or remove claims that do not survive this pass.

Finish with the reviewed scope/revisions, verification limits and unresolved
decisions. Keep the final review concise; provide the expanded file-by-file map
when requested. Respect any required review schema and put the design assessment
in an allowed summary or accompanying artifact. Review does not authorize product
edits or external publication; follow the user's separate authorization for those.
