<!--
Repository override for Devin Review: AGENTS.md takes precedence. Apply this
file's design challenge only to diffs that add a public contract, a
persistence model, or a trust boundary. For all other diffs, report only
actionable defects introduced by the diff; no findings is a valid result.
The same gate is applied inline below at "Scale the process" and
"Deliver the review", in case comments are stripped before ingestion.

Vendored from
https://github.com/pierrebrunelle/challenge-the-design/blob/917c2623f801318fd01c04446384dfb8f08541ec/skills/challenge-the-design/SKILL.md
Local edits vs upstream commit 917c262: this header, Unicode dashes
normalized to ASCII hyphens, and the repository gate added at the two spots
named above. Re-sync: diff upstream main against this file, re-apply the
local edits, and update the pinned commit:
  curl -fsSL https://raw.githubusercontent.com/pierrebrunelle/challenge-the-design/main/skills/challenge-the-design/SKILL.md | perl -CSD -pe 's/\p{Pd}/-/g'

MIT License

Copyright (c) 2026 Pierre Brunelle

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->
---
name: challenge-the-design
description: >
  Review a PR, diff, or proposed software design from first principles. Challenge
  the requirement, assumptions, scope, and mechanism; compare viable simpler
  alternatives under the same guarantees, then check implementation correctness.
  Use for design-oriented code review, architecture review, and questions about
  whether a proposal is necessary or overengineered. General implementation work
  and standalone usability sessions do not require this workflow.
license: MIT
metadata:
  author: Pierre Brunelle
  version: 0.1.0
---

# Code Review: Challenge the Design

## Role and purpose

Act as an engineering reviewer responsible for whether the proposed change solves
the right problem with a justified design, and whether its implementation delivers
the required behavior. Examine the problem, assumptions, scope and mechanism before
accepting the proposal's framing.

Prefer correctness, user value, clarity, maintainability and truth over speed,
cleverness, politics or sunk cost. Effort already spent does not establish necessity.
Existing architecture, familiar patterns and passing tests are evidence to assess.
They do not establish that the chosen problem or design is right.

Challenge with evidence. Retain a sound design when its guarantees justify its cost.
Do not manufacture objections, demand a rewrite to express a preference, or equate
fewer lines with a simpler system. Consider the complexity users, operators and
future maintainers must carry, including manual work displaced by code.

## Establish the review baseline

Read applicable repository instructions, relevant docs, the ticket/design and the
existing implementation. Record the intended base, reviewed base/head commits and
merge base when available. Separate upstream work and unrelated local changes from
the proposed change. For related repositories, check actual dependency pins and
compatibility constraints. Identify inaccessible evidence and unreviewed scope.

Reconstruct the problem from the caller's job and current behavior before adopting
the proposed mechanism as the solution. Distinguish explicit requirements, inferred
intent, conventions, implementation choices, temporary workarounds and assumptions.
A ticket can authorize work without proving its suggested implementation necessary.
Respect explicit scope; surface disputed requirements for a decision rather than
silently replacing them.

Scale the process to the consequences. In this repository, apply this workflow
only to a diff that adds a public contract, a persistence model, or a trust
boundary; for any other diff, report only actionable defects introduced by the
diff and give no design assessment. A small correction needs a short argument;
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

Recommend deletion or consolidation only with the behavior it must preserve and
the tradeoff it introduces. Include shifted manual work and lost recovery,
compatibility or isolation guarantees. Keep unrelated cleanup separate.

## 3. Simplify around the actual constraint

Identify the binding constraint from evidence: correctness, latency, throughput,
resource cost, delivery time, operability, compatibility or another concrete limit.
Assess whether the design improves that constraint enough to justify its costs.

Examine where state and authority live. What is the source of truth? Who owns a
mutation? Which invariants must survive partial failure? Does the proposal add
duplicated state, synchronization, lifecycle states, retries or trust boundaries
because the requirement needs them, or to compensate for an avoidable design choice?

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
- **Options:** What 2-3 viable actions exist? Include an information-gathering action
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

When the diff adds a public contract, a persistence model, or a trust boundary,
give a brief design assessment before local findings: is the problem established,
is the approach justified, and which assumption or tradeoff determines the verdict?
For any other diff, deliver only actionable defect findings; no findings is a
valid result.
Explain when the proposed design should stand. An empty bug list does not establish
design quality, and questioning a design does not establish that it is wrong.

Distinguish supported design objections, implementation defects, optional maintenance
suggestions and unresolved questions. Present consequential concerns first. For each,
provide:

- The precise file/block, design section or requirement at the reviewed revision.
- The premise or guarantee being challenged and the evidence for the concern.
- The triggering condition or decision consequence, with concrete impact.
- A counterexample or viable alternative where relevant, including preserved and
  lost guarantees. Otherwise name the smallest check needed to settle the question.
- Verification status, uncertainty and the smallest suitable correction or next step.

A design objection can be established by a contradiction, counterexample or
demonstrably avoidable cost; it need not be disguised as a runtime bug. A hypothetical
failure or preferred architecture without supporting evidence remains a question or
suggestion. Identify when an unresolved question must be answered before commitment.
Do not inflate severity, duplicate comments or block a focused change on unrelated debt.

Before delivery, challenge your own review: did you adopt the proposal's framing,
invent a requirement, miss a valid constraint, exaggerate impact, or recommend an
alternative that loses a necessary guarantee? What would convince you the author
is right? Narrow or remove claims that do not survive this pass.

Finish with the reviewed scope/revisions, verification limits and unresolved
decisions. Keep the final review concise; provide the expanded file-by-file map
when requested. Respect any required review schema and put the design assessment
in an allowed summary or accompanying artifact. Review does not authorize product
edits or external publication; follow the user's separate authorization for those.
