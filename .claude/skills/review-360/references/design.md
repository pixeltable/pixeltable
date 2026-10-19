# Challenge the design

Review the whole proposal before accepting its framing. Existing architecture, effort already spent, and passing tests do not establish necessity. A smaller diff is not automatically a simpler system.

## Establish the problem

State the observable problem, who encounters it, the condition that triggers it, and what result would prove it solved. Reconstruct this from the caller's job and current behavior. An issue may authorize work without proving its suggested implementation necessary.

Distinguish hard constraints from habits, workarounds, and constraints introduced by the proposal. For each consequential assumption, ask what depends on it, what fails if it is false, and what evidence could settle it. An unknown is not itself a defect.

For a PR stack or related repositories, review each change as a complete unit and the combined design as a whole. Check actual dependency pins and compatibility. Trace entry points to effects across files and boundaries. Identify duplicated plumbing and temporary mechanisms that remain after the combined change.

## Attempt deletion with the contract preserved

Answer this explicitly: what could be deleted, inlined, or consolidated while preserving all required behavior and guarantees?

First establish the contract: entry points, outputs, side effects, ordering, compatibility, error behavior, authorization, cleanup, and any retry, crash, or idempotency guarantees. Then trace the shortest direct path that satisfies it. Examine intermediate layers for:

- Pass-through wrappers, mirrored models, and conversions that enforce no invariant.
- One-implementation interfaces, static registries, one-value strategy options, and unused extension points.
- Custom queues, locks, retries, caches, or reconciliation already supplied by an existing capability.
- Parallel validation, serialization, and error handling; legacy branches made obsolete by the new path.
- Derived state stored independently and kept synchronized with its source.
- Caller workarounds that disappear when the shared boundary owns the complete contract.

For each substantial mechanism, identify the guarantee that requires it and what breaks without it. Necessity must trace back to the job or invariant; one unnecessary component needing another proves neither necessary.

For a proposed removal, name what absorbs its responsibility, why the contract survives, and what check would disprove that claim. Keep a component when its required guarantee cannot otherwise be preserved. Do not count lost validation, weaker recovery, broken compatibility, deleted behavioral tests, or dense syntax as simplification. Keep unrelated cleanup separate.

## Compare viable alternatives

Consider current behavior, a direct correction or reuse, and the proposal when each is viable. Do not invent options to fill a quota. Compare total complexity, including manual work, recovery, migration, ownership, and reversibility.

Identify the actual constraint from evidence: correctness, latency, resources, operability, compatibility, or delivery. Require evidence for scale or performance claims. Locate the source of truth and mutation authority. Ask whether extra state, synchronization, lifecycle states, and trust boundaries solve a requirement or compensate for a design choice.

Check the implementation as well as the direction: defaults, bounds, permissions, atomicity, concurrency, cancellation, cleanup, replay, and failure behavior where relevant. A sound design can contain a defect; a defect does not prove the architecture wrong.

When automation is proposed, establish that the underlying work is necessary and sufficiently understood. Include monitoring, recovery, and exceptions in its cost. Automation required by a reliability or security guarantee can be justified immediately.

## Settle and deliver

Use a focused reproduction, caller trace, measurement, or bounded experiment that could change the decision. Test required outcomes and assumptions, rather than only internal wiring. Walk the affected job through success, repeat use, relevant trouble, and return when those paths matter. Name new work created after success.

For a consequential dispute, state the requirement, unknowns, viable options, constraints, downside, choice, falsifying evidence, and verification. Keep smaller decisions proportionate.

Lead with whether the problem and approach are justified. Answer the deletion question for each PR and across the combined set. If nothing can be removed safely, name the guarantees requiring the current pieces. A design objection needs evidence of contradiction, avoidable cost, or a counterexample. A preferred architecture alone remains a suggestion.

Before delivery, challenge your own premise: did you invent a requirement, miss a valid constraint, or recommend losing a guarantee? Narrow claims that do not survive.
