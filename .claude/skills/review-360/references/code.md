# Review the code

Judge the code by what it costs future readers and by the behavior it guarantees. Follow the target project's conventions and language. A local pattern is evidence to assess, not a universal rule.

## Gather the relevant implementation

For a change, read the description, applicable instructions, changed files at base and head, callers, and tests. Read deleted code, moved code, comments, and tests as carefully as additions. Consult earlier review discussions when available; distinguish unresolved points from addressed ones. Read a PR head without changing the user's working checkout.

For an existing codebase, select paths supporting the stated jobs and consequential invariants. Follow them from public entry point to effect. Report the boundaries inspected. Do not extrapolate a sampled subsystem to the entire codebase.

Establish the required behavior before judging its implementation. Separate core implementation from peripheral configuration, build files, and incidental edits.

## Articulate each abstraction

For each added, changed, or moved class and function in the core change, state what it does and owns in one or two crisp sentences. Apply the same test to the inspected path of an existing system.

- If a coherent responsibility cannot be stated, explain the conflicting jobs or hidden dependencies.
- Check that the name states that responsibility in the project's vocabulary.
- Put behavior with the concept and state it owns. A function that reaches into another object's internals may belong elsewhere.
- An abstraction must make sense without knowing all its callers or an undocumented sequence of earlier calls.
- A new requirement should fit naturally, or the abstraction should evolve to cover the complete set of requirements. Do not preserve the old shape by stacking conflicting meanings and special cases beside it.
- Judge a move in its new home. Carried-over comments must still be true.

## Inspect simplicity and clarity

Sweep new and changed identifiers, including parameters, fields, configuration keys, and tests. Use exact names that describe the thing, distinguish siblings, remain accurate in the new scope, and mean something to a new reader. Propose a replacement when a name is misleading.

Look for duplicated and near-duplicated behavior, dead code, speculative options, long lists of related primitive parameters, needless inheritance, exposed internals, pass-through classes, and inconsistent structure. Length is a clue, not a verdict. A split that adds helpers without coherent responsibilities can increase reading cost. Similar requirements should use similar code; normalize inputs once when it removes downstream cases.

Challenge assertions, TODOs, unsupported paths, and removed rationale. Identify whether a limitation is necessary, reachable, and expressed through the appropriate public error behavior. Preserve explanations of non-obvious constraints. Review changed prose with the prose guide linked from the skill entry point.

## Check consequential behavior

Trace relevant guarantees through callers and state: public contracts, defaults, compatibility, optional dependencies, input validation, authorization, transaction boundaries, partial failure, cleanup, cancellation, retries, concurrency, ordering, serialization, and bounds. Distinguish a cache from authoritative state. Check resource lifetime and invalidation as well as success.

Match effort to impact. Trace cost where data volume or external calls matter. A claim about an optimization requires evidence about the limiting factor and the work actually measured. Leave formatting and mechanically enforced conventions to the project's existing checks unless they conceal a substantive issue.

Tests should establish observable behavior through suitable public boundaries, cover distinct input and failure classes, and preserve existing edge cases. Prefer deterministic cases and existing fixtures. Question tests that only reimplement internals or assert component wiring. Internal testing can be warranted when no useful external observation exists; explain why. Match test proposals to the project's practical test boundaries.

Errors should identify the problem and relevant values, and explain recovery when it is unclear. Documentation should describe current public semantics, side effects, and limitations. Verify every suggested helper or existing pattern in the current code before citing it.

## Calibrate findings

For a defect, establish the trigger, execution path, and consequence. For a maintainability finding, identify the reading or ownership problem and a concrete improvement. Reading difficulty can be a finding without being mislabeled as a runtime bug.

Give overall comments for issues spanning the design and short local comments for the rest. State the problem first and offer the smallest suitable correction. One point per comment. Keep uncertainty explicit and remove remarks with no consequence. Attribute problems against the base in a change review.

A review recommendation follows its findings and coverage. Separate required corrections from optional improvements. Do not block on unrelated debt, stylistic preference, or an unverified suspicion. A partial pass stays partial.
