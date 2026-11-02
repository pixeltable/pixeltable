# Principles

This file applies Marcel's guidelines in [effective-coding-practices.md](effective-coding-practices.md) to
Pixeltable reviews. The central act comes first, then the three lenses, then the supporting checks. The
examples are his own review comments, with the PR number. They illustrate a principle and do not limit it.

## The central act: articulate each abstraction

Every class and function should reflect an abstraction: self-contained, with a limited, coherent set of
responsibilities, and stated in a few crisp sentences. If it cannot be stated, there is no abstraction yet,
and the code needs rework rather than polish. Use the statement to check the name, the ownership of the
state, the comment and the fit of new requirements. Moved code gets the same test in its new home.

Marcel's words:
- "I'm finding the logic of this function hard to follow. I'm not sure it really stands on its own (=
  should be a function; a function is an abstraction; can you cleanly articulate what this one does?)."
  (#1472)
- "That would effectively mean that in order to understand the semantics of this function, you also need to
  understand the semantics of all of its call sites, ie, there are no abstractions." (#917)
- "Missing a class docstring explaining the scope of this abstraction." (#1215)
- "This needs to check supports_idxs. As it is, it's only correct to call this method after _build_...(),
  meaning it doesn't stand on its own and has subtle control flow dependencies that aren't directly
  visible." (#1515)
- "I have no idea what this should do." (#403)
- "This is a method of the Column class (create() is sufficient)." (#1127)

## Lens 1. Abstractions hold up: evolve, don't shoehorn

When a change adds a requirement to existing code, the code should end up shaped as if it had been written
for both requirements from the start. If the class can be naturally augmented, augmenting it is right. If
the new requirement conflicts with the abstraction, the change needs a new class or set of classes for the
complete set of requirements: often a base class with one implementation per case, a parameter instead of
a subclass, or an existing mechanism, generalized. Sometimes the old path can go. Stacking keeps the old
shape and adds the new case next to it.

Signs of stacking:
- a function grows parameters next to its existing ones;
- a flag or a type branch selects between the old and the new behavior;
- a field or value takes on a second meaning;
- a concrete class becomes the base class of the new one;
- the old path stays next to the new one;
- a general mechanism gains a special case;
- a near-copy of an existing helper or class appears for the new case.

The same lens applies to a fix: fix the cause once, where it originates, rather than patching each
symptom or each caller.

Marcel's words:
- "I don't think this flag makes sense. If both column and parameter have json typing information, it
  should be checked by is_supertype_of()." (#1215)
- "It looks like the whole purpose of this subclass is to override _http from the base class. Instead,
  ProxyClient should parameterize the transport." (#1440)
- "Let's create a new expr subtype for string operations. This isn't really arithmetic, and it'll be easier
  to special-case later." (#529)
- "Can't we treat an anonymous snapshot as a pure snapshot (and then not special-case it here and
  elsewhere)?" (#710)
- "If ancestors() included the root, you wouldn't need this special-casing." (#525)
- "aren't we saying that we don't want that situation (interleaving creating overloads with creating
  FunctionCalls)? what else can be simplified if we don't have to deal with that situation?" (#403)
- "The parallel hierarchies for FunctionCall and Iterator feel like they have a lot of overlap,
  consolidation would be nice." (#1559)
- "httpx.AsyncClient() is used in a number of our integrations. Let's fix this in a comprehensive way, so we
  don't need to bake it into each one of them separately." (#1062)

## Lens 2. Simplicity: the smell catalog

Check the core files for the smells in [effective-coding-practices.md](effective-coding-practices.md). In
Python terms:

- **Long function, large class:** one coherent responsibility each. Split along the abstraction, not by
  line count.
- **Long parameter list, missing abstraction:** pass the object, not several of its fields. Data that
  always travels together becomes a class or TypedDict.
- **Duplicated code, near-duplication included:** a new class or function modeled on a sibling shares the
  common part instead of copying it. Use the existing mechanism instead of a parallel one, including what
  Postgres already provides.
- **Dead code:** delete it, together with parameters, branches and members that nothing uses.
- **Speculative and gratuitous generality:** no API, parameter, flag, generic type or `Collection` where
  today's requirement needs none. Introduce an abstraction when a second case arrives (lens 1), not before.
- **Questionable inheritance:** a subclass that only overrides one detail is a parameter. A concrete class
  serving as the base of another becomes an abstract base plus two implementations.
- **Entangled classes, exposed internals:** classes know as little as possible about each other and never
  read each other's internal state. Every public name needs a compelling reason.
- **Superfluous classes:** a wrapper or pass-through that delegates all its work goes.
- **Inconsistent structure:** similar code looks similar, and local patterns are followed.
- **Performance cliffs:** no optimization that serves one narrow scenario and leaves its siblings slow.

Beyond the catalog:
- Normalize inputs once, so special cases disappear downstream.
- Order logic so its completeness is visible: validation first, then traversal; separate phases rather
  than interleaved ones.
- Introduce a new variable when a value changes meaning or type.
- Keep code debuggable: materialize a small result rather than generate it, and no property that raises
  or runs a transaction.
- Readability beats a lint rule.

Marcel's words:
- "I don't think there is any real use case for this and would prefer not to have it, simply to reduce
  complexity. Saying "it might be useful in the future" is not a use case." (#682)
- "Instead of providing 3 fields from the ProxyRequest structure, just pass the request itself." (#1545)
- "It would be cleaner to have this be an abstract class and an InlinePartSink as this particular
  implementation." (#1545)
- "Make this an instance method of TV: it should not/will not be used outside of it, and it needs TV
  state." (#1472)
- "It's identical to the previous response model, unify them." (#1336)
- "re Collection: is this gratuitous generality? set[] is easier to understand. Would set[] have any
  practical downsides?" (#1391)
- "Is there any reason to allow both None and [] (and treat them the same)? You could further simplify the
  code by normalizing to []." (#649)
- "Do we need schema_version? It always changes in lockstep with version." (#1298)
- "Are there other callers? If not, inlining it would make the code easier to follow. This doesn't really
  add up to a named piece of logic (ie, I have to read the entire function to understand what it does; the
  name doesn't help me)." (#1472)
- "The logic would be a bit easier to follow if the structural validation happened first, followed by the
  traversal you're doing here." (#1215)
- "We already have a QColumnId." (#841)

## Lens 3. Clarity

A reader new to the code follows it from the names and the code alone, without the PR's history and
without reading the body of each called function. Reading difficulty is a finding in itself.

- **Names:**
  - exact and specific; a name that is only roughly accurate is wrong;
  - no type in the name;
  - the codebase's established vocabulary, one term per concept, used consistently (an `open()` gets a
    `close()`);
  - fitted to the identifier's current home: a method of `Column` is `create()`, not `create_column()`;
  - renamed when the meaning changes; a parameter, its metadata field and its API key share one name.

  The test: read the name alone to another developer. They can say what it does.
- **Comments** state intent, which code cannot express, and act as a check: a comment that is hard to
  write, or that contradicts the code, points at an unclear abstraction.
  - A class comment gives its user the abstraction, non-obvious observable behavior, usage patterns and
    restrictions such as thread safety.
  - A function comment gives the caller its observable behavior, the roles of the parameters and its side
    effects, but nothing already visible in the signature.
  - Invariants between members are commented next to the members and asserted in the code.
  - Comments explain the non-obvious: the reason for a choice, a unit, the keys of a dict, the fields of a
    tuple, the output shape of a numpy operation, a magic number, what a lock protects.
  - The answer to a reviewer's "why?" goes into the code. Comments describe only the code at hand and stay
    accurate; brevity is a virtue.
  - A missing comment is as much a finding as a superfluous one. Ask for one wherever a reader would ask
    why (an unusual construct such as `__slots__`), and wherever a function's contract is not visible from
    its signature.
- **Types** say what a value is:
  - full signatures, and a class-level declaration for every member;
  - no `Any` where a real type exists;
  - `X | None` and `dict` in new code, not `Optional` and `Dict`;
  - a `Literal` alias for a fixed set of strings;
  - a TypedDict or dataclass instead of `dict[str, Any]`;
  - real objects instead of string comparisons;
  - explicit comparisons (`is not None`, `len(x) > 0`).
- **Prose** is plain. Run the checks in [prose.md](prose.md).

Marcel's words:
- "Is that basically a resource delay (or resource refill delay)? (Ie, a good name would describe what it
  is, not what the caller does with it.)" (#912)
- "Please make an effort to conform to existing naming conventions and avoid gratuitous terminology
  changes." (#701)
- "Very awkward name. additional_columns is a misnomer. Let's clean this up." (#1127)
- "Is __resolve_type() more appropriate? Infer usually indicates you're inferring from values." (#1215)
- "Missing class comment. In particular, the relationship to ColumnType and their respective
  responsibilities." (#314)
- "When I'm asking for clarification in a review comment, I usually mean for that to happen in a comment in
  the code (not in the review response)." (#211)
- "Claude loves dropping references to other pieces of code (or caller logic) into code comments. This
  creates a problem when the referenced code changes." (#1301)
- "Do we have a defined set of storage targets? If so, create a Literal or something. Having everything be
  :str makes it really hard to understand the code or do any meaningful type checks." (#1604)
- "All member vars require a class-level decl" (#1324)
- "Comment: why slots" (#1427)
- "Should state that this is a contextmgr and a drop-in for telemetry.span()." (#1427)

## Supporting checks

### Ownership and layering

Each piece of logic goes in the class or layer that owns the state and the concept, and a function makes
sense without knowledge of its callers. A lower layer never depends on a higher one: utils do not import
catalog types. New process-wide state goes into Env, and per-thread state into Runtime, never into a new
singleton or module-level globals. Runtime state (clients, thread pools) does not go in plan nodes or
syntactic objects, and generic helpers do not go in one provider's module.

- "Table should not be running queries against the stored metadata itself. It's only a user-facing
  interface class. All metadata retrieval is handled by Catalog." (#640)
- "We have ColumnType.to_sa_type(). Why shouldn't the other direction be in ColumnType?" (#1324)
- "(Another aside, Column shouldn't leak into utils.)" (#886)
- "Why not call this class Telemetry and add an instance to Env?" (#1427)

### Semantics and invariants

Every state has one defined meaning, and the code asserts it where it matters. One user operation creates
one table version, persisted references use ids, and ids never derive from names. Behavior matches Postgres
or the established system unless a reason is stated. A loud failure beats a wrong result or a silent one.

- "This should be: is_replica == (anchor_tbl_id is not None)" (#917)
- "This creates one new version per row batch. That would be problematic, because revert() would take you
  back to a state that never existed." (#246)
- "Can we do something stronger here than get() (and for 'image_url')? If this gets renamed, we'll never
  find out and it'll fail silently." (#1293)

### Public API

The public surface is small, modeled on existing Pixeltable APIs and on pandas, Python, Postgres or
SQLAlchemy, and stated in the user's vocabulary.
- Coding agents are primary users: one way to express each thing, one name per concept, one error message
  per problem, structured output with human-readable formatting on top, and no forced decision without a
  consequence.
- Analogous cases (table kinds, index kinds, local and hosted) behave the same, and an add gets a matching
  drop.
- API changes are discussed before they are coded.

- "'raise' sounds like an implementation detail (and would be meaningless if we ever do SDKs for languages
  w/o exceptions)." (#327)
- "remove a source of syntactic variation that doesn't have any functional benefit and will probably just
  provide another opportunity for agent confusion." (#1479)
- "Why not allow this? We do it for embedding indices." (#1472)
- "For changes to the API (which this is), I would prefer to see some preceding discussion." (#682)

### Failure paths

Check what happens under concurrent access, after a drop or rename, on an exception, and on a partial
failure. Cleanup goes in `finally`. Stale objects are invalidated, never patched. A cache is only a cache,
and module-level state needs a lock.

- "This needs to happen in a finally block." (#814)
- "We shouldn't try to fix up Table instances, we need to invalidate them and force a reload." (#868)
- "This is not thread-safe, but would be with a lock." (#1385)

### Cost

Where data volume matters:
- filter in SQL, and avoid extra passes;
- bound memory;
- use async clients for IO, and one shared client per service.

Keep the install lean: import optional packages where they are used, and keep version bounds loose.
Micro-optimizations get dismissed.

- "Don't do count() here, that could be pretty expensive and in effect double the runtime of this
  operation." (#246)
- "There is a azure.storage.blob.aio, which is what we should be using." (#886)
- "This is problematic, because it'll require torch to be installed, which is not a Pixeltable
  dependency." (#88)

### Errors

A user-caused condition raises a specific Pixeltable `Error` subclass with a matching `ErrorCode`, from the
public function at the user's entry point. Lower layers assert. A message states the problem in user
terms, includes the offending values, and suggests a fix when users would otherwise be stuck. Order the
checks by what a user must fix first.

- "This is not a user-facing error (Error is reserved for situations that bubble up to the user). It
  should really be an assertion." (#138)
- "This needs to include the actual values, otherwise it would be hard for the user to diagnose what's
  going on." (#1203)
- "Check this first. Otherwise a user might hit the name collision error first (and correct it and then
  get this)." (#1472)

### Tests

- Tests are black box: they exercise the public API (SDK, CLI, HTTP) the way users do, and assert only on
  observable results. White box only when the behavior cannot be observed any other way, with a comment
  that says why. Flag:
  - a test of a private function, or a call to a non-public method of a public object;
  - reads of private attributes, internal state, store tables or system tables;
  - mocks or fakes of internal components;
  - test logic built on knowledge of the implementation: an expected value derived from an internal
    constant or structure, or a reimplementation of the logic under test.

  The same rule applies to the review's own test proposals.
- Coverage starts from an enumeration of every kind of input to the new code, not from the defects found
  along the way. New behavior is proven across its whole surface:
  - tables, views, snapshots and reload;
  - local and hosted modes;
  - every new UDF;
  - default and explicit forms;
  - CLI commands with `--json`;
  - error paths.
- Test code is concise and reads like user code: `assert all(...)` instead of an assertion loop, no
  redundant calls, compact asserts, parametrized cases instead of repetition.
- Tests are deterministic: `order_by()` whenever order matters, and no timing dependence.
- Extend existing tests and fixtures rather than adding near-copies. An edited test keeps its edge cases.

- "I really want us to get away from using module internals in tests, which has created a fair amount of
  work for me in the past." (#696)
- "That's internal to the implementation and shouldn't be part of the test." (#411)
- "Let's stick to the public interface (which this isn't part of)." (#1096)
- "This tests internals. A better way to test this is end-to-end, eg, via the cli." (#1745)
- "You're basically reimplementing the logic in FileCache here, which I think is problematic to do for a
  test." (#276)
- "Missing: aggregate queries, both grouping and non-grouping." (#1258)
- "Missing: view on a snapshot, with the view referencing c1." (#751)
- "assert all(...) instead of the loop" (#1258)
- "Can you express these cases more succinctly with 3 lambdas (per create_x(...)) and two nested loops
  over those lambdas? It looks very repetitive." (#1150)
- "You're expecting a particular order, which means you need an order_by (otherwise Postgres might decide
  to return data in a different order and the test breaks for no obvious reason)" (#139)
- "How is this verifying the edge case? It looks like it dropped the edge case." (#1515)

### Docs

Public docstrings and docs state the real semantics in database terms, cover only what exists today, and
change with the API. They show realistic examples, link each related function and write it with `()`,
and leave out marketing words, implementation details and hand-written defaults.

- "I think we should stick to things/functionality that is part of the code base as it is today." (#112)
- ""high-throughput/efficient" sounds like marketing, which we typically avoid, and aside from that the
  sentence doesn't convey anything useful." (#1385)

### Limitations and deferral

Challenge every new assert, TODO and "not supported" path. Is the limitation needed, or is it easy to lift?
Can a user reach it? Then it raises a specific `Error`, not an assert. Is the assert meaningful at all?

A gap may wait only as a TODO at the site, stating what has to happen, ideally with a Jira reference.
Broken core semantics and API changes cannot wait. A small fix next to the change belongs in this PR. A
skipped test or check gets a TODO to restore it.

- "This cannot be a todo, it breaks the insert semantics." (#123)
- "You need to check for the mixed versioning case in join() and return a pxt.Error." (#1233)
- "I don't think that makes sense. We can only have replicas of specific table versions, never of live
  tables." (#1233)
- "It's little work, and doing this is a separate PR generates a lot of overhead in terms of review
  mechanics." (#244)

### Diff hygiene

Every hunk has a reason: renames, restructurings, removed comments and changed tests included. A move is
judged by ownership (see the central act). Modify a function in place so the diff stays reviewable. No
debug prints or commented-out code. A mechanical change keeps functionality, comments and test scenarios.

- "What's the point of restructuring this? It seems like a purely syntactic change." (#1090)
- "It's hard to see the diff this way. Please modify the original function in-place." (#705)
