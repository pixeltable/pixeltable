# Review prose

Apply these checks to changed comments, documentation, errors, and the review itself. Follow project conventions rather than imposing a particular punctuation style or language.

- State the concrete action, behavior, or constraint. Cut filler and claims that could describe any product.
- Use identifiers and terms from the current system. Define unfamiliar concepts and make pronoun references clear.
- Use a precise name before adding a comment that restates the code. Keep comments explaining intent, invariants, surprising behavior, or constraints the code cannot express.
- Verify that a comment describes what its location guarantees. Move a claim about caller behavior to the caller when the callee cannot guarantee it.
- Public documentation states observable semantics, parameter meaning, side effects, and limitations where needed. Internal documentation covers non-obvious points without boilerplate.
- Remove obsolete subjects and rationale. Preserve rationale still needed to understand the mechanism.
- State an error in the user's terms, include the relevant value when safe, and give a useful next step when recovery is unclear.
- Name what a TODO requires. Do not defer broken core behavior behind a TODO.

When flagging prose, quote the phrase and explain the ambiguity, false claim, or reading cost. Give a replacement when useful. Do not infer authorship or reject text because it sounds machine-written. Report a repeated pattern once.
