# Prose checks

Apply these checks in two places:
- every comment, docstring, error message and doc line in the diff;
- every sentence of your own review.

A sentence that fails a check below is a defect, not a nit. Quote the phrase and state which check it
fails. Example: `"where media that names no destination goes": the noun phrase ends in a verb; say "media
without a destination"`. Flag a pattern once, with "(here and everywhere else in this PR)".

## Sentence checks

Run these in order on each sentence:

1. **Read it aloud.** If you cannot say what happens in one breath, rewrite it; reflowing it does not help.
2. **Find reduced relative clauses.** A noun phrase that ends in a verb is a relative clause with the
   pronoun dropped: "the type its column declares", "the version the client computes", "the models a
   service serves". Rewrite it as a possessive or a prepositional phrase: "its column type". Scanning for
   "that" or "which" does not find these.
3. **Cut qualifiers the context supplies.** Examples: "brand new", ", as far as it needs any", "this
   process serves".
4. **Use the concrete referent.** At least one noun must be the concrete method, class or field, not a
   category ("member", "the model"). Use the identifier itself. Every function reference in a comment or
   docstring ends in `()`.
5. **Define every term.** Jargon from another system, a term absent from the codebase, or an unexplained
   path, prefix or key needs a plain description or a question ("What's X?").
6. **Surface checks:**
   - No preposition stranded at the end of a clause.
   - No noun-phrase pileup of the form "the X a Y is Z to".
   - No fused emphatic: "X is what makes Y work" becomes "X makes Y work".
   - No "holds" or "carries" where "has" or "contains" is meant. No "names" where "refers to" or
     "specifies" is meant. Nothing of the form "...that has none".
   - No vacuous terms: footgun, load-bearing, happy path, self-heal(ing), envelope, leaf, knob, magic,
     under the hood, seam, or any coined term absent from the codebase. Describe the concrete behavior
     instead.
   - No marketing words in user docs: easily, efficient, high-throughput, seamless.
   - ASCII only: no em or en dashes, smart quotes, arrows or ellipsis characters.

## Comment checks

- **One fact per comment.** Keep it to one line where possible, stated once, at the site that does the
  thing. Cut the "so that ..." clause when the code shows it. A comment wrapping past two lines is a
  prompt to cut.
- **State the constraint that forces the code,** with the exact identifier that imposes it. Good:
  "create_table_version_md() requires IndexSpec.indexed_column to be a Column, not str". Never a narrative
  of how the value got there.
- **Describe only the code at hand.** No caller or caller intent, no other function's internals, no
  history, no product decisions, and no claim beyond the code's guarantees.
- **No docstring that paraphrases the signature.** A property whose name tells the whole story needs none.
- **Internal docstrings cover only non-obvious points,** with no Args or Raises boilerplate.
- **No backticks in code comments or internal docstrings** unless the file already uses them. Rendered
  public docstrings use single backticks.
- **Delete a comment whose subject no longer exists.**

## Docstring checks

- The first line states the action on the parameters, in the imperative: "Create a module that represents
  dir as a package", not a noun phrase about the return value.
- The body states what happens, in the affirmative. It never explains by listing what is absent.
- No undefined placeholder name.
- An abstract method's docstring describes what it is, not one implementation's mechanics.

## TODO checks

A TODO states what has to happen, not how the problem arises. Good: "TODO: Catalog needs to discard cached
TableVersion that references reloaded modules".

## Examples

- "'seam': don't leave llm-generated verbal spew in the code." (#1440)
- "Garbled/backwards LLM grammar, please fix." (#1593)
- "\"this process serves\": please remove vacuous filler commentary (what process if not this one?) left by
  Claude (here and everywhere else in this PR)." (#1604)
- "This looks like LLM trash, commenting on caller logic. Nothing in this function guarantees that it runs
  outside any transaction. If this is critical, move the comment to the call site." (#1545)
- "\", as far as it needs any\" doesn't add anything (it already says \"that idx needs\" earlier)." (#1515)
- "This narrates a product decision, which has no place in a docstring." (#1600)
- "Remove; we don't need a historical account." (#1586)
- "Is this llm output? What is the point of the \"(enforced by ...)\"?" (#1096)
- "Don't comment on caller logic, those things can change." (#1324)
