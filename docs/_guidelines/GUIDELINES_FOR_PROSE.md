# Guidelines for MDX prose

**Purpose:** Make every important word in `docs/release/*.mdx` mean a habit of action. Notebooks follow this plus [`GUIDELINES_FOR_NOTEBOOKS.md`](GUIDELINES_FOR_NOTEBOOKS.md).

Source: [How to Make Our Ideas Clear](https://www.peirce.org/writings/p119.html) (Peirce, 1878). A few clear ideas beat many confused ones.

## The test

For each important word on a page, the reader must be able to say what they would **do differently** if the word were false.

`TableModel` means: a class in `app.py`, then `pxt schema update`, then insert. If that is true, do not also need "declarative multimodal backend."

If a sentence does not change what the reader types, runs, or gets on insert, it is not part of the thought. Delete it.

## Three grades, in order, then stop

1. **Familiarity.** Show the thing: the application file, a curl, a command.
2. **Distinctness.** A short definition only when the reader must not mix two things up (annotation vs assignment; `schema update` vs `service update`).
3. **Effects.** What happens when you act. `pxt schema update` creates tables. It does not start HTTP.

Do not define a thing as a mysterious entity and then add effects. If we know the effects, that is the whole idea.

## One name per idea

If two phrases produce the same practice, they are the same belief. Keep one.

| Idea | Name |
|------|------|
| The Python file that declares tables and routes | application file (`app.py`) |
| Stored column vs computed column | annotation vs assignment |
| Where tables live | catalog (local directory or `pxt://` URI) |
| The HTTP process | service (`pxt service`) |
| Local first run | `pxt schema update`, then `pxt service update`, then insert or POST |
| Cloud first run | `pxt db update`, then schema, then service, against `pxt://` |
| `pxt schema update` | Creates tables. Does not start HTTP |
| `pxt service update` | Starts HTTP. Does not create tables |
| `pxt db update` | Hosted database only: image, secrets, capacity. Does not insert rows or start app HTTP |

On user-facing pages the **command is the subject**. Declare / Experiment / Serve / Pack are an internal map only. Never write "It is not Experiment." Never write "Pack, declare, then serve." "Apply" is not a command: write `pxt schema update`. "TARGET" is a CLI metavar: on first-run pages say "last argument" (`my_app`, `agent`, or `pxt://org:db`). Do not write "an application's contract."

Do not write "schema file," "the file," and "application contract" as if they were three objects.
Do not name the loop Store, Transform, Serve. Do not name the middle phase Insert.

## One doubt per page

A page settles one question: what to type, what command to run, what happens on insert. Canonical homes:

- Why: identity (one application file; insert runs compute)
- Quickstart: first run
- How it works: schema, database, services, and the `diff` / `update` / `prune` verbs
- Coming from...: stack-shaped snippets
- Cloud: hosted target, secrets, runtime UDFs
- HTTP serving: route API

Everyone else is a pointer, not a second copy.

## Next cards

One next action, or a fork with distinct effects (local HTTP vs `pxt://`). Do not list five cards that reopen the same doubt.

Prefer Mintlify components over a mermaid flowchart for three layers: `Columns` of `Card`s, `FileTree`, `Steps`, `Tabs`, `AccordionGroup`. Use mermaid only when the edges themselves are the idea.

## Standing bans

- No em dashes (U+2014). Use a period, a colon, or a comma. Empty placeholders use ASCII `-`. This applies to MDX, notebooks, READMEs, the site, the skill, and the kit.
- No Convex product nouns.
- Pixeltable words: `TableModel`, catalog, same file. Name the command, not Declare / Experiment / Serve / Pack.
