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

State a negation only where readers really mix two things up, and once per page. A run of "X does A. It does not B."
sentences, or a chain of one-line definitions with no reason, reads machine-written.

Do not define a thing as a mysterious entity and then add effects. If we know the effects, that is the whole idea.

## One name per idea

If two phrases produce the same practice, they are the same belief. Keep one.

| Idea | Name |
|------|------|
| The Python file that declares tables and routes | application file (`app.py`) |
| Stored column vs computed column | annotation vs assignment |
| Where tables live | catalog directory (`my_app`) or `pxt://` URI |
| The HTTP process | service (`pxt service`) |
| Local first run | `pxt schema update`, then `pxt service update`, then insert or POST |
| Cloud first run | `pxt db update`, then schema, then service, against `pxt://` |
| `pxt schema update` | Creates tables. Does not start HTTP |
| `pxt service update` | Starts HTTP. Does not create tables |
| `pxt db update` | Hosted database only: image, secrets, capacity. Does not insert rows or start app HTTP |
| Change a computed column | `pxt schema update`, then `pxt service update`, then `pxt recompute` for existing rows |
| The background process that runs the local database | daemon (`pxt daemon`) |

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

## README and overview pages

`README.md`, Why Pixeltable (`overview/pixeltable.mdx`), Quickstart (`overview/quick-start.mdx`) and How it works
(`overview/how-it-works.mdx`) are written for a developer deciding whether to try Pixeltable. On these pages:

- The README and Why Pixeltable may say why: what the stack looked like before, why we built it, and when Pixeltable
  is the wrong tool.
- The subject can be "you", "we" or Pixeltable, and a step can give its reason ("run them in that order, since a
  service needs its tables to exist") instead of a negation pair.
- The Quickstart may show one change loop after the first POST: `pxt schema diff`, `pxt schema update`,
  `pxt service update`, then `pxt recompute`.
- What reruns, how failures are stored and how versions work live in How it works. The README keeps a short list,
  and the other pages link there.

Their commands, code and names still follow this guide. Rules that keep a coding agent on track (stale skill,
notebook vs app, port discovery) belong in the skill, `get-started.md` and `resources/ai-skills.mdx`; the README
points to [AI coding agents](https://docs.pixeltable.com/overview/building-pixeltable-with-llms) instead.

## Next cards

One next action, or a fork with distinct effects (local HTTP vs `pxt://`). Do not list five cards that reopen the same doubt.

Prefer Mintlify components over a mermaid flowchart for three layers: `Columns` of `Card`s, `FileTree`, `Steps`, `Tabs`, `AccordionGroup`. Use mermaid only when the edges themselves are the idea.

## Test what you print

Every command, flag, output sample and behavior claim on a page comes from a run against the current PyPI release.
When a step can't run (no API key, no Cloud account), say so in the PR. Counts match their lists: "four systems"
next to a list of five is a defect.

## Standing bans

- No em dashes (U+2014). Use a period, a colon, or a comma. Empty placeholders use ASCII `-`. This applies to MDX, notebooks, READMEs, the site, the skill, and the kit.
- No Convex product nouns.
- No emoji headings, `**Bold lead** - claim` bullet walls, slogan triplets, "not just X, it's Y", or adjective stacks
  (seamless, powerful, comprehensive).
- Pixeltable words: `TableModel`, catalog, same file. Name the command, not Declare / Experiment / Serve / Pack.
