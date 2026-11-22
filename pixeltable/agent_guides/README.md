# Agent guides

The guides in this directory tell a coding agent how this release of Pixeltable behaves. They ship in the wheel,
so an agent reads the guides of its installed version.

This file is for whoever changes Pixeltable: it says what a guide contains, which guides exist, and when a change
has to update one.

## The split with the skill

An agent that builds an application reads two sources:

- The Pixeltable skill ([pixeltable/pixeltable-skill](https://github.com/pixeltable/pixeltable-skill)) teaches
  judgment that does not change between releases: how a conventional design maps to tables, computed columns and
  routes, how to design a schema, which stacks to avoid, and which changes need the user's approval.
- The guides state what this release does: what an API does, its defaults, edge cases, errors and limits, and its
  known issues, with their workarounds.

Neither repeats signatures, parameter lists or flag lists: the docstrings and `pxt COMMAND --help` give those for
the installed version.

## Which guides exist

One guide per functional area of the public API: the APIs that an agent uses together for one job. `tables.md`
covers declaring tables, writing rows, schema changes and versions; the other guides cover expressions, queries, views,
indexes, UDFs, Json, providers, serving, the CLI and Cloud.

Add a guide when a new area does not fit an existing one. Split a guide when it no longer reads as one job.

## What a guide contains

- Behavior that a signature does not show: what is stored or recomputed, what is atomic, what runs in which order,
  which defaults matter, and the errors, with their messages.
- Known issues, each with its ticket id, until the fix ships.
- Examples in application form (`TableModel` classes) first, and in notebook form where that form differs.

A guide leaves out design advice, which belongs in the skill, and version history, since each wheel ships its own
guides: write what this release does, not since when.

## The review rules

`review.md` is not a guide to an area: it contains the rules for `pxt lint`'s review of a project's code, and
`pxt guide review` prints them. A rule describes Python code that redoes Pixeltable's work or uses Pixeltable wrongly,
and gives the reason, the fix and the exceptions. The rules ship with the release rather than in the skill because each
fix uses this release's APIs and costs. Add a rule, or an exception to one, when an eval or a user report shows a new
case. Code in a rule is indented rather than fenced, since the tests run fenced blocks.

## When to update a guide

- A pull request that changes user-visible behavior updates the guide of each area it touches, in the same pull
  request.
- A fix for a known issue removes or rewrites its entry.
- A new public API area gets a guide.

## Format and tests

- Markdown in ASCII, one fact per bullet, in plain sentences.
- The first line after the title summarizes the guide; `pxt guide` lists it.
- A reference to another guide is its command, such as `pxt guide tables`, since an agent may read the guide from
  `pxt guide` or from its file, which `pxt guide --path` locates.
- `tests/test_agent_guides.py` runs the `python` blocks of each guide in order, as one module, against a new
  catalog, and then calls `create_all()` on each `TableModel` base in the module. Blocks can define UDFs,
  application classes and notebook tables. They must not insert rows into a column that calls a provider: the test
  has no keys. A guide that imports a package missing from the environment is skipped.
- The same test checks every `pxt` line of a `bash` block: the command exists, and so does each flag on the line.
