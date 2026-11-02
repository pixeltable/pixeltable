---
name: walk-the-job
description: >
  Walk one person through one job in a product, quote what the product
  shows, and note where the job breaks. Use when asked to walk a flow,
  dogfood a feature, try an empty state or an error, say who it is for,
  research a fact the job depends on, criticise the writeup, or write the
  decision from what survived. Commands: walk, cast, trouble, research,
  critic, brief. The product supplies the words. Do not use for a PRD, a
  roadmap, lifecycle email, or account administration.
license: MIT
metadata:
  author: Pierre Brunelle
  version: 0.7.0
---

# Walk the job

Sit with one person while they try to do a job. You are the product manager. The product in front of you is the one that was built. Every word they follow is a word the product shows. They cannot see a sentence you wrote for them.

The commands share one job card. A later command keeps the person and the job, and reads the product again because the path may have moved.

A good product solves the problem this person arrived with. A great one, once that job is done, hands them a new set of problems, and those problems are now the product's. [references/questions.md](references/questions.md) is the ten questions a brief has to answer, and the rule for the new problems. An open question stays open.

## Commands

| Command | What it does | Reference |
| --- | --- | --- |
| `walk` | Full session, from arrival until they finish or give up | [references/walk.md](references/walk.md) |
| `cast` | The job card only | [references/cast.md](references/cast.md) |
| `trouble` | The break: empty, wrong, wait, closed, return, undo | [references/trouble.md](references/trouble.md) |
| `research` | Facts outside the product that this job depends on, each with a source | [references/research.md](references/research.md) |
| `critic` | Attack the session or the brief. Change nothing | [references/critic.md](references/critic.md) |
| `brief` | The decision, from the card and the facts that survived | [references/brief.md](references/brief.md) |

Read the reference for the command before you start. Read [references/card.md](references/card.md) whenever the command uses the job card.

Routing:

- They named a command, or the ask is clearly one row of the table: run that command.
- They named rows from a session table: the second pass in [references/walk.md](references/walk.md).
- A short ask to walk, dogfood, or confirm the experience: `walk`.
- No command, and a product is open: `walk`.
- No product is open: ask which product and which job. Do not invent one.
- Two commands fit: ask once.

You may hand `research` to the lifecycle-researcher subagent and `critic` to the lifecycle-critic subagent. The reference is the same either way.

## The job card

The card is `lifecycle/job.md` in the product's project. [references/card.md](references/card.md) is its shape.

If the file exists, use its person, job, done, and leave. If a command needs a card and the file is absent, write the card from the product and save the file. `brief` is the exception: with no card, it stops.

`research` and `critic` leave the product as they found it. `critic` leaves the job card as it found it. `research` writes the card only when the file is absent.

Change the product only when they asked you to fix what you find.

## Shared rules

- One person. A second person leaves you with no single place the job stops.
- Quote the product. Name the file, the page, the command, or the screen.
- When two places disagree, keep both.
- A change names that place and the words to add, move, or remove.
- Drop a change that still fits if you swap in another product.
- [references/floor.md](references/floor.md) is the picture of a writeup that looks finished and leaves nothing to point at. Read it before `walk`, `trouble`, `critic`, and `brief`.
- [references/questions.md](references/questions.md) is the ten questions. Read it before `walk`, `research`, `critic`, and `brief`.
