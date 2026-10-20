# Brief

The decision, after the card exists. With no `lifecycle/job.md`, stop and tell them to run cast. Do not invent the person inside the brief.

Read [floor.md](floor.md), [card.md](card.md), and [questions.md](questions.md). Read `lifecycle/session.md`, `lifecycle/research.md`, and `lifecycle/critic.md` when they exist.

## What may enter

- The person, the job, done, and leave come from the card.
- A Fact from `lifecycle/research.md` may be stated, with its source.
- An Inference stays under "Check before building."
- A critic finding stays under "Check before building." It is not silently repaired.
- A product change comes from a row in the session table. If several rows are Flow, the change is the one that blocks Done. The other Flows stay listed as still open.
- If there is no session, the change section says the session has not been run.

## Write it like this

### The job

The card, in a few lines. This is the problem they arrived with.

### The four questions

One row per question in [questions.md](questions.md).

| Domain | Question | Answer | Where |

Where is a file, a step, or `open`. An open row has no invented answer in the Answer cell. Write `open` there too.

### What broke

The session table, or a line that the session has not been run.

### Problems this creates

The new problems from the session, after they succeed. Each one is already helped, in the product's words, or open. If the session never reached success, say that.

### The change

One change, or none. Name the place and the words. This is the row that blocks Done. Name what you deferred, and why it can wait.

### Check before building

Inferences, critic findings, troubles the session could not produce, and every question marked open.

Save it to `lifecycle/brief.md`. The reply is that brief. Ask whether to change the product. Wait.
