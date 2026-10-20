# Example

This is the grain of a short session. The person, the product, and the labels belong to this page. Use none of them on the product in front of you.

The product is a one-screen settings page. The session was read from `src/settings.tsx`, because the page needs an account.

## Who, and the job

- Person: Sam, who has a call in ten minutes and wants the app quiet during it.
- Arrival kind: finish a job.
- Job: turn quiet hours on from 2pm to 4pm today.
- Done: the screen shows quiet hours on, with those times.
- Leave: Sam cannot tell whether the switch took, or whether the hours are those two.
- Path: `Settings`, then the row `Quiet hours`.
- Stages this product does not have: arrival from outside. Sam is already in the app. The session starts at entry.

## What I checked

- `src/settings.tsx` renders the title `Settings` and a row labeled `Quiet hours` with a switch. Watched: no. Read from source.
- The switch handler writes `quietHours` and renders no confirmation string. I looked for a toast, a status line, and a changed label. The file has none of them.
- I looked for a trouble of the kind Wrong. The switch takes on and off, and there is no value the page can refuse.

## The session

### Sam tries to turn quiet hours on

1. **Entry.** Sam opens `Settings`. The first row under the title is `Quiet hours`, with a switch in the off position. Sam can see the action that matches the job.
2. **First success.** Sam turns the switch on. The switch moves to on. The row still reads `Quiet hours`. Offering quiet hours handed Sam a new problem: which hours. The screen has no control for 2pm or 4pm, so that problem is open. Sam cannot tell the job is done, because the hours were the job.
3. **Trouble, return.** Sam leaves and opens `Settings` again. The switch is on. The row still reads `Quiet hours`. Nothing on the screen says the hours. Sam has to trust that "on" means 2pm to 4pm, and the product never said that.
4. **Leaving.** The switch state is kept in `quietHours`. The screen never says the work was kept, or where.

## Where the job broke

| # | Moment | What they needed | What the product did | Change | Size |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | A way to set 2pm to 4pm, and a line that reads those times back | The switch turns on. The row still says `Quiet hours`. | On the `Quiet hours` row, add start and end, and under the switch a line that reads the times back once it is on. | Flow |

Ask which rows to change. Wait.

The row is Flow because the job cannot be finished from this screen. A new label on the same switch would leave the times missing. That sentence is for you. Leave it out of the writeup.
