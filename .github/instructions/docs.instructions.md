---
applyTo: "docs/**,README.md"
---

No CI job checks prose, so review is the only place these are caught.

## Prose

- No em dashes (U+2014). Use a period, a colon, or a comma. ASCII `-` for empty placeholders.
- Name the command and say what it does: `pxt schema update` creates tables and does not start HTTP; `pxt service update` starts HTTP and does not create tables. Do not label the loop Declare / Experiment / Serve / Pack on a user-facing page.
- One name per idea. "Application file", "schema file", and "the file" are not three objects.
- No emojis unless asked for. Sentence case for headings.

## Notebooks

- Exactly one title source: either a raw cell with YAML frontmatter, or a leading H1 that Quarto converts. Flag a notebook carrying both, which renders a double title. Do not flag a leading H1 on its own; 93 of 100 notebooks use one.
- Code cells format at line length **74**, not the 120 that applies to `.py` files (`scripts/check-notebooks.sh`).
- At least 50% of code cells must have outputs (`tool/check_notebooks.py`). Never advise clearing all outputs.
- Markdown cells must be `nbqa mdformat` clean. Use `raw.githubusercontent.com`, never `raw.github.com`.
- No badge images in markdown cells. Kaggle/Colab/download links belong in the frontmatter `description`.
- Schema ops in examples must use `if_exists='ignore'` / `if_not_exists=True`.

## Accuracy traps

- Pixeltable reads the process environment only. It never loads `.env`, and `python-dotenv` is not a dependency. Flag any page telling a reader to put a key in `.env` without also saying to source or export it.
- `~/.pixeltable/` paths must not appear in user-facing text.
- A page describing changed behavior must be updated with it. Flag prose that contradicts the code in the same PR.
