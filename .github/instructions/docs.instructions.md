---
applyTo: "docs/**,README.md"
---

# Documentation review

- Verify commands and examples with current APIs. `pxt schema update` creates tables; `pxt service update`
  starts HTTP; `pxt service run` is local. `pxt serve` does not exist. Keep naming consistent and update
  prose when behavior changes. Read the relevant `docs/_guidelines/` file for prose, docstrings or recipes.
- Pixeltable does not load `.env` automatically. Examples must export/source variables or explicitly
  configure a loader. Never include real credentials. Supported storage/configuration docs may describe
  `~/.pixeltable/`; application quickstarts should use public APIs rather than implementation paths.
- Make examples rerunnable through scoped setup/reset or current `if_exists` options. Preserve intentional
  failure/replacement examples; do not force `ignore` when an existing schema may be incompatible.
- Notebook titles use either first-cell raw YAML with `title` or a leading H1, never both. Retain outputs on at least
  50% of code cells (`tool/check_notebooks.py`); remove noisy progress/warnings only.
- Notebook code uses 74-column formatting and Markdown uses `nbqa mdformat` (`scripts/check-notebooks.sh`).
  Use `raw.githubusercontent.com` for raw GitHub links. The build generates badges and open/download links.
- Check rendered MDX and navigation. Use sentence-case headings, concrete command names and no unsolicited emojis.
