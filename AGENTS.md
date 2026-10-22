# Pixeltable contributor instructions

Pixeltable is a Python library for incremental multimodal tables, computed columns, views, indexes and queries.
These rules govern library contributions. Application builders should use the
[Pixeltable skill](https://github.com/pixeltable/pixeltable-skill) and starter-kit examples.
Application guidance belongs in that skill repo; `docs/release/skill.md` is only a pointer.

## Scope and context

Before editing or reviewing a matching area, read its scoped rules below. Consult only relevant sections
of longer guides; Codex and Claude do not automatically apply Copilot's path-specific files.

| Area | Read |
|---|---|
| Docs or README | `.github/instructions/docs.instructions.md` |
| HTTP, CLI, authentication, hosted catalogs, app loading or packaging | `.github/instructions/serving.instructions.md` |
| Tests or test coverage | `.github/instructions/tests.instructions.md` |
| User-facing prose | `docs/_guidelines/GUIDELINES_FOR_PROSE.md` |
| Docstrings | `docs/_guidelines/GUIDELINES_FOR_DOCSTRINGS.md` |
| Notebooks / new cookbook recipes | `docs/_guidelines/GUIDELINES_FOR_NOTEBOOKS.md` / `docs/_guidelines/GUIDELINES_FOR_COOKBOOKS.md` |
| Dashboard or dashboard-facing server APIs | `dashboard/DESIGN.md` and `dashboard/ARCHITECTURE.md` |
| Environment setup or PR workflow | `CONTRIBUTING.md` and `Makefile` |

## Development

- Keep changes surgical and preserve unrelated work. Read affected call sites and existing tests before adding abstractions.
- Use Python >=3.11, complete annotations and the project's mypy configuration. Follow existing naming;
  Ruff defines formatting (120 columns, single quotes). Aggregate class names intentionally use lowercase.
- After code changes, run `make format`, `make check` and relevant tests before completion. These targets depend
  on `install`; in a shared environment use equivalent direct checks instead of reinstalling dependencies.
  `make check` includes mypy, Ruff and notebook checks. Report checks run and any omissions.
- Review the entire task diff, including new files, before completion. Keep comments only for constraints,
  non-obvious choices or invariants; update comments, help and docs when their described behavior changes.
  Use concrete, concise prose, `function()` for function references and ASCII typography in new prose.
- Public docstrings generate SDK MDX: use `>>>`/`...` prompts, paired backticks and self-closing void HTML tags.

## Invariants

- `integrations.telemetry.enabled` in `docs/release/docs.json` must remain `true`.
- Never commit real credentials or expose secrets or private diagnostic paths in user-facing responses.
- Agents must not manually edit users' `~/.pixeltable/` state. Use supported SDK/CLI operations;
  library code implementing storage is not prohibited from writing its managed files.
- Agents may deploy docs only with `make docs-deploy TARGET=dev`; `stage` and `prod` are for humans.
- Validate user input with a specific `pixeltable.exceptions.Error` subclass and matching `ErrorCode`;
  constructing `Error` itself or using a mismatched code asserts. Assertions are for internal invariants.
- In expression contexts, use overloaded operators: `t.x == None`, `t.active == False`, `&`, `|`, `~`.
  Python `is None`, `and`, `or`, `not` evaluate Python truth rather than build Pixeltable expressions.
  Ordinary Python checks remain valid outside the DSL.
- UDF annotations determine column types. Non-nullable parameters short-circuit null arguments to null;
  nullable parameters let the UDF handle null. Review schema semantics even if mypy passes.
- Application files declare `TableModel` tables and `FastAPIRouter` routes; indexes belong in `__indexes__`.
  Tests, notebooks and REPLs may use `pxt.create_table()` and incremental schema APIs.
- Model columns are non-nullable by default (`pxt.String`); nullable columns use `pxt.String | None`.
  Public index metadata uses `TableMetadata.indexes`, not `indices`. Check current APIs before suggesting replacements.

## Code Review Rules

- Prioritize correctness, data loss, security, concurrency and hosted behavior over prose or style.
  Trace relevant callers and tests; report only actionable defects introduced by the diff, with a concrete
  trigger, impact and smallest useful line range. Group duplicate findings; no findings is a valid result.
- Leave formatting and statically detectable type errors to CI. Flag misleading contracts or UDF schema
  annotations when they change runtime behavior. Do not request unrelated cleanup or speculative safeguards.
- Check transaction/cache boundaries, incremental recomputation, nullable inputs and resource ownership
  where affected. Verify runtime-introspection claims before reporting them.
- For persisted metadata changes, inspect `metadata.VERSION`, converters and compatibility tests before
  claiming a breaking change; a field rename alone does not prove a migration is required.
- Check affected contracts, not a fixed co-change checklist: public SDK docs (`docs/public_api.opml`,
  `docs/release/docs.json`), CLI help/docs, dashboard response types/consumers, provider model examples,
  dependency lockfiles and migration fixtures. Require updates only where the change makes them stale.
