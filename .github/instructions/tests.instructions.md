---
applyTo: "tests/**"
---

- New features need tests here; the tree mirrors `pixeltable/`.
- A test takes `db_root: DatabaseRoot` and builds paths with `db_root.make_catalog_path(...)`. Never hardcode a catalog path. 586 tests already do this.
- A test using `uses_db` needs `@pytest.mark.db_roots('local', reason='...')`; `tests/conftest.py` raises a `UsageError` at collection without it. The reason names a PXT ticket, not a bare TODO. Delete the exclusion when the ticket lands.
- Any test for `pxt.Error` or a subclass uses `pxt_raises()`, not `pytest.raises()`. Both always take `match=` to verify error text.
- Assert on user-visible behavior through the public API, not `col.stored`, `ColumnRef`, or `TableVersion` internals. Use `Table.get_metadata()`, `t.describe()`, or queries.
- AI provider tests go in `tests/functions/test_<provider>.py`, marked `remote_api`. Anything hitting a third-party model or service also needs `very_expensive`.
- Never dodge one backend with a bare `@pytest.mark.skip`. Scope it with `db_roots`. Register any new marker in `pyproject.toml`.
- No `http://` or `https://` literals. Use the `sample_file_server` fixture, which serves the repo tree over localhost and still exercises the download path.
- A `skipif`/`xfail` on a test parametrized over catalog mode disables every variant. Gate inside the test body when the reason names one parametrization.
- Prefer extending an existing parametrized test over adding a bespoke one beside it.
- Test names must be specific enough for `pytest -k`. Prefer `pytest.parametrize` over duplication.
- Use the shared utilities (`validate_update_status()`, `skip_test_if_not_installed()`, `ReloadTester`) and extend shared fixtures rather than duplicating setup.
- Put markers on the class or method, not in a module-level `pytestmark`.
