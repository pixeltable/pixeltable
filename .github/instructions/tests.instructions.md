---
applyTo: "tests/**"
---

# Test review

- Cover changed behavior through public SDK/CLI/HTTP results, metadata or errors. Use internal APIs only
  when the behavior is impractical to exercise publicly. Require regression coverage for concrete risks,
  not implementation-mirroring tests for every edit.
- Prefer `db_root: DatabaseRoot` and `db_root.make_catalog_path()` for portable catalog tests.
  `uses_db` and plain paths are valid for in-process tests; they require
  `@pytest.mark.db_roots('local', reason='...')`. The reason explains the restriction or is `TODO: convert`.
- Use `pxt_raises(..., match=...)` for Pixeltable errors, and `pytest.raises(..., match=...)` for other
  exceptions. Verify meaningful error text.
- Mark new tests that make live third-party model/service calls or download models `skip_merge_queue`,
  unless the service is reliable enough for the merge queue (as OpenAI's is). Independently, mark tests
  too costly for every CI configuration `expensive` (for example, image generation; runs only on
  `standard+`/`standard++`), or `very_expensive` (for example, video generation; `standard++` only). Tests
  that exceed a standard GitHub-hosted runner's memory or disk take `resource_intensive`. Register new
  markers in `pyproject.toml`; put new markers on classes/methods.
- Scope backend restrictions with `db_roots`; gate a single parametrization inside the test instead of
  applying `skipif`/`xfail` to every variant.
- For catalog-parametrized media tests, use `sample_file_server.url(path, db_root)` so hosted modes get
  reachable URLs. Explicit HTTP URLs remain valid for protocol fixtures and URL-handling tests.
  Isolate auth/config state from the developer's credentials.
- Keep database-dump `-info.toml` metadata consistent with its matching `.dump.gz` fixture.
- Extend shared fixtures/utilities and parametrized tests where practical; keep names useful for `pytest -k`.
