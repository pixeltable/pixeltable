"""Smoke tests: drive state via the pxt API, validate via pcli.

Each class focuses on one pcli command. Methods bundle related scenarios so a single
table can serve multiple assertions (JSON + text + flag variants) without re-creating it.
"""

import json
import os
import urllib.error
import urllib.request

import numpy as np
import pytest

import pixeltable as pxt
from tests.utils import get_image_files

from .conftest import PcliRunner


@pxt.udf
def _boom_on_zero(x: int) -> int:
    """Module-level UDF: raises for k=0 so we can populate a stored errortype column."""
    if x == 0:
        raise ValueError('boom')
    return x


@pxt.udf
def _trivial_embed(s: str) -> pxt.Array[(8,), np.float32]:
    """Module-level embedder for embedding-index tests: deterministic, no model download."""
    return np.zeros(8, dtype=np.float32)


class TestHealth:
    def test_basics(self, pcli: PcliRunner, pcli_daemon: int) -> None:
        # main health endpoint: always JSON
        out = pcli('health').json
        assert out['ok'] is True
        assert out['pid'] > 0
        # legacy dashboard probe lives at /api/pixeltable-health; not reachable through the CLI.
        # The shape matches pixeltable.dashboard.server's response: status + version, so existing
        # clients (which read both fields) keep working.
        with urllib.request.urlopen(f'http://127.0.0.1:{pcli_daemon}/api/pixeltable-health') as r:
            body = json.loads(r.read())
        assert body['status'] == 'ok'
        assert body['version'] == pxt.__version__


class TestLs:
    def test_lists(self, pcli: PcliRunner) -> None:
        """Bare ls (text + json) lists what's in the catalog and reflects mutations."""
        pxt.create_dir('pcli_ls', if_exists='ignore')
        pxt.create_table('pcli_ls.t', {'x': pxt.Int}, if_exists='replace')

        # json reports the entry
        entries = pcli('ls', 'pcli_ls', '--json').json['entries']
        assert 'pcli_ls/t' in {e['path'] for e in entries}

        # plain text lists the headers + path + kind
        text = pcli('ls', 'pcli_ls').stdout
        assert 'path' in text
        assert 'kind' in text
        assert 'pcli_ls/t' in text

        # an empty dir is still a valid listing target
        pxt.create_dir('pcli_ls_empty', if_exists='ignore')
        assert pcli('ls', 'pcli_ls_empty').returncode == 0

        # after a drop, the entry is gone from the listing
        pxt.drop_table('pcli_ls.t')
        entries = pcli('ls', 'pcli_ls', '--json').json['entries']
        assert 'pcli_ls/t' not in {e['path'] for e in entries}

    def test_long_and_metadata(self, pcli: PcliRunner) -> None:
        """-l / -l --json populate num_cols and flags via get_metadata(). Bare --json is the
        cheap path: it skips the per-entry metadata fetch and returns num_cols=None."""
        pxt.create_dir('pcli_ls_long', if_exists='ignore')
        t = pxt.create_table('pcli_ls_long.t', {'a': pxt.Int}, if_exists='replace')
        t.add_computed_column(b=t.a * 2)

        # -l text: headers + 'c' flag for computed column
        text = pcli('ls', 'pcli_ls_long', '-l').stdout
        assert all(h in text for h in ('cols', 'version', 'flags'))
        assert 'c' in text

        # -l --json: details populated
        entries = pcli('ls', 'pcli_ls_long', '-l', '--json').json['entries']
        row = next(e for e in entries if e['path'] == 'pcli_ls_long/t')
        assert row['num_cols'] is not None
        assert row['num_cols'] >= 2
        assert 'c' in row['flags']

        # bare --json: cheap-by-default, no metadata fetch
        entries = pcli('ls', 'pcli_ls_long', '--json').json['entries']
        row = next(e for e in entries if e['path'] == 'pcli_ls_long/t')
        assert row['num_cols'] is None
        assert row['flags'] == ''

    def test_tree_and_counts(self, pcli: PcliRunner) -> None:
        """--tree formats the nested catalog with ASCII prefixes. --counts populates
        num_rows in both text and JSON; a dirs-only target skips the count pool entirely."""
        pxt.create_dir('pcli_ls_tree', if_exists='ignore')
        pxt.create_dir('pcli_ls_tree.sub', if_exists='ignore')
        pxt.create_table('pcli_ls_tree.sub.leaf', {'a': pxt.Int}, if_exists='replace')
        t = pxt.create_table('pcli_ls_tree.t', {'a': pxt.Int}, if_exists='replace')
        t.insert([{'a': i} for i in range(4)])

        # --tree: nested entries and a box-drawing prefix
        tree_text = pcli('ls', 'pcli_ls_tree', '--tree').stdout
        assert all(marker in tree_text for marker in ('sub', 'leaf'))
        assert any(prefix in tree_text for prefix in ('|--', '\\--'))

        # --counts json: num_rows populated for the table
        entries = pcli('ls', 'pcli_ls_tree', '--counts', '--json').json['entries']
        row = next(e for e in entries if e['path'] == 'pcli_ls_tree/t')
        assert row['num_rows'] == 4

        # --counts text: includes the 'rows' column header
        counts_text = pcli('ls', 'pcli_ls_tree', '--counts').stdout
        assert 'rows' in counts_text
        assert '4' in counts_text

        # --counts on a dirs-only target: _fill_counts sees an empty target list and short-circuits
        pxt.create_dir('pcli_ls_dirs', if_exists='ignore')
        pxt.create_dir('pcli_ls_dirs.sub1', if_exists='ignore')
        pxt.create_dir('pcli_ls_dirs.sub2', if_exists='ignore')
        entries = pcli('ls', 'pcli_ls_dirs', '--counts', '--json').json['entries']
        assert all(e['kind'] == 'dir' for e in entries)
        assert all(e.get('num_rows') is None for e in entries)

    def test_errors(self, pcli: PcliRunner) -> None:
        """ls distinguishes 'path does not exist' (404) from 'path exists but is not a directory'
        (422). The latter case names the offending component and its kind so the user can fix
        a typo like `pcli ls my_table` vs `pcli describe my_table`."""
        pxt.create_dir('pcli_ls_err', if_exists='ignore')
        pxt.create_table('pcli_ls_err.t', {'a': pxt.Int}, if_exists='replace')

        # name absent at the root -> 404 path-not-found
        r = pcli('ls', 'does_not_exist', check=False)
        assert r.returncode != 0
        assert 'does not exist' in r.stderr.lower() or 'not found' in r.stderr.lower()

        # name exists at the leaf but is a table, not a directory -> 422 with the kind named
        r = pcli('ls', 'pcli_ls_err/t', check=False)
        assert r.returncode != 0
        assert "'pcli_ls_err/t' is a table, not a directory" in r.stderr

        # a table appearing mid-path: the error names the table-rooted prefix, not the full path
        r = pcli('ls', 'pcli_ls_err/t/sub', check=False)
        assert r.returncode != 0
        assert "'pcli_ls_err/t' is a table, not a directory" in r.stderr


class TestDescribe:
    def test_basics(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_desc', if_exists='ignore')
        pxt.create_table('pcli_desc.t', {'a': pxt.Int, 'b': pxt.String}, if_exists='replace')

        # --json returns the full get_metadata() dict
        meta = pcli('describe', 'pcli_desc/t', '--json').json
        assert 'a' in meta['columns']

        # text is repr(table) which lists the columns
        text = pcli('describe', 'pcli_desc/t').stdout
        assert 'a' in text
        assert 'b' in text

    def test_errors(self, pcli: PcliRunner) -> None:
        r = pcli('describe', 'does_not_exist', check=False)
        assert r.returncode != 0


class TestColumns:
    def test_lists(self, pcli: PcliRunner) -> None:
        """columns lists every column; `computed` is an alias for `columns --computed`.
        The no-path form walks every table in the catalog."""
        pxt.create_dir('pcli_cols', if_exists='ignore')
        t = pxt.create_table('pcli_cols.t', {'a': pxt.Int, 'b': pxt.String}, if_exists='replace')
        t.add_computed_column(doubled=t.a * 2)

        # --json: all three columns
        entries = pcli('columns', 'pcli_cols/t', '--json').json
        assert {e['column'] for e in entries} == {'a', 'b', 'doubled'}

        # text format: stored vs computed flag
        text = pcli('columns', 'pcli_cols/t').stdout
        assert 'a' in text
        assert 'stored' in text

        # `pcli computed` is the --computed shorthand
        entries = pcli('computed', 'pcli_cols/t', '--json').json
        assert {e['column'] for e in entries} == {'doubled'}

        # `pcli computed` text mode also works
        assert 'doubled' in pcli('computed', 'pcli_cols/t').stdout

        # no-path: walks _all_tables, exercises the try/except continue for unloadable metadata
        entries = pcli('columns', '--json').json
        assert ('pcli_cols/t', 'a') in {(e['table'], e['column']) for e in entries}


class TestIdxs:
    def test_lists(self, pcli: PcliRunner) -> None:
        """idxs runs against one table or globally; the no-path form walks every table."""
        pxt.create_dir('pcli_idx', if_exists='ignore')
        pxt.create_table('pcli_idx.t', {'a': pxt.Int}, if_exists='replace')

        # --json: no embedding idx exists on a plain table
        entries = pcli('idxs', 'pcli_idx/t', '--json').json
        assert all(e['index_type'] != 'embedding' for e in entries)

        # text: command runs cleanly regardless of catalog content
        assert pcli('idxs', 'pcli_idx/t').returncode == 0

        # no-path: walks every table; smoke check only since auto-indexes vary
        assert pcli('idxs', '--json').returncode == 0

    def test_embedding_filter(self, pcli: PcliRunner) -> None:
        """--embedding filters to embedding indexes only; a table with both a btree-style
        index and an embedding index reports one entry each, then only the embedding under
        --embedding."""
        pxt.create_dir('pcli_idx_emb', if_exists='ignore')
        t = pxt.create_table('pcli_idx_emb.t', {'s': pxt.String}, if_exists='replace')
        t.add_embedding_index('s', idx_name='emb_idx', string_embed=_trivial_embed)

        all_entries = pcli('idxs', 'pcli_idx_emb/t', '--json').json
        emb_only = pcli('idxs', 'pcli_idx_emb/t', '--embedding', '--json').json
        assert any(e['name'] == 'emb_idx' and e['index_type'] == 'embedding' for e in all_entries)
        assert all(e['index_type'] == 'embedding' for e in emb_only)
        assert any(e['name'] == 'emb_idx' for e in emb_only)


class TestHistory:
    def test_basics(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_hist', if_exists='ignore')
        t = pxt.create_table('pcli_hist.t', {'a': pxt.Int}, if_exists='replace')
        for i in range(4):
            t.insert([{'a': i}])

        # --json: every committed version
        versions = pcli('history', 'pcli_hist/t', '--json').json
        assert len(versions) >= 5  # create + 4 inserts

        # -n caps the result count to the last N versions
        capped = pcli('history', 'pcli_hist/t', '-n', '2', '--json').json
        assert len(capped) == 2

        # text: tab-separated header
        text = pcli('history', 'pcli_hist/t').stdout
        assert 'version' in text
        assert 'change_type' in text


class TestRows:
    def test_basics(self, pcli: PcliRunner) -> None:
        """rows: text + --json default; -n limit; --cols subset; null cells render as empty."""
        pxt.create_dir('pcli_rows', if_exists='ignore')
        t = pxt.create_table('pcli_rows.t', {'n': pxt.Int, 's': pxt.String}, if_exists='replace')
        t.insert([{'n': i, 's': f'row{i}'} for i in range(5)])

        # --json + -n
        out = pcli('rows', 'pcli_rows/t', '-n', '3', '--json').json
        assert len(out) == 3

        # --cols subset
        out = pcli('rows', 'pcli_rows/t', '-n', '10', '--cols', 'n', '--json').json
        assert sorted(r['n'] for r in out) == [0, 1, 2, 3, 4]

        # text contains the inserted values
        text = pcli('rows', 'pcli_rows/t').stdout
        assert 'row0' in text
        assert 'row4' in text

        # null cell renders as empty, not the literal 'None'
        t2 = pxt.create_table('pcli_rows.nulls', {'a': pxt.Int, 's': pxt.String}, if_exists='replace')
        t2.insert([{'a': 1, 's': None}])
        assert 'None' not in pcli('rows', 'pcli_rows/nulls').stdout

    def test_image_column(self, pcli: PcliRunner) -> None:
        """Image cells must render as `<Image WxH MODE>` in both text and JSON modes -
        not as raw bytes, base64, or a PIL repr."""
        img_paths = get_image_files()[:2]
        pxt.create_dir('pcli_img', if_exists='ignore')
        t = pxt.create_table(
            'pcli_img.t',
            {'k': pxt.Required[pxt.Int], 'img': pxt.Image, 'tag': pxt.String},
            primary_key='k',
            if_exists='replace',
        )
        t.insert([{'k': i, 'img': p, 'tag': f't{i}'} for i, p in enumerate(img_paths)])

        rows_json = pcli('rows', 'pcli_img/t', '-n', '5', '--json').json
        assert len(rows_json) == 2
        assert all(r['img'].startswith('<Image ') for r in rows_json)
        assert all(r['img'].endswith('>') for r in rows_json)

        rows_text = pcli('rows', 'pcli_img/t').stdout
        assert '<Image ' in rows_text
        assert 'PIL.' not in rows_text  # no PIL repr leaked
        assert '\\x' not in rows_text  # no raw bytes leaked

        # `pcli get` uses the same renderer
        get_json = pcli('get', 'pcli_img/t', '1', '--json').json
        assert get_json['row']['img'].startswith('<Image ')
        get_text = pcli('get', 'pcli_img/t', '0').stdout
        assert 'img\t<Image ' in get_text

    def test_errors(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_rows_err', if_exists='ignore')
        pxt.create_table('pcli_rows_err.t', {'a': pxt.Int}, if_exists='replace')

        # n must be >= 1 (pydantic Field constraint on RowsRequest.n)
        r = pcli('rows', 'pcli_rows_err/t', '-n', '0', check=False)
        assert r.returncode != 0
        assert 'greater than or equal to 1' in r.stderr

        # n upper bound: pydantic Field(le=1000)
        r = pcli('rows', 'pcli_rows_err/t', '-n', '10001', check=False)
        assert r.returncode != 0
        assert 'less than or equal to 1000' in r.stderr

        # unknown column
        r = pcli('rows', 'pcli_rows_err/t', '--cols', 'nope', check=False)
        assert r.returncode != 0
        assert 'unknown columns' in r.stderr


class TestGet:
    def test_basics(self, pcli: PcliRunner) -> None:
        """Single + composite PK; text + json + missing-row; --cols subset."""
        pxt.create_dir('pcli_get', if_exists='ignore')
        t = pxt.create_table(
            'pcli_get.t',
            {'k': pxt.Required[pxt.Int], 'a': pxt.Int, 'v': pxt.String},
            primary_key='k',
            if_exists='replace',
        )
        t.insert([{'k': 1, 'a': 100, 'v': 'one'}, {'k': 2, 'a': 200, 'v': 'two'}])

        # --json: found
        out = pcli('get', 'pcli_get/t', '2', '--json').json
        assert out['pk_columns'] == ['k']
        assert out['row']['v'] == 'two'

        # --json: missing row -> row is None
        out = pcli('get', 'pcli_get/t', '99', '--json').json
        assert out['row'] is None

        # text: found shows the row, missing prints 'no row found'
        assert 'two' in pcli('get', 'pcli_get/t', '2').stdout
        assert 'no row found' in pcli('get', 'pcli_get/t', '99').stdout

        # --cols subset: only the named columns returned
        out = pcli('get', 'pcli_get/t', '1', '--cols', 'a', '--json').json
        assert out['row'] == {'a': 100}

    def test_pk_coercion(self, pcli: PcliRunner) -> None:
        """A numeric-looking PK token is coerced to int or float; everything else stays a
        string. There is no quoting escape for a string PK that looks like a number."""
        pxt.create_dir('pcli_get_coerce', if_exists='ignore')

        # float PK: '1.5' coerces to float; whitespace around the numeric token is stripped
        # by float() (documented Python behavior) and the lookup still finds the row.
        t = pxt.create_table(
            'pcli_get_coerce.f', {'k': pxt.Required[pxt.Float], 'v': pxt.String}, primary_key='k', if_exists='replace'
        )
        t.insert([{'k': 1.5, 'v': 'one-and-a-half'}])
        assert pcli('get', 'pcli_get_coerce/f', '1.5', '--json').json['row']['v'] == 'one-and-a-half'
        assert pcli('get', 'pcli_get_coerce/f', '  1.5  ', '--json').json['row']['v'] == 'one-and-a-half'

        # string PK: a token that doesn't parse as a number stays a string.
        t = pxt.create_table(
            'pcli_get_coerce.s', {'k': pxt.Required[pxt.String], 'v': pxt.Int}, primary_key='k', if_exists='replace'
        )
        t.insert([{'k': 'alpha', 'v': 1}])
        assert pcli('get', 'pcli_get_coerce/s', 'alpha', '--json').json['row']['v'] == 1

    def test_errors(self, pcli: PcliRunner) -> None:
        """No-PK rejection, PK count mismatch, unknown col, empty/whitespace PK, empty --cols token."""
        pxt.create_dir('pcli_get_err', if_exists='ignore')
        pxt.create_table('pcli_get_err.no_pk', {'a': pxt.Int}, if_exists='replace')
        pxt.create_table('pcli_get_err.t', {'k': pxt.Required[pxt.Int]}, primary_key='k', if_exists='replace')
        pxt.create_table(
            'pcli_get_err.tc',
            {'a': pxt.Required[pxt.Int], 'b': pxt.Required[pxt.String]},
            primary_key=['a', 'b'],
            if_exists='replace',
        )

        # no primary key declared
        r = pcli('get', 'pcli_get_err/no_pk', '1', check=False)
        assert r.returncode != 0
        assert 'primary key' in r.stderr.lower()

        # wrong number of PK values for a single-col PK
        r = pcli('get', 'pcli_get_err/t', '1', '2', check=False)
        assert r.returncode != 0
        assert 'expected 1 PK' in r.stderr

        # unknown column in --cols
        r = pcli('get', 'pcli_get_err/t', '1', '--cols', 'missing', check=False)
        assert r.returncode != 0
        assert 'unknown columns' in r.stderr

        # empty/whitespace PK token rejected client-side
        r = pcli('get', 'pcli_get_err/t', '', check=False)
        assert r.returncode != 0
        assert 'empty or whitespace' in r.stderr
        r = pcli('get', 'pcli_get_err/t', '   ', check=False)
        assert r.returncode != 0
        assert 'empty or whitespace' in r.stderr

        # composite PK: even one slot empty is enough to reject
        r = pcli('get', 'pcli_get_err/tc', '1', '', check=False)
        assert r.returncode != 0
        assert 'empty or whitespace' in r.stderr

        # empty --cols token
        r = pcli('get', 'pcli_get_err/t', '1', '--cols', 'k,', check=False)
        assert r.returncode != 0
        assert 'must not be empty' in r.stderr


class TestCount:
    def test_basics(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_count', if_exists='ignore')
        t = pxt.create_table('pcli_count.t', {'a': pxt.Int}, if_exists='replace')
        t.insert([{'a': i} for i in range(7)])

        assert pcli('count', 'pcli_count/t', '--json').json['count'] == 7
        # plain output is just the integer
        assert pcli('count', 'pcli_count/t').stdout.strip() == '7'

    def test_errors(self, pcli: PcliRunner) -> None:
        # count on a directory is not allowed
        pxt.create_dir('pcli_count_dir', if_exists='ignore')
        r = pcli('count', 'pcli_count_dir', check=False)
        assert r.returncode != 0


class TestStatus:
    def test_basics(self, pcli: PcliRunner) -> None:
        # --json: paths are reported raw (no redaction)
        out = pcli('status', '--json').json
        assert out['pxt_version'] != ''
        assert out['pid'] > 0
        home = os.environ['PIXELTABLE_HOME']
        # PIXELTABLE_HOME-rooted paths come through verbatim - users want to see their actual layout
        assert out['home'] == home
        path_keys = ('home', 'media_dir', 'file_cache_dir')
        assert all(out.get(k) is None or '$PIXELTABLE_HOME' not in out[k] for k in path_keys)
        assert all(out.get(k) is None or '$HOME' not in out[k] for k in path_keys)

        # db_url password is still redacted: that's value protection, not path obfuscation
        if out.get('db_url') is not None:
            assert 'password' not in out['db_url'].lower() or '***' in out['db_url']

        # text: header lines for each field. Without --sizes, no size parenthetical is
        # emitted - the scan is skipped so there's nothing to report.
        text = pcli('status').stdout
        assert all(h in text for h in ('pxt_version', 'daemon_pid', 'home'))
        assert '(-)' not in text
        media_line = next(ln for ln in text.splitlines() if ln.startswith('media_dir'))
        assert '(' not in media_line

        # --sizes: _fmt_size renders a B/KB/MB suffix once sizes are populated
        sized = pcli('status', '--sizes').stdout
        assert any(unit in sized for unit in ('B)', 'KB)', 'MB)', 'GB)'))


class TestConfig:
    def test_basics(self, pcli: PcliRunner) -> None:
        """pcli config reports every documented configuration setting with its resolved value and
        source (env / file / unset). Credentials show '<redacted>' when set; the source
        field reveals presence even when the value is masked."""
        out = pcli('config', '--json').json
        assert out['config_file'].endswith('config.toml')
        entries = out['entries']

        # every entry has the expected shape
        expected_keys = {'section', 'key', 'value', 'source', 'description', 'expected_type'}
        assert all(set(e.keys()) == expected_keys for e in entries)
        # source is one of the three documented values
        assert all(e['source'] in ('env', 'file', 'unset') for e in entries)

        # the registry covers the top-level 'pixeltable' section plus per-provider sections
        sections = {e['section'] for e in entries}
        assert 'pixeltable' in sections
        assert 'openai' in sections

        # openai.api_key is a known credential entry
        openai_key = next(e for e in entries if e['section'] == 'openai' and e['key'] == 'api_key')
        # if it's set in the test environment's config.toml, value is '<redacted>'; if not, None
        if openai_key['source'] != 'unset':
            assert openai_key['value'] == '<redacted>'
        else:
            assert openai_key['value'] is None

        # text mode: set entries in aligned table, unset entries collapsed into a "not set" line.
        # Defining keys section by section avoids the noise of one-per-line for things at default.
        text = pcli('config').stdout
        assert 'config_file' in text
        # openai.api_key appears either in the aligned table (if set) or in the not-set list
        assert 'openai.api_key' in text
        # the unset bucket is summarized on a single line; the legacy '(unset)' marker is gone
        assert '(unset)' not in text

        # -v: descriptions inline under each entry, and unset entries get full table rows
        # (with '-' as the value) so the description has somewhere to land.
        verbose = pcli('config', '-v').stdout
        assert 'config_file' in verbose
        assert 'not set:' not in verbose
        assert '[unset]' in verbose
        # at least one description (every entry in the registry has one)
        openai_key = next(
            e for e in pcli('config', '--json').json['entries'] if e['section'] == 'openai' and e['key'] == 'api_key'
        )
        assert openai_key['description'] in verbose

    def test_filters(self, pcli: PcliRunner) -> None:
        """--section filters to one section; --source filters by where the value comes from."""
        out = pcli('config', '--section', 'openai', '--json').json
        assert all(e['section'] == 'openai' for e in out['entries'])

        out = pcli('config', '--source', 'unset', '--json').json
        assert all(e['source'] == 'unset' for e in out['entries'])

        # --source unset text mode collapses everything into the "not set" line - no table header
        text = pcli('config', '--source', 'unset').stdout
        assert 'not set:' in text
        # no per-entry table rows when everything is unset
        assert '[unset]' not in text

        # invalid source value rejected by argparse
        r = pcli('config', '--source', 'nope', check=False)
        assert r.returncode == 2


class TestErrors:
    """The `pcli errors` command itself."""

    def test_basics(self, pcli: PcliRunner) -> None:
        """Populated + empty cases, JSON and text."""
        pxt.create_dir('pcli_errs', if_exists='ignore')

        # No computed columns -> no errors -> empty JSON list and empty text output
        pxt.create_table('pcli_errs.ok', {'k': pxt.Required[pxt.Int]}, primary_key='k', if_exists='replace')
        assert pcli('errors', 'pcli_errs/ok', '--json').json == []
        r = pcli('errors', 'pcli_errs/ok')
        assert r.returncode == 0
        assert r.stdout.strip() == ''

        # Computed column that raises for k=0: row 0 shows up in the errors listing
        t = pxt.create_table('pcli_errs.bad', {'k': pxt.Required[pxt.Int]}, primary_key='k', if_exists='replace')
        t.add_computed_column(b=_boom_on_zero(t.k), on_error='ignore')
        t.add_computed_column(c=_boom_on_zero(t.k), on_error='ignore')
        t.insert([{'k': 0}, {'k': 1}], on_error='ignore')
        text = pcli('errors', 'pcli_errs/bad').stdout
        assert 'k: 0' in text
        assert 'b' in text

        # --col filters to one stored computed column: an existing column with errors comes back,
        # and asking for the other computed column also returns its errors.
        out_b = pcli('errors', 'pcli_errs/bad', '--col', 'b', '--json').json
        assert len(out_b) == 1
        assert out_b[0]['column'] == 'b'
        out_c = pcli('errors', 'pcli_errs/bad', '--col', 'c', '--json').json
        assert len(out_c) == 1
        assert out_c[0]['column'] == 'c'

    def test_errors(self, pcli: PcliRunner) -> None:
        """No-PK rejection, unknown --col, --col on a non-stored-computed column."""
        pxt.create_dir('pcli_errs_err', if_exists='ignore')
        pxt.create_table('pcli_errs_err.no_pk', {'a': pxt.Int}, if_exists='replace')
        pxt.create_table(
            'pcli_errs_err.t', {'k': pxt.Required[pxt.Int], 'plain': pxt.String}, primary_key='k', if_exists='replace'
        )

        # no PK -> 400
        r = pcli('errors', 'pcli_errs_err/no_pk', check=False)
        assert r.returncode != 0
        assert 'primary key' in r.stderr.lower()

        # unknown --col
        r = pcli('errors', 'pcli_errs_err/t', '--col', 'nope', check=False)
        assert r.returncode != 0
        assert 'unknown column' in r.stderr

        # --col on a column that isn't a stored computed column
        r = pcli('errors', 'pcli_errs_err/t', '--col', 'plain', check=False)
        assert r.returncode != 0
        assert 'stored computed column' in r.stderr


class TestDrop:
    """`pcli drop` (tables) and `pcli rm` (directories). They share the universal mutation
    surface (--force, --dry-run, --json) and the no-TTY refusal."""

    def test_basics(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_drop', if_exists='ignore')
        pxt.create_dir('pcli_drop.nest', if_exists='ignore')
        pxt.create_table('pcli_drop.nest.victim', {'a': pxt.Int}, if_exists='replace')
        pxt.create_table('pcli_drop.dry', {'a': pxt.Int}, if_exists='replace')
        pxt.create_table('pcli_drop.txt', {'a': pxt.Int}, if_exists='replace')

        # drop a table: --json reports dropped=True; table is gone
        out = pcli('drop', 'pcli_drop/nest/victim', '-f', '--json').json
        assert out['dropped'] is True
        assert pxt.get_table('pcli_drop/nest/victim', if_not_exists='ignore') is None

        # rm the (now empty) directory
        assert pcli('rm', 'pcli_drop/nest', '-f', '--json').json['dropped'] is True

        # rm refuses a non-empty dir without -r; -r succeeds
        pxt.create_dir('pcli_drop.nest2', if_exists='ignore')
        pxt.create_table('pcli_drop.nest2.t', {'a': pxt.Int}, if_exists='replace')
        assert pcli('rm', 'pcli_drop/nest2', '-f', check=False).returncode != 0
        assert pcli('rm', 'pcli_drop/nest2', '-r', '-f', '--json').json['dropped'] is True

        # dry-run text + --json: no side effect, message includes 'would drop'
        r = pcli('drop', 'pcli_drop/dry', '-n')
        assert 'would drop' in r.stdout
        assert pxt.get_table('pcli_drop/dry', if_not_exists='ignore') is not None

        # plain text confirmation
        assert 'dropped' in pcli('drop', 'pcli_drop/txt', '-f').stdout

        # rm text output
        pxt.create_dir('pcli_drop_rmtxt', if_exists='ignore')
        assert 'removed' in pcli('rm', 'pcli_drop_rmtxt', '-f').stdout

        # rm dry-run text (both with and without -r)
        pxt.create_dir('pcli_drop_dr', if_exists='ignore')
        r = pcli('rm', 'pcli_drop_dr', '-n')
        assert 'would remove' in r.stdout
        r = pcli('rm', 'pcli_drop_dr', '-n', '-r')
        assert 'would remove' in r.stdout
        assert 'recursive' in r.stdout

    def test_cascade(self, pcli: PcliRunner) -> None:
        """--cascade maps to force=True server-side: drops a table that has dependent views.
        Without --cascade, dropping a table with a view fails."""
        pxt.create_dir('pcli_drop_csc', if_exists='ignore')
        t = pxt.create_table('pcli_drop_csc.base', {'a': pxt.Int}, if_exists='replace')
        pxt.create_view('pcli_drop_csc.dep_view', t, if_exists='replace')

        # without --cascade: drop fails because of the dependent view
        r = pcli('drop', 'pcli_drop_csc/base', '-f', check=False)
        assert r.returncode != 0
        assert pxt.get_table('pcli_drop_csc/base', if_not_exists='ignore') is not None

        # with --cascade: both base table and dependent view are dropped
        assert pcli('drop', 'pcli_drop_csc/base', '--cascade', '-f', '--json').json['dropped'] is True
        assert pxt.get_table('pcli_drop_csc/base', if_not_exists='ignore') is None
        assert pxt.get_table('pcli_drop_csc/dep_view', if_not_exists='ignore') is None

    def test_errors(self, pcli: PcliRunner) -> None:
        # no -f, no TTY: refuse to proceed
        pxt.create_dir('pcli_drop_err', if_exists='ignore')
        pxt.create_table('pcli_drop_err.protected', {'a': pxt.Int}, if_exists='replace')
        r = pcli('drop', 'pcli_drop_err/protected', check=False)
        assert r.returncode != 0
        assert '--force' in r.stderr
        # still exists
        assert pxt.get_table('pcli_drop_err/protected', if_not_exists='ignore') is not None

        # `pcli drop` on a directory is refused (server-side: drop_table on a dir errors)
        pxt.create_dir('pcli_drop_err_dir', if_exists='ignore')
        r = pcli('drop', 'pcli_drop_err_dir', '-f', check=False)
        assert r.returncode != 0


class TestRename:
    def test_basics(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_rn', if_exists='ignore')
        pxt.create_table('pcli_rn.old_name', {'a': pxt.Int}, if_exists='replace')
        pxt.create_table('pcli_rn.dr', {'a': pxt.Int}, if_exists='replace')
        pxt.create_table('pcli_rn.txt', {'a': pxt.Int}, if_exists='replace')

        # rename + --json
        out = pcli('rename', 'pcli_rn/old_name', 'new_name', '--json').json
        assert out['new_path'] == 'pcli_rn/new_name'
        assert pxt.get_table('pcli_rn/new_name', if_not_exists='ignore') is not None
        assert pxt.get_table('pcli_rn/old_name', if_not_exists='ignore') is None

        # dry-run: 'would rename', no side effect
        r = pcli('rename', 'pcli_rn/dr', 'dr2', '-n')
        assert 'would rename' in r.stdout
        assert pxt.get_table('pcli_rn/dr', if_not_exists='ignore') is not None

        # text confirmation
        assert 'renamed' in pcli('rename', 'pcli_rn/txt', 'txt2').stdout

    def test_errors(self, pcli: PcliRunner) -> None:
        """`new_name` must be a leaf, no '/' or '.'."""
        pxt.create_dir('pcli_rn_err', if_exists='ignore')
        pxt.create_table('pcli_rn_err.t', {'a': pxt.Int}, if_exists='replace')
        r = pcli('rename', 'pcli_rn_err/t', 'a/b', check=False)
        assert r.returncode != 0
        assert 'leaf name' in r.stderr


class TestMv:
    def test_basics(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_mv', if_exists='ignore')
        pxt.create_dir('pcli_mv.src', if_exists='ignore')
        pxt.create_dir('pcli_mv.dst', if_exists='ignore')
        pxt.create_table('pcli_mv.src.movee', {'a': pxt.Int}, if_exists='replace')
        pxt.create_table('pcli_mv.toroot', {'a': pxt.Int}, if_exists='replace')
        pxt.create_table('pcli_mv.dr', {'a': pxt.Int}, if_exists='replace')

        # mv into another dir: --json reports the new path
        out = pcli('mv', 'pcli_mv/src/movee', 'pcli_mv/dst', '--json').json
        assert out['new_path'] == 'pcli_mv/dst/movee'
        assert pxt.get_table('pcli_mv/dst/movee', if_not_exists='ignore') is not None

        # text confirmation
        pxt.create_table('pcli_mv.src.tx', {'a': pxt.Int}, if_exists='replace')
        assert 'moved' in pcli('mv', 'pcli_mv/src/tx', 'pcli_mv/dst').stdout

        # '/' as destination moves to root
        pcli('mv', 'pcli_mv/toroot', '/')
        assert pxt.get_table('toroot', if_not_exists='ignore') is not None

        # dry-run: 'would move', no side effect
        r = pcli('mv', 'pcli_mv/dr', 'pcli_mv', '-n')
        assert 'would move' in r.stdout

    def test_errors(self, pcli: PcliRunner) -> None:
        # nonexistent source path
        r = pcli('mv', 'pcli_mv_err/missing', 'pcli_mv_err', check=False)
        assert r.returncode != 0


class TestRevert:
    def test_basics(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_rv', if_exists='ignore')
        t = pxt.create_table('pcli_rv.t', {'a': pxt.Int}, if_exists='replace')
        t.insert([{'a': 1}])
        v_before = t.get_metadata()['version']
        pxt.create_table('pcli_rv.txt', {'a': pxt.Int}, if_exists='replace').insert([{'a': 1}])
        pxt.create_table('pcli_rv.dr', {'a': pxt.Int}, if_exists='replace')

        # revert + --json
        out = pcli('revert', 'pcli_rv/t', '-f', '--json').json
        assert out['from_version'] == v_before
        assert out['to_version'] == v_before - 1

        # text: 'reverted ... -> ...'
        text = pcli('revert', 'pcli_rv/txt', '-f').stdout
        assert 'reverted' in text
        assert '->' in text

        # dry-run: 'would revert', no side effect
        assert 'would revert' in pcli('revert', 'pcli_rv/dr', '-n').stdout

    def test_errors(self, pcli: PcliRunner, pcli_daemon: int) -> None:
        """Client preflight: --steps must be >= 1. Server: cannot revert past version 0.
        Plus a direct HTTP test confirming the server's own steps<1 check fires for any
        future programmatic caller that bypasses the CLI."""
        pxt.create_dir('pcli_rv_err', if_exists='ignore')
        pxt.create_table('pcli_rv_err.t', {'a': pxt.Int}, if_exists='replace')

        # client preflight
        r = pcli('revert', 'pcli_rv_err/t', '--steps', '0', '-f', check=False)
        assert r.returncode != 0
        assert '--steps must be >= 1' in r.stderr

        # server preflight: cannot revert beyond the current version
        r = pcli('revert', 'pcli_rv_err/t', '--steps', '999', '-f', check=False)
        assert r.returncode != 0
        assert 'cannot revert' in r.stderr

        # direct HTTP: server's own steps<1 check fires when the client preflight is bypassed
        req = urllib.request.Request(
            f'http://127.0.0.1:{pcli_daemon}/pcli/v0/revert',
            data=json.dumps({'path': 'whatever', 'steps': 0}).encode(),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(req)
        assert ei.value.code == 400
        assert 'steps must be >= 1' in json.loads(ei.value.read())['detail']


class TestPathValidator:
    """Client-side path validator (pcli.models._slash_only). Catches every well-known bad
    shape before the request reaches the server so the user gets a clear error message
    instead of a generic 'Invalid path' from pxt."""

    def test_rejects_bad_shapes(self, pcli: PcliRunner) -> None:
        # '.' is reserved (pxt's legacy separator)
        r = pcli('describe', 'a.b', check=False)
        assert r.returncode != 0
        assert "'/'" in r.stderr or 'separator' in r.stderr
        # leading '/' would create an empty leading component
        r = pcli('describe', '/x', check=False)
        assert r.returncode != 0
        assert 'relative' in r.stderr
        # trailing '/' would create an empty trailing component
        r = pcli('describe', 'x/', check=False)
        assert r.returncode != 0
        assert "must not end with '/'" in r.stderr
        # '//' produces an empty internal component
        r = pcli('describe', 'a//b', check=False)
        assert r.returncode != 0
        assert 'empty components' in r.stderr


class TestColsValidator:
    """Client-side --cols validator (parser.parse_cols). Rejects every shape that would
    yield an empty token. Shared between `rows` and `get`."""

    def test_rejects_empty_tokens(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_colsv', if_exists='ignore')
        pxt.create_table('pcli_colsv.r', {'a': pxt.Int, 'b': pxt.Int}, if_exists='replace')
        pxt.create_table('pcli_colsv.g', {'k': pxt.Required[pxt.Int]}, primary_key='k', if_exists='replace')
        results = [pcli('rows', 'pcli_colsv/r', '--cols', bad, check=False) for bad in ('a,', ',a', 'a,,b')]
        assert all(r.returncode != 0 for r in results)
        assert all('must not be empty' in r.stderr for r in results)
        r = pcli('get', 'pcli_colsv/g', '1', '--cols', 'k,', check=False)
        assert r.returncode != 0
        assert 'must not be empty' in r.stderr


class TestCli:
    """Top-level CLI surface (help, unknown commands, argparse errors)."""

    def test_help(self, pcli: PcliRunner) -> None:
        # --help: lists every command, exits 0
        r = pcli('--help')
        assert r.returncode == 0
        assert all(name in r.stdout for name in ('health', 'ls', 'shell'))

        # no args: prints usage and exits 2
        r = pcli(check=False)
        assert r.returncode == 2
        assert 'usage' in r.stdout.lower() or 'usage' in r.stderr.lower()

    def test_unknown_command(self, pcli: PcliRunner) -> None:
        r = pcli('not_a_command', check=False)
        assert r.returncode == 2
        assert 'unknown command' in r.stderr

    def test_subcommand_arg_errors(self, pcli: PcliRunner) -> None:
        # `pcli rows` is missing the required path positional; argparse prints usage + epilog
        r = pcli('rows', check=False)
        assert r.returncode == 2
        assert 'usage' in r.stderr.lower()
        assert 'Examples' in r.stderr  # the per-command epilog block is appended on error
