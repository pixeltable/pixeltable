"""Smoke tests: drive state via the pxt API, validate via cli.

Each class focuses on one cli command. Methods bundle related scenarios so a single
table can serve multiple assertions (JSON + text + flag variants) without re-creating it.
"""

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable

import numpy as np
import pytest

import pixeltable as pxt
from tests.utils import CatalogMode, get_image_files

from .conftest import PxtRunner


@pxt.udf
def _fail_on_zero(x: int) -> int:
    """Module-level UDF: raises for k=0 so we can populate a stored errortype column."""
    if x == 0:
        raise ValueError('fail')
    return x


@pxt.udf
def _trivial_embed(s: str) -> pxt.Array[(8,), np.float32]:
    """Module-level embedder for embedding-index tests: deterministic, no model download."""
    return np.zeros(8, dtype=np.float32)


@pytest.mark.local('reports daemon liveness/version; not catalog-specific')
class TestHealth:
    def test_basics(self, cli: PxtRunner, pxt_daemon: int) -> None:
        out = cli('health').json
        assert out['ok'] is True
        assert out['pid'] > 0
        assert out['service'] == 'pxt'
        assert out['pxt_version'] == pxt.__version__


class TestLs:
    def test_lists(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        """Bare ls (text + json) lists what's in the catalog and reflects mutations."""
        p = make_catalog_path
        pxt.create_dir(p('cli_ls'), if_exists='ignore')
        pxt.create_table(p('cli_ls.t'), {'x': pxt.Int}, if_exists='replace')

        # json reports the entry
        entries = cli('ls', p('cli_ls'), '--json').json['entries']
        assert p('cli_ls/t') in {e['path'] for e in entries}

        # plain text lists the headers + path + kind
        text = cli('ls', p('cli_ls')).stdout
        assert 'path' in text
        assert 'kind' in text
        assert p('cli_ls/t') in text

        # an empty dir is still a valid listing target
        pxt.create_dir(p('cli_ls_empty'), if_exists='ignore')
        assert cli('ls', p('cli_ls_empty')).returncode == 0

        # after a drop, the entry is gone from the listing
        pxt.drop_table(p('cli_ls.t'))
        entries = cli('ls', p('cli_ls'), '--json').json['entries']
        assert p('cli_ls/t') not in {e['path'] for e in entries}

    def test_long_and_metadata(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        """-l / -l --json populate num_cols and flags via get_metadata(). Bare --json is the
        cheap path: it skips the per-entry metadata fetch and returns num_cols=None."""
        p = make_catalog_path
        pxt.create_dir(p('cli_ls_long'), if_exists='ignore')
        t = pxt.create_table(p('cli_ls_long.t'), {'a': pxt.Int}, if_exists='replace')
        t.add_computed_column(b=t.a * 2)

        # -l text: headers + 'c' flag for computed column
        text = cli('ls', p('cli_ls_long'), '-l').stdout
        assert all(h in text for h in ('cols', 'version', 'flags'))
        assert 'c' in text

        # -l --json: details populated
        entries = cli('ls', p('cli_ls_long'), '-l', '--json').json['entries']
        row = next(e for e in entries if e['path'] == p('cli_ls_long/t'))
        assert row['num_cols'] is not None
        assert row['num_cols'] >= 2
        assert 'c' in row['flags']

        # bare --json: cheap-by-default, no metadata fetch
        entries = cli('ls', p('cli_ls_long'), '--json').json['entries']
        row = next(e for e in entries if e['path'] == p('cli_ls_long/t'))
        assert row['num_cols'] is None
        assert row['flags'] == ''

    def test_tree_and_counts(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        """--tree formats the nested catalog with ASCII prefixes. --counts populates
        num_rows in both text and JSON; a dirs-only target skips the count pool entirely."""
        p = make_catalog_path
        pxt.create_dir(p('cli_ls_tree'), if_exists='ignore')
        pxt.create_dir(p('cli_ls_tree.sub'), if_exists='ignore')
        pxt.create_table(p('cli_ls_tree.sub.leaf'), {'a': pxt.Int}, if_exists='replace')
        t = pxt.create_table(p('cli_ls_tree.t'), {'a': pxt.Int}, if_exists='replace')
        t.insert([{'a': i} for i in range(4)])

        # --tree: nested entries and a box-drawing prefix
        tree_text = cli('ls', p('cli_ls_tree'), '--tree').stdout
        assert all(marker in tree_text for marker in ('sub', 'leaf'))
        assert any(prefix in tree_text for prefix in ('|--', '\\--'))

        # --counts json: num_rows populated for the table
        entries = cli('ls', p('cli_ls_tree'), '--counts', '--json').json['entries']
        row = next(e for e in entries if e['path'] == p('cli_ls_tree/t'))
        assert row['num_rows'] == 4

        # --counts text: includes the 'rows' column header
        counts_text = cli('ls', p('cli_ls_tree'), '--counts').stdout
        assert 'rows' in counts_text
        assert '4' in counts_text

        # --counts on a dirs-only target: directories report no row count
        pxt.create_dir(p('cli_ls_dirs'), if_exists='ignore')
        pxt.create_dir(p('cli_ls_dirs.sub1'), if_exists='ignore')
        pxt.create_dir(p('cli_ls_dirs.sub2'), if_exists='ignore')
        entries = cli('ls', p('cli_ls_dirs'), '--counts', '--json').json['entries']
        assert all(e['kind'] == 'dir' for e in entries)
        assert all(e.get('num_rows') is None for e in entries)

    def test_errors(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        """ls distinguishes 'path does not exist' (404) from 'path exists but is not a directory'
        (422). The latter case names the offending component and its kind so the user can fix
        a typo like `pxt ls my_table` vs `pxt describe my_table`."""
        p = make_catalog_path
        pxt.create_dir(p('cli_ls_err'), if_exists='ignore')
        pxt.create_table(p('cli_ls_err.t'), {'a': pxt.Int}, if_exists='replace')

        # name absent at the root -> 404 path-not-found
        r = cli('ls', p('does_not_exist'), check=False)
        assert r.returncode != 0
        assert 'does not exist' in r.stderr.lower() or 'not found' in r.stderr.lower()

        # name exists at the leaf but is a table, not a directory -> 422 with the kind named. The message
        # names the in-catalog path (the daemon reports paths relative to the catalog it navigated).
        r = cli('ls', p('cli_ls_err/t'), check=False)
        assert r.returncode != 0
        assert "'cli_ls_err/t' is a table, not a directory" in r.stderr

        # a table appearing mid-path: the error names the table-rooted prefix, not the full path
        r = cli('ls', p('cli_ls_err/t/sub'), check=False)
        assert r.returncode != 0
        assert "'cli_ls_err/t' is a table, not a directory" in r.stderr


class TestDescribe:
    def test_basics(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        pxt.create_dir(p('cli_desc'), if_exists='ignore')
        pxt.create_table(p('cli_desc.t'), {'a': pxt.Int, 'b': pxt.String}, if_exists='replace')

        # --json returns the full get_metadata() dict
        meta = cli('describe', p('cli_desc/t'), '--json').json
        assert 'a' in meta['columns']

        # text is repr(table) which lists the columns
        text = cli('describe', p('cli_desc/t')).stdout
        assert 'a' in text
        assert 'b' in text

    def test_errors(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        r = cli('describe', p('does_not_exist'), check=False)
        assert r.returncode != 0


class TestColumns:
    def test_lists(self, cli: PxtRunner, make_catalog_path: Callable[[str], str], catalog_mode: CatalogMode) -> None:
        """columns lists every column; `computed` is an alias for `columns --computed`.
        The no-path form walks every table in the catalog."""
        p = make_catalog_path
        pxt.create_dir(p('cli_cols'), if_exists='ignore')
        t = pxt.create_table(p('cli_cols.t'), {'a': pxt.Int, 'b': pxt.String}, if_exists='replace')
        t.add_computed_column(doubled=t.a * 2)

        # --json: all three columns
        entries = cli('columns', p('cli_cols/t'), '--json').json
        assert {e['column'] for e in entries} == {'a', 'b', 'doubled'}

        # text format: stored vs computed flag
        text = cli('columns', p('cli_cols/t')).stdout
        assert 'a' in text
        assert 'stored' in text

        # the computed subcommand is the --computed shorthand
        entries = cli('computed', p('cli_cols/t'), '--json').json
        assert {e['column'] for e in entries} == {'doubled'}

        # text mode also works
        assert 'doubled' in cli('computed', p('cli_cols/t')).stdout

        # no-path: walks every table in the daemon's in-process catalog, exercising the try/except continue
        # for unloadable metadata. This form has no proxy analog (it doesn't target a specific catalog).
        if catalog_mode == 'local':
            entries = cli('columns', '--json').json
            assert ('cli_cols/t', 'a') in {(e['table'], e['column']) for e in entries}


class TestIdxs:
    def test_lists(self, cli: PxtRunner, make_catalog_path: Callable[[str], str], catalog_mode: CatalogMode) -> None:
        """idxs runs against one table or globally; the no-path form walks every table."""
        p = make_catalog_path
        pxt.create_dir(p('cli_idx'), if_exists='ignore')
        pxt.create_table(p('cli_idx.t'), {'a': pxt.Int}, if_exists='replace')

        # --json: no embedding idx exists on a plain table
        entries = cli('idxs', p('cli_idx/t'), '--json').json
        assert all(e['index_type'] != 'embedding' for e in entries)

        # text: command runs cleanly regardless of catalog content
        assert cli('idxs', p('cli_idx/t')).returncode == 0

        # no-path: walks every table in the daemon's in-process catalog (no proxy analog); smoke check
        # only since auto-indexes vary
        if catalog_mode == 'local':
            assert cli('idxs', '--json').returncode == 0

    def test_embedding_filter(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        """--embedding filters to embedding indexes only; a table with both a btree-style
        index and an embedding index reports one entry each, then only the embedding under
        --embedding."""
        p = make_catalog_path
        pxt.create_dir(p('cli_idx_emb'), if_exists='ignore')
        t = pxt.create_table(p('cli_idx_emb.t'), {'s': pxt.String}, if_exists='replace')
        t.add_embedding_index('s', idx_name='emb_idx', string_embed=_trivial_embed)

        all_entries = cli('idxs', p('cli_idx_emb/t'), '--json').json
        emb_only = cli('idxs', p('cli_idx_emb/t'), '--embedding', '--json').json
        assert any(e['name'] == 'emb_idx' and e['index_type'] == 'embedding' for e in all_entries)
        assert all(e['index_type'] == 'embedding' for e in emb_only)
        assert any(e['name'] == 'emb_idx' for e in emb_only)


class TestHistory:
    def test_basics(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        pxt.create_dir(p('cli_hist'), if_exists='ignore')
        t = pxt.create_table(p('cli_hist.t'), {'a': pxt.Int}, if_exists='replace')
        for i in range(4):
            t.insert([{'a': i}])

        # --json: every committed version
        versions = cli('history', p('cli_hist/t'), '--json').json
        assert len(versions) >= 5  # create + 4 inserts

        # -n caps the result count to the last N versions
        capped = cli('history', p('cli_hist/t'), '-n', '2', '--json').json
        assert len(capped) == 2

        # text: tab-separated header
        text = cli('history', p('cli_hist/t')).stdout
        assert 'version' in text
        assert 'change_type' in text

    @pytest.mark.local('direct-HTTP test of the daemon route validator; fires before catalog resolution')
    def test_server_rejects_malformed_n(self, pxt_daemon: int) -> None:
        # A non-integer or out-of-range n must produce a structured 4xx, not bubble up as a
        # generic 500. The CLI client validates -n before sending, so this exercises the
        # programmatic-caller path.
        for bad, expected in (('abc', 'must be an integer'), ('0', 'must be >= 1'), ('-3', 'must be >= 1')):
            req = urllib.request.Request(f'http://127.0.0.1:{pxt_daemon}/api/tables/cli_hist/t/history?n={bad}')
            with pytest.raises(urllib.error.HTTPError) as ei:
                urllib.request.urlopen(req)
            assert ei.value.code == 422
            assert expected in json.loads(ei.value.read())['detail']


class TestRows:
    def test_basics(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        """rows: text + --json default; -n limit; --cols subset; null cells render as empty."""
        p = make_catalog_path
        pxt.create_dir(p('cli_rows'), if_exists='ignore')
        t = pxt.create_table(p('cli_rows.t'), {'n': pxt.Int, 's': pxt.String}, if_exists='replace')
        t.insert([{'n': i, 's': f'row{i}'} for i in range(5)])

        # --json + -n
        out = cli('rows', p('cli_rows/t'), '-n', '3', '--json').json
        assert len(out) == 3

        # --cols subset
        out = cli('rows', p('cli_rows/t'), '-n', '10', '--cols', 'n', '--json').json
        assert sorted(r['n'] for r in out) == [0, 1, 2, 3, 4]

        # text contains the inserted values
        text = cli('rows', p('cli_rows/t')).stdout
        assert 'row0' in text
        assert 'row4' in text

        # null cell renders as empty, not the literal 'None'
        t2 = pxt.create_table(p('cli_rows.nulls'), {'a': pxt.Int, 's': pxt.String}, if_exists='replace')
        t2.insert([{'a': 1, 's': None}])
        assert 'None' not in cli('rows', p('cli_rows/nulls')).stdout

    def test_image_column(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        """Image cells must render as `<Image WxH MODE>` in both text and JSON modes -
        not as raw bytes, base64, or a PIL repr."""
        p = make_catalog_path
        img_paths = get_image_files()[:2]
        pxt.create_dir(p('cli_img'), if_exists='ignore')
        t = pxt.create_table(
            p('cli_img.t'),
            {'k': pxt.Required[pxt.Int], 'img': pxt.Image, 'tag': pxt.String},
            primary_key='k',
            if_exists='replace',
        )
        t.insert([{'k': i, 'img': img_path, 'tag': f't{i}'} for i, img_path in enumerate(img_paths)])

        rows_json = cli('rows', p('cli_img/t'), '-n', '5', '--json').json
        assert len(rows_json) == 2
        assert all(r['img'].startswith('<Image ') for r in rows_json)
        assert all(r['img'].endswith('>') for r in rows_json)

        rows_text = cli('rows', p('cli_img/t')).stdout
        assert '<Image ' in rows_text
        assert 'PIL.' not in rows_text  # no PIL repr leaked
        assert '\\x' not in rows_text  # no raw bytes leaked

        # the get subcommand uses the same renderer
        get_json = cli('get', p('cli_img/t'), '1', '--json').json
        assert get_json['row']['img'].startswith('<Image ')
        get_text = cli('get', p('cli_img/t'), '0').stdout
        assert 'img\t<Image ' in get_text

    def test_errors(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        pxt.create_dir(p('cli_rows_err'), if_exists='ignore')
        pxt.create_table(p('cli_rows_err.t'), {'a': pxt.Int}, if_exists='replace')

        # n must be >= 1
        r = cli('rows', p('cli_rows_err/t'), '-n', '0', check=False)
        assert r.returncode != 0
        assert "'n' must be >= 1" in r.stderr

        # n must be <= 1000
        r = cli('rows', p('cli_rows_err/t'), '-n', '10001', check=False)
        assert r.returncode != 0
        assert "'n' must be <= 1000" in r.stderr

        # unknown column
        r = cli('rows', p('cli_rows_err/t'), '--cols', 'nope', check=False)
        assert r.returncode != 0
        assert 'unknown columns' in r.stderr


class TestGet:
    def test_basics(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        """Single + composite PK; text + json + missing-row; --cols subset."""
        p = make_catalog_path
        pxt.create_dir(p('cli_get'), if_exists='ignore')
        t = pxt.create_table(
            p('cli_get.t'),
            {'k': pxt.Required[pxt.Int], 'a': pxt.Int, 'v': pxt.String},
            primary_key='k',
            if_exists='replace',
        )
        t.insert([{'k': 1, 'a': 100, 'v': 'one'}, {'k': 2, 'a': 200, 'v': 'two'}])

        # --json: found
        out = cli('get', p('cli_get/t'), '2', '--json').json
        assert out['pk_columns'] == ['k']
        assert out['row']['v'] == 'two'

        # --json: missing row -> row is None
        out = cli('get', p('cli_get/t'), '99', '--json').json
        assert out['row'] is None

        # text: found shows the row, missing prints 'no row found'
        assert 'two' in cli('get', p('cli_get/t'), '2').stdout
        assert 'no row found' in cli('get', p('cli_get/t'), '99').stdout

        # --cols subset: only the named columns returned
        out = cli('get', p('cli_get/t'), '1', '--cols', 'a', '--json').json
        assert out['row'] == {'a': 100}

    def test_pk_coercion(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        """A numeric-looking PK token is coerced to int or float; everything else stays a
        string. There is no quoting escape for a string PK that looks like a number."""
        p = make_catalog_path
        pxt.create_dir(p('cli_get_coerce'), if_exists='ignore')

        # float PK: '1.5' coerces to float; whitespace around the numeric token is stripped
        # by float() (documented Python behavior) and the lookup still finds the row.
        t = pxt.create_table(
            p('cli_get_coerce.f'), {'k': pxt.Required[pxt.Float], 'v': pxt.String}, primary_key='k', if_exists='replace'
        )
        t.insert([{'k': 1.5, 'v': 'one-and-a-half'}])
        assert cli('get', p('cli_get_coerce/f'), '1.5', '--json').json['row']['v'] == 'one-and-a-half'
        assert cli('get', p('cli_get_coerce/f'), '  1.5  ', '--json').json['row']['v'] == 'one-and-a-half'

        # string PK: a token that doesn't parse as a number stays a string.
        t = pxt.create_table(
            p('cli_get_coerce.s'), {'k': pxt.Required[pxt.String], 'v': pxt.Int}, primary_key='k', if_exists='replace'
        )
        t.insert([{'k': 'alpha', 'v': 1}])
        assert cli('get', p('cli_get_coerce/s'), 'alpha', '--json').json['row']['v'] == 1

    def test_errors(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        """No-PK rejection, PK count mismatch, unknown col, empty/whitespace PK, empty --cols token."""
        p = make_catalog_path
        pxt.create_dir(p('cli_get_err'), if_exists='ignore')
        pxt.create_table(p('cli_get_err.no_pk'), {'a': pxt.Int}, if_exists='replace')
        pxt.create_table(p('cli_get_err.t'), {'k': pxt.Required[pxt.Int]}, primary_key='k', if_exists='replace')
        pxt.create_table(
            p('cli_get_err.tc'),
            {'a': pxt.Required[pxt.Int], 'b': pxt.Required[pxt.String]},
            primary_key=['a', 'b'],
            if_exists='replace',
        )

        # no primary key declared
        r = cli('get', p('cli_get_err/no_pk'), '1', check=False)
        assert r.returncode != 0
        assert 'primary key' in r.stderr.lower()

        # wrong number of PK values for a single-col PK
        r = cli('get', p('cli_get_err/t'), '1', '2', check=False)
        assert r.returncode != 0
        assert 'expected 1 PK' in r.stderr

        # unknown column in --cols
        r = cli('get', p('cli_get_err/t'), '1', '--cols', 'missing', check=False)
        assert r.returncode != 0
        assert 'unknown columns' in r.stderr

        # empty/whitespace PK token rejected client-side
        r = cli('get', p('cli_get_err/t'), '', check=False)
        assert r.returncode != 0
        assert 'empty or whitespace' in r.stderr
        r = cli('get', p('cli_get_err/t'), '   ', check=False)
        assert r.returncode != 0
        assert 'empty or whitespace' in r.stderr

        # composite PK: even one slot empty is enough to reject
        r = cli('get', p('cli_get_err/tc'), '1', '', check=False)
        assert r.returncode != 0
        assert 'empty or whitespace' in r.stderr

        # empty --cols token
        r = cli('get', p('cli_get_err/t'), '1', '--cols', 'k,', check=False)
        assert r.returncode != 0
        assert 'must not be empty' in r.stderr


class TestCount:
    def test_basics(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        pxt.create_dir(p('cli_count'), if_exists='ignore')
        t = pxt.create_table(p('cli_count.t'), {'a': pxt.Int}, if_exists='replace')
        t.insert([{'a': i} for i in range(7)])

        assert cli('count', p('cli_count/t'), '--json').json['count'] == 7
        # plain output is just the integer
        assert cli('count', p('cli_count/t')).stdout.strip() == '7'

    def test_errors(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        # count on a directory is not allowed
        pxt.create_dir(p('cli_count_dir'), if_exists='ignore')
        r = cli('count', p('cli_count_dir'), check=False)
        assert r.returncode != 0


@pytest.mark.local('reports daemon/host status (version, pid, home, dir sizes); not catalog-specific')
class TestStatus:
    def test_basics(self, cli: PxtRunner) -> None:
        # --json: paths are reported raw (no redaction)
        out = cli('status', '--json').json
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
        text = cli('status').stdout
        assert all(h in text for h in ('pxt_version', 'daemon_pid', 'home'))
        assert '(-)' not in text
        media_line = next(ln for ln in text.splitlines() if ln.startswith('media_dir'))
        assert '(' not in media_line

        # --sizes: _fmt_size renders a B/KB/MB suffix once sizes are populated
        sized = cli('status', '--sizes').stdout
        assert any(unit in sized for unit in ('B)', 'KB)', 'MB)', 'GB)'))


@pytest.mark.local('reports resolved configuration settings; not catalog-specific')
class TestConfig:
    def test_basics(self, cli: PxtRunner) -> None:
        """cli config reports every documented configuration setting with its resolved value and
        source ('env', 'unset', or the path of the file the value came from). Credentials show
        '<redacted>' when set; the source field reveals presence even when the value is masked."""
        out = cli('config', '--json').json
        assert out['config_file'].endswith('config.toml')
        entries = out['entries']

        # every entry has the expected shape
        expected_keys = {'section', 'key', 'value', 'source', 'description', 'expected_type'}
        assert all(set(e.keys()) == expected_keys for e in entries)
        # source is 'env', 'unset', or an absolute file path
        assert all(e['source'] in ('env', 'unset') or os.path.isabs(e['source']) for e in entries)

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
        text = cli('config').stdout
        assert 'config_file' in text
        # openai.api_key appears either in the aligned table (if set) or in the not-set list
        assert 'openai.api_key' in text
        # the unset bucket is summarized on a single line; the per-row '(unset)' marker is gone
        assert '(unset)' not in text

        # -v: descriptions inline under each entry, and unset entries get full table rows
        # (with '-' as the value) so the description has somewhere to land.
        verbose = cli('config', '-v').stdout
        assert 'config_file' in verbose
        assert 'not set:' not in verbose
        assert '[unset]' in verbose
        # at least one description (every entry in the registry has one)
        openai_key = next(
            e for e in cli('config', '--json').json['entries'] if e['section'] == 'openai' and e['key'] == 'api_key'
        )
        assert openai_key['description'] in verbose

    def test_filters(self, cli: PxtRunner) -> None:
        """--section filters to one section; --source filters by exact source string ('env',
        'unset', or a file path)."""
        out = cli('config', '--section', 'openai', '--json').json
        assert all(e['section'] == 'openai' for e in out['entries'])

        out = cli('config', '--source', 'unset', '--json').json
        assert all(e['source'] == 'unset' for e in out['entries'])

        # --source unset text mode collapses everything into the "not set" line - no table header
        text = cli('config', '--source', 'unset').stdout
        assert 'not set:' in text
        # no per-entry table rows when everything is unset
        assert '[unset]' not in text

        # an unmatched source string returns no entries but exits cleanly
        out = cli('config', '--source', 'no-such-path', '--json').json
        assert out['entries'] == []


class TestErrors:
    """The `cli errors` command itself."""

    def test_basics(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        """Populated + empty cases, JSON and text."""
        p = make_catalog_path
        pxt.create_dir(p('cli_errs'), if_exists='ignore')

        # No computed columns -> no errors -> empty JSON list and empty text output
        pxt.create_table(p('cli_errs.ok'), {'k': pxt.Required[pxt.Int]}, primary_key='k', if_exists='replace')
        assert cli('errors', p('cli_errs/ok'), '--json').json == []
        r = cli('errors', p('cli_errs/ok'))
        assert r.returncode == 0
        assert r.stdout.strip() == ''

        # Computed column that raises for k=0: row 0 shows up in the errors listing
        t = pxt.create_table(p('cli_errs.bad'), {'k': pxt.Required[pxt.Int]}, primary_key='k', if_exists='replace')
        t.add_computed_column(b=_fail_on_zero(t.k), on_error='ignore')
        t.add_computed_column(c=_fail_on_zero(t.k), on_error='ignore')
        t.insert([{'k': 0}, {'k': 1}], on_error='ignore')
        text = cli('errors', p('cli_errs/bad')).stdout
        assert 'k: 0' in text
        assert 'b' in text

        # --col filters to one stored computed column: an existing column with errors comes back,
        # and asking for the other computed column also returns its errors.
        out_b = cli('errors', p('cli_errs/bad'), '--col', 'b', '--json').json
        assert len(out_b) == 1
        assert out_b[0]['column'] == 'b'
        out_c = cli('errors', p('cli_errs/bad'), '--col', 'c', '--json').json
        assert len(out_c) == 1
        assert out_c[0]['column'] == 'c'

    def test_errors(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        """No-PK rejection, unknown --col, --col on a non-stored-computed column."""
        p = make_catalog_path
        pxt.create_dir(p('cli_errs_err'), if_exists='ignore')
        pxt.create_table(p('cli_errs_err.no_pk'), {'a': pxt.Int}, if_exists='replace')
        pxt.create_table(
            p('cli_errs_err.t'), {'k': pxt.Required[pxt.Int], 'plain': pxt.String}, primary_key='k', if_exists='replace'
        )

        # no PK -> 400
        r = cli('errors', p('cli_errs_err/no_pk'), check=False)
        assert r.returncode != 0
        assert 'primary key' in r.stderr.lower()

        # unknown --col
        r = cli('errors', p('cli_errs_err/t'), '--col', 'nope', check=False)
        assert r.returncode != 0
        assert 'unknown column' in r.stderr

        # --col on a column that isn't a stored computed column
        r = cli('errors', p('cli_errs_err/t'), '--col', 'plain', check=False)
        assert r.returncode != 0
        assert 'stored computed column' in r.stderr


class TestDrop:
    """`cli drop` (tables) and `cli rm` (directories). They share the universal mutation
    surface (--force, --dry-run, --json) and the no-TTY refusal."""

    def test_basics(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        pxt.create_dir(p('cli_drop'), if_exists='ignore')
        pxt.create_dir(p('cli_drop.nest'), if_exists='ignore')
        pxt.create_table(p('cli_drop.nest.victim'), {'a': pxt.Int}, if_exists='replace')
        pxt.create_table(p('cli_drop.dry'), {'a': pxt.Int}, if_exists='replace')
        pxt.create_table(p('cli_drop.txt'), {'a': pxt.Int}, if_exists='replace')

        # drop a table: --json reports dropped=True; table is gone
        out = cli('drop', p('cli_drop/nest/victim'), '-f', '--json').json
        assert out['dropped'] is True
        assert pxt.get_table(p('cli_drop/nest/victim'), if_not_exists='ignore') is None

        # rm the (now empty) directory
        assert cli('drop-dir', p('cli_drop/nest'), '-f', '--json').json['dropped'] is True

        # rm refuses a non-empty dir without -r; -r succeeds
        pxt.create_dir(p('cli_drop.nest2'), if_exists='ignore')
        pxt.create_table(p('cli_drop.nest2.t'), {'a': pxt.Int}, if_exists='replace')
        assert cli('drop-dir', p('cli_drop/nest2'), '-f', check=False).returncode != 0
        assert cli('drop-dir', p('cli_drop/nest2'), '-r', '-f', '--json').json['dropped'] is True

        # dry-run text + --json: no side effect, message includes 'would drop'
        r = cli('drop', p('cli_drop/dry'), '-n')
        assert 'would drop' in r.stdout
        assert pxt.get_table(p('cli_drop/dry'), if_not_exists='ignore') is not None

        # plain text confirmation
        assert 'dropped' in cli('drop', p('cli_drop/txt'), '-f').stdout

        # rm text output
        pxt.create_dir(p('cli_drop_rmtxt'), if_exists='ignore')
        assert 'removed' in cli('drop-dir', p('cli_drop_rmtxt'), '-f').stdout

        # rm dry-run text (both with and without -r)
        pxt.create_dir(p('cli_drop_dr'), if_exists='ignore')
        r = cli('drop-dir', p('cli_drop_dr'), '-n')
        assert 'would remove' in r.stdout
        r = cli('drop-dir', p('cli_drop_dr'), '-n', '-r')
        assert 'would remove' in r.stdout
        assert 'recursive' in r.stdout

    def test_cascade(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        """--cascade maps to force=True server-side: drops a table that has dependent views.
        Without --cascade, dropping a table with a view fails."""
        p = make_catalog_path
        pxt.create_dir(p('cli_drop_csc'), if_exists='ignore')
        t = pxt.create_table(p('cli_drop_csc.base'), {'a': pxt.Int}, if_exists='replace')
        pxt.create_view(p('cli_drop_csc.dep_view'), t, if_exists='replace')

        # without --cascade: drop fails because of the dependent view
        r = cli('drop', p('cli_drop_csc/base'), '-f', check=False)
        assert r.returncode != 0
        assert pxt.get_table(p('cli_drop_csc/base'), if_not_exists='ignore') is not None

        # with --cascade: both base table and dependent view are dropped
        assert cli('drop', p('cli_drop_csc/base'), '--cascade', '-f', '--json').json['dropped'] is True
        assert pxt.get_table(p('cli_drop_csc/base'), if_not_exists='ignore') is None
        assert pxt.get_table(p('cli_drop_csc/dep_view'), if_not_exists='ignore') is None

    def test_errors(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        # no -f, no TTY: refuse to proceed
        pxt.create_dir(p('cli_drop_err'), if_exists='ignore')
        pxt.create_table(p('cli_drop_err.protected'), {'a': pxt.Int}, if_exists='replace')
        r = cli('drop', p('cli_drop_err/protected'), check=False)
        assert r.returncode != 0
        assert '--force' in r.stderr
        # still exists
        assert pxt.get_table(p('cli_drop_err/protected'), if_not_exists='ignore') is not None

        # drop on a directory is refused (server-side: drop_table on a dir errors)
        pxt.create_dir(p('cli_drop_err_dir'), if_exists='ignore')
        r = cli('drop', p('cli_drop_err_dir'), '-f', check=False)
        assert r.returncode != 0


class TestRename:
    def test_basics(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        pxt.create_dir(p('cli_rn'), if_exists='ignore')
        pxt.create_table(p('cli_rn.old_name'), {'a': pxt.Int}, if_exists='replace')
        pxt.create_table(p('cli_rn.dr'), {'a': pxt.Int}, if_exists='replace')
        pxt.create_table(p('cli_rn.txt'), {'a': pxt.Int}, if_exists='replace')

        # rename + --json
        out = cli('rename', p('cli_rn/old_name'), 'new_name', '--json').json
        assert out['new_path'] == p('cli_rn/new_name')
        assert pxt.get_table(p('cli_rn/new_name'), if_not_exists='ignore') is not None
        assert pxt.get_table(p('cli_rn/old_name'), if_not_exists='ignore') is None

        # dry-run: 'would rename', no side effect
        r = cli('rename', p('cli_rn/dr'), 'dr2', '-n')
        assert 'would rename' in r.stdout
        assert pxt.get_table(p('cli_rn/dr'), if_not_exists='ignore') is not None

        # text confirmation
        assert 'renamed' in cli('rename', p('cli_rn/txt'), 'txt2').stdout

    def test_errors(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        """`new_name` must be a leaf, no '/' or '.'."""
        p = make_catalog_path
        pxt.create_dir(p('cli_rn_err'), if_exists='ignore')
        pxt.create_table(p('cli_rn_err.t'), {'a': pxt.Int}, if_exists='replace')
        r = cli('rename', p('cli_rn_err/t'), 'a/b', check=False)
        assert r.returncode != 0
        assert 'leaf name' in r.stderr


class TestMv:
    def test_basics(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        pxt.create_dir(p('cli_mv'), if_exists='ignore')
        pxt.create_dir(p('cli_mv.src'), if_exists='ignore')
        pxt.create_dir(p('cli_mv.dst'), if_exists='ignore')
        pxt.create_table(p('cli_mv.src.movee'), {'a': pxt.Int}, if_exists='replace')
        pxt.create_table(p('cli_mv.toroot'), {'a': pxt.Int}, if_exists='replace')
        pxt.create_table(p('cli_mv.dr'), {'a': pxt.Int}, if_exists='replace')

        # mv into another dir: --json reports the new path
        out = cli('mv', p('cli_mv/src/movee'), p('cli_mv/dst'), '--json').json
        assert out['new_path'] == p('cli_mv/dst/movee')
        assert pxt.get_table(p('cli_mv/dst/movee'), if_not_exists='ignore') is not None

        # text confirmation
        pxt.create_table(p('cli_mv.src.tx'), {'a': pxt.Int}, if_exists='replace')
        assert 'moved' in cli('mv', p('cli_mv/src/tx'), p('cli_mv/dst')).stdout

        # empty destination moves to the catalog root (p('') is '' locally, the db uri over proxy; a
        # literal '/' would target the local root even over proxy, i.e. a different catalog)
        cli('mv', p('cli_mv/toroot'), p(''))
        assert pxt.get_table(p('toroot'), if_not_exists='ignore') is not None

        # dry-run: 'would move', no side effect
        r = cli('mv', p('cli_mv/dr'), p('cli_mv'), '-n')
        assert 'would move' in r.stdout

    def test_errors(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        # nonexistent source path
        r = cli('mv', p('cli_mv_err/missing'), p('cli_mv_err'), check=False)
        assert r.returncode != 0


class TestRevert:
    def test_basics(self, cli: PxtRunner, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        pxt.create_dir(p('cli_rv'), if_exists='ignore')
        t = pxt.create_table(p('cli_rv.t'), {'a': pxt.Int}, if_exists='replace')
        t.insert([{'a': 1}])
        v_before = t.get_metadata()['version']
        pxt.create_table(p('cli_rv.txt'), {'a': pxt.Int}, if_exists='replace').insert([{'a': 1}])
        pxt.create_table(p('cli_rv.dr'), {'a': pxt.Int}, if_exists='replace')

        # revert + --json
        out = cli('revert', p('cli_rv/t'), '-f', '--json').json
        assert out['from_version'] == v_before
        assert out['to_version'] == v_before - 1

        # text: 'reverted ... -> ...'
        text = cli('revert', p('cli_rv/txt'), '-f').stdout
        assert 'reverted' in text
        assert '->' in text

        # dry-run: 'would revert', no side effect
        assert 'would revert' in cli('revert', p('cli_rv/dr'), '-n').stdout

    def test_errors(self, cli: PxtRunner, make_catalog_path: Callable[[str], str], pxt_daemon: int) -> None:
        """Client preflight: --steps must be >= 1. Server: cannot revert past version 0.
        Plus a direct HTTP test confirming the server's own steps<1 check fires for any
        future programmatic caller that bypasses the CLI."""
        p = make_catalog_path
        pxt.create_dir(p('cli_rv_err'), if_exists='ignore')
        pxt.create_table(p('cli_rv_err.t'), {'a': pxt.Int}, if_exists='replace')

        # client preflight
        r = cli('revert', p('cli_rv_err/t'), '--steps', '0', '-f', check=False)
        assert r.returncode != 0
        assert '--steps must be >= 1' in r.stderr

        # server preflight: cannot revert beyond the current version
        r = cli('revert', p('cli_rv_err/t'), '--steps', '999', '-f', check=False)
        assert r.returncode != 0
        assert 'cannot revert' in r.stderr

        # direct HTTP: server's own steps<1 check fires when the client preflight is bypassed
        req = urllib.request.Request(
            f'http://127.0.0.1:{pxt_daemon}/api/tables/whatever/revert',
            data=json.dumps({'steps': 0}).encode(),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(req)
        # 422: server-side semantic validation (excs.ErrorCode.INVALID_ARGUMENT)
        assert ei.value.code == 422
        assert 'steps must be >= 1' in json.loads(ei.value.read())['detail']


@pytest.mark.local('client-side path-shape validator; the pxt:// prefix is validated elsewhere')
class TestPathValidator:
    """Client-side path validator (pixeltable_cli.client.http.quote_path). Catches every well-known
    bad shape before the request reaches the server so the user gets a clear error message
    instead of a generic 'Invalid path' from pxt."""

    def test_rejects_bad_shapes(self, cli: PxtRunner) -> None:
        # '.' is reserved (pxt's legacy separator)
        r = cli('describe', 'a.b', check=False)
        assert r.returncode != 0
        assert "'/'" in r.stderr or 'separator' in r.stderr
        # leading '/' would create an empty leading component
        r = cli('describe', '/x', check=False)
        assert r.returncode != 0
        assert 'relative' in r.stderr
        # trailing '/' would create an empty trailing component
        r = cli('describe', 'x/', check=False)
        assert r.returncode != 0
        assert "must not end with '/'" in r.stderr
        # '//' produces an empty internal component
        r = cli('describe', 'a//b', check=False)
        assert r.returncode != 0
        assert 'empty components' in r.stderr

    def test_server_rejects_control_chars(self, pxt_daemon: int) -> None:
        # A control character in the URL-decoded path would otherwise be interpolated into
        # response headers (eg the Content-Disposition emitted by dashboard_table_export),
        # enabling header injection / response splitting. The server-side validator must
        # reject every ASCII control character before the path reaches any downstream sink.
        # LF (%0A) is filtered out earlier by the route-matching regex; the test covers the
        # remaining control chars that do reach _validate_path.
        for encoded in ('foo%0Dbar', 'foo%00bar', 'foo%7Fbar', 'foo%01bar'):
            req = urllib.request.Request(f'http://127.0.0.1:{pxt_daemon}/api/tables/{encoded}')
            with pytest.raises(urllib.error.HTTPError) as ei:
                urllib.request.urlopen(req)
            assert ei.value.code == 422
            assert 'control characters' in json.loads(ei.value.read())['detail']


@pytest.mark.local('client-side --cols token validator; fires before the request reaches the catalog')
class TestColsValidator:
    """Client-side --cols validator (parser.parse_cols). Rejects every shape that would
    yield an empty token. Shared between `rows` and `get`."""

    def test_rejects_empty_tokens(self, cli: PxtRunner) -> None:
        pxt.create_dir('cli_colsv', if_exists='ignore')
        pxt.create_table('cli_colsv.r', {'a': pxt.Int, 'b': pxt.Int}, if_exists='replace')
        pxt.create_table('cli_colsv.g', {'k': pxt.Required[pxt.Int]}, primary_key='k', if_exists='replace')
        results = [cli('rows', 'cli_colsv/r', '--cols', bad, check=False) for bad in ('a,', ',a', 'a,,b')]
        assert all(r.returncode != 0 for r in results)
        assert all('must not be empty' in r.stderr for r in results)
        r = cli('get', 'cli_colsv/g', '1', '--cols', 'k,', check=False)
        assert r.returncode != 0
        assert 'must not be empty' in r.stderr


@pytest.mark.local('top-level CLI surface (help, argparse); no catalog interaction')
class TestCli:
    """Top-level CLI surface (help, unknown commands, argparse errors)."""

    def test_help(self, cli: PxtRunner) -> None:
        # --help: lists every command, exits 0
        r = cli('--help')
        assert r.returncode == 0
        assert all(name in r.stdout for name in ('health', 'ls', 'shell'))

        # no args: prints usage and exits 0 (matches the prior pxt behavior)
        r = cli(check=False)
        assert r.returncode == 0
        assert 'usage' in r.stdout.lower() or 'usage' in r.stderr.lower()

    def test_unknown_command(self, cli: PxtRunner) -> None:
        r = cli('not_a_command', check=False)
        assert r.returncode == 2
        assert 'unknown command' in r.stderr

    def test_subcommand_arg_errors(self, cli: PxtRunner) -> None:
        # rows is missing the required path positional; argparse prints usage + epilog
        r = cli('rows', check=False)
        assert r.returncode == 2
        assert 'usage' in r.stderr.lower()
        assert 'Examples' in r.stderr  # the per-command epilog block is appended on error


@pytest.mark.local('dashboard SPA routes read the daemon in-process catalog directly over HTTP')
class TestDashboard:
    """Dashboard SPA + SPA-only routes are always available when the daemon is up."""

    def test_search(self, cli: PxtRunner, pxt_daemon: int) -> None:
        pxt.create_dir('cli_dash', if_exists='ignore')
        pxt.create_table('cli_dash.t', {'x': pxt.Int}, if_exists='replace')
        search_url = f'http://127.0.0.1:{pxt_daemon}/api/dashboard/search?q=cli_dash'
        with urllib.request.urlopen(search_url, timeout=5) as r:
            data = json.loads(r.read())
        assert data['query'] == 'cli_dash'
        assert any(d['path'] == 'cli_dash' for d in data['directories'])

    def test_table_meta_data_export(self, cli: PxtRunner, pxt_daemon: int) -> None:
        pxt.create_dir('cli_dash_t', if_exists='ignore')
        t = pxt.create_table('cli_dash_t.t', {'x': pxt.Int}, if_exists='replace')
        t.insert([{'x': 1}, {'x': 2}, {'x': 3}])

        base = f'http://127.0.0.1:{pxt_daemon}'
        with urllib.request.urlopen(f'{base}/api/dashboard/tables/cli_dash_t/t/meta', timeout=5) as r:
            meta = json.loads(r.read())
        assert 'columns' in meta
        assert 'x' in meta['columns']

        with urllib.request.urlopen(f'{base}/api/dashboard/tables/cli_dash_t/t/data?limit=10', timeout=5) as r:
            data = json.loads(r.read())
        assert data['total_count'] == 3
        assert all('x' in row for row in data['rows'])

        with urllib.request.urlopen(f'{base}/api/dashboard/tables/cli_dash_t/t/export?limit=10', timeout=5) as r:
            csv_body = r.read().decode('utf-8')
            disp = r.headers.get('Content-Disposition', '')
        assert csv_body.splitlines()[0] == 'x'
        assert 'cli_dash_t_t.csv' in disp

    def test_dirs_and_status_contract(self, cli: PxtRunner, pxt_daemon: int) -> None:
        """Pin the response shapes the dashboard SPA reads in dashboard/src/api/client.ts.
        getDirectoryTree reads the node list from tree.entries of /api/dirs?tree=true (the response
        is an object, not a top-level array), and getStatus reads the flat pxt_version / home /
        total_* fields from /api/status and maps them into its own nested shape."""
        pxt.create_dir('cli_dash_contract', if_exists='ignore')
        pxt.create_table('cli_dash_contract.t', {'x': pxt.Int}, if_exists='replace')
        base = f'http://127.0.0.1:{pxt_daemon}'

        with urllib.request.urlopen(f'{base}/api/dirs?tree=true', timeout=5) as r:
            dirs = json.loads(r.read())
        # The SPA indexes into tree.entries; calling .reduce() on the response itself is the crash
        # this guards against, so the response must be an object and tree.entries must be a list.
        assert isinstance(dirs, dict)
        assert isinstance(dirs['tree']['entries'], list)
        dir_node = next(n for n in dirs['tree']['entries'] if n['path'] == 'cli_dash_contract')
        assert dir_node['kind'] == 'directory'
        assert all(k in dir_node for k in ('name', 'path', 'kind', 'entries'))
        table_node = next(n for n in dir_node['entries'] if n['path'] == 'cli_dash_contract/t')
        assert table_node['kind'] == 'table'
        assert all(k in table_node for k in ('name', 'path', 'kind', 'version', 'error_count', 'base'))

        with urllib.request.urlopen(f'{base}/api/status', timeout=5) as r:
            status = json.loads(r.read())
        # The fields the SPA maps from. They are flat on the response; a regression to a nested
        # {version, config} shape would silently break getStatus, so pin the flat layout.
        assert all(
            k in status
            for k in ('pxt_version', 'home', 'db_url', 'media_dir', 'file_cache_dir', 'total_tables', 'total_errors')
        )
        assert 'version' not in status
        assert 'config' not in status

    def test_spa_static_files(self, cli: PxtRunner, pxt_daemon: int) -> None:
        """GET / returns the SPA shell; bundled assets are served from /."""
        base = f'http://127.0.0.1:{pxt_daemon}'
        with urllib.request.urlopen(f'{base}/', timeout=5) as r:
            body = r.read().decode('utf-8')
            ctype = r.headers.get('Content-Type', '')
        assert '<!doctype html>' in body.lower()
        assert 'text/html' in ctype
        # Stable bundled asset; the hashed JS/CSS filenames change every build.
        with urllib.request.urlopen(f'{base}/favicon.svg', timeout=5) as r:
            assert r.status == 200
            assert 'svg' in r.headers.get('Content-Type', '').lower()


@pytest.mark.local('dashboard command is a URL launcher; no catalog interaction')
class TestDashboardCommand:
    """The `pxt dashboard` command (a thin URL-launcher) exercised through subprocess."""

    def test_prints_url(self, cli: PxtRunner, pxt_daemon: int) -> None:
        # Note: normally, `pxt dashboard` opens browser, but `PxtRunner` suppresses that in tests.
        r = cli('dashboard')
        assert r.returncode == 0
        assert f':{pxt_daemon}' in r.stdout
        with urllib.request.urlopen(f'http://127.0.0.1:{pxt_daemon}/', timeout=5) as resp:
            assert resp.status == 200

    def test_help_exits_zero(self, cli: PxtRunner, pxt_daemon: int) -> None:
        r = cli('--help')
        assert 'dashboard' in r.stdout
        r = cli('dashboard', '--help')
        assert r.returncode == 0


@pytest.mark.local('measures client import cost; independent of the catalog backend')
class TestColdStartBudget:
    """Locks in the cold-start budget for daemon-routed commands.

    `pxt ls`, `pxt health`, etc. run in a stdlib-only client that hands off to the daemon
    over HTTP. Importing pixeltable in that path would re-pull the heavy dependency tree
    (sqlalchemy, pandas, torch, ...) on every invocation, blowing through the latency
    budget and defeating the daemon split. The `-X importtime` log is authoritative.
    """

    def test_pixeltable_not_imported_by_pxt_ls(self, cli: PxtRunner, pxt_daemon: int) -> None:
        # Use sys.executable so the subprocess runs under the same interpreter as the test,
        # not whatever python resolves to on PATH.
        env = {**os.environ, 'PXT_PORT': str(pxt_daemon)}
        r = subprocess.run(
            [sys.executable, '-X', 'importtime', '-m', 'pixeltable_cli.client.main', 'ls'],
            capture_output=True,
            text=True,
            env=env,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        # We only inspect the import log; the underlying ls call may pass or fail
        # depending on catalog state, which is irrelevant here.
        imported = [line for line in r.stderr.splitlines() if line.startswith('import time:')]
        # Each line of the form 'import time: ...' ends with the dotted module name; we want to catch
        # the top-level package alone, not e.g. a stdlib numbers module sharing a prefix.
        forbidden = ('pixeltable', 'sqlalchemy', 'pandas', 'numpy')
        offenders: dict[str, str] = {}
        for line in imported:
            module = line.rsplit(' ', 1)[-1]
            top = module.split('.', 1)[0]
            if top in forbidden and top not in offenders:
                offenders[top] = line
        assert len(offenders) == 0, (
            'cold-start budget broken; the following packages were imported during `pxt ls` startup:\n'
            + '\n'.join(f'  {pkg}: {line}' for pkg, line in offenders.items())
        )
