"""Smoke tests: drive state via the pxt API, validate via pcli."""

import json
import os

import pytest

import pixeltable as pxt

from .conftest import PcliRunner


@pxt.udf
def _boom_on_zero(x: int) -> int:
    """Module-level UDF: raises for k=0 so we can populate a stored errortype column."""
    if x == 0:
        raise ValueError('boom')
    return x


class TestPcliSmoke:
    def test_health(self, pcli: PcliRunner) -> None:
        out = pcli('health').json
        assert out['ok'] is True
        assert out['pid'] > 0

    def test_ls_reflects_api_state(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        pxt.create_table('pcli_smoke.t', {'x': pxt.Int, 'y': pxt.String}, if_exists='ignore')

        entries = pcli('ls', 'pcli_smoke', '--json').json['entries']
        paths = {e['path'] for e in entries}
        assert 'pcli_smoke/t' in paths

        pxt.drop_table('pcli_smoke.t')
        entries = pcli('ls', 'pcli_smoke', '--json').json['entries']
        paths = {e['path'] for e in entries}
        assert 'pcli_smoke/t' not in paths

    def test_describe(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        pxt.create_table('pcli_smoke.describe_me', {'a': pxt.Int}, if_exists='ignore')

        out = pcli('describe', 'pcli_smoke/describe_me', '--json').json
        assert 'a' in out['columns']

    def test_rows(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        t = pxt.create_table('pcli_smoke.rows', {'n': pxt.Int, 's': pxt.String}, if_exists='replace')
        t.insert([{'n': i, 's': f'row{i}'} for i in range(5)])

        out = pcli('rows', 'pcli_smoke/rows', '-n', '3', '--json').json
        assert len(out) == 3

        out2 = pcli('rows', 'pcli_smoke/rows', '-n', '10', '--cols', 'n', '--json').json
        assert sorted(r['n'] for r in out2) == [0, 1, 2, 3, 4]

    def test_columns(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        pxt.create_table('pcli_smoke.cols', {'a': pxt.Int, 'b': pxt.String}, if_exists='replace')

        entries = pcli('columns', 'pcli_smoke/cols', '--json').json
        names = {e['column'] for e in entries}
        assert names == {'a', 'b'}

    def test_idxs_no_embedding(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        pxt.create_table('pcli_smoke.no_idx', {'a': pxt.Int}, if_exists='replace')

        entries = pcli('idxs', 'pcli_smoke/no_idx', '--json').json
        assert all(e['index_type'] != 'embedding' for e in entries)

    def test_history(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        t = pxt.create_table('pcli_smoke.hist', {'a': pxt.Int}, if_exists='replace')
        t.insert([{'a': 1}])

        out = pcli('history', 'pcli_smoke/hist', '--json').json
        assert len(out) >= 2

    def test_count(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        t = pxt.create_table('pcli_smoke.cnt', {'a': pxt.Int}, if_exists='replace')
        t.insert([{'a': i} for i in range(7)])

        out = pcli('count', 'pcli_smoke/cnt', '--json').json
        assert out['count'] == 7

        # plain output is just the integer
        plain = pcli('count', 'pcli_smoke/cnt').stdout.strip()
        assert plain == '7'

    def test_get_by_pk(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        t = pxt.create_table(
            'pcli_smoke.pk_tbl', {'k': pxt.Required[pxt.Int], 'v': pxt.String}, primary_key='k', if_exists='replace'
        )
        t.insert([{'k': 1, 'v': 'one'}, {'k': 2, 'v': 'two'}])

        out = pcli('get', 'pcli_smoke/pk_tbl', '2', '--json').json
        assert out['pk_columns'] == ['k']
        assert out['row']['v'] == 'two'

        # missing row
        out = pcli('get', 'pcli_smoke/pk_tbl', '99', '--json').json
        assert out['row'] is None

    def test_get_requires_pk(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        pxt.create_table('pcli_smoke.no_pk_get', {'a': pxt.Int}, if_exists='replace')

        r = pcli('get', 'pcli_smoke/no_pk_get', '1', check=False)
        assert r.returncode != 0
        assert 'primary key' in r.stderr.lower()

    def test_status(self, pcli: PcliRunner) -> None:
        out = pcli('status', '--json').json
        assert out['pxt_version'] != ''
        assert out['pid'] > 0
        # PIXELTABLE_HOME-rooted paths must be returned in redacted form, not the resolved path.
        home = os.environ['PIXELTABLE_HOME']
        path_keys = ('home', 'media_dir', 'file_cache_dir')
        assert all(out.get(k) is None or home not in out[k] for k in path_keys)
        assert all(out.get(k) is None or out[k].startswith('$PIXELTABLE_HOME') for k in path_keys)

    def test_env(self, pcli: PcliRunner) -> None:
        out = pcli('env', '--json').json
        # PIXELTABLE_DB is always set by init_env
        assert 'PIXELTABLE_DB' in out['env_vars']

    def test_computed(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        t = pxt.create_table('pcli_smoke.cmp', {'a': pxt.Int}, if_exists='replace')
        t.add_computed_column(b=t.a * 2)

        entries = pcli('computed', 'pcli_smoke/cmp', '--json').json
        names = {e['column'] for e in entries}
        assert names == {'b'}

    def test_drop_and_rm(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        pxt.create_dir('pcli_smoke.tmp_dir', if_exists='ignore')
        pxt.create_table('pcli_smoke.tmp_dir.victim', {'a': pxt.Int}, if_exists='replace')

        # drop the table
        out = pcli('drop', 'pcli_smoke/tmp_dir/victim', '-f', '--json').json
        assert out['dropped'] is True
        assert pxt.get_table('pcli_smoke/tmp_dir/victim', if_not_exists='ignore') is None

        # remove the now-empty dir
        out = pcli('rm', 'pcli_smoke/tmp_dir', '-f', '--json').json
        assert out['dropped'] is True

    def test_rm_recursive(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        pxt.create_dir('pcli_smoke.nest', if_exists='ignore')
        pxt.create_table('pcli_smoke.nest.t', {'a': pxt.Int}, if_exists='replace')

        # non-recursive should fail
        r = pcli('rm', 'pcli_smoke/nest', '-f', check=False)
        assert r.returncode != 0

        # recursive succeeds
        out = pcli('rm', 'pcli_smoke/nest', '-r', '-f', '--json').json
        assert out['dropped'] is True

    def test_drop_dry_run(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        pxt.create_table('pcli_smoke.dry_run_target', {'a': pxt.Int}, if_exists='replace')

        r = pcli('drop', 'pcli_smoke/dry_run_target', '-n')
        assert 'would drop' in r.stdout
        # still exists
        assert pxt.get_table('pcli_smoke/dry_run_target', if_not_exists='ignore') is not None

    def test_no_force_no_tty(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        pxt.create_table('pcli_smoke.protected', {'a': pxt.Int}, if_exists='replace')

        r = pcli('drop', 'pcli_smoke/protected', check=False)
        assert r.returncode != 0
        assert '--force' in r.stderr
        assert pxt.get_table('pcli_smoke/protected', if_not_exists='ignore') is not None

    def test_rename(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        pxt.create_table('pcli_smoke.old_name', {'a': pxt.Int}, if_exists='replace')

        out = pcli('rename', 'pcli_smoke/old_name', 'new_name', '--json').json
        assert out['new_path'] == 'pcli_smoke/new_name'
        assert pxt.get_table('pcli_smoke/new_name', if_not_exists='ignore') is not None
        assert pxt.get_table('pcli_smoke/old_name', if_not_exists='ignore') is None

    def test_mv(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        pxt.create_dir('pcli_smoke.src', if_exists='ignore')
        pxt.create_dir('pcli_smoke.dst', if_exists='ignore')
        pxt.create_table('pcli_smoke.src.movee', {'a': pxt.Int}, if_exists='replace')

        out = pcli('mv', 'pcli_smoke/src/movee', 'pcli_smoke/dst', '--json').json
        assert out['new_path'] == 'pcli_smoke/dst/movee'
        assert pxt.get_table('pcli_smoke/dst/movee', if_not_exists='ignore') is not None

    def test_revert(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        t = pxt.create_table('pcli_smoke.revert_me', {'a': pxt.Int}, if_exists='replace')
        t.insert([{'a': 1}])
        v_before = t.get_metadata()['version']

        out = pcli('revert', 'pcli_smoke/revert_me', '-f', '--json').json
        assert out['from_version'] == v_before
        assert out['to_version'] == v_before - 1

    def test_errors_requires_pk(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        pxt.create_table('pcli_smoke.no_pk', {'a': pxt.Int}, if_exists='replace')

        r = pcli('errors', 'pcli_smoke/no_pk', check=False)
        assert r.returncode != 0
        assert 'primary key' in r.stderr.lower()


class TestPcliTextOutput:
    """Plain (non-JSON) human-format output for every command."""

    def test_ls_default(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        pxt.create_table('pcli_text.t', {'a': pxt.Int}, if_exists='replace')
        out = pcli('ls', 'pcli_text').stdout
        assert 'path' in out
        assert 'kind' in out
        assert 'pcli_text/t' in out

    def test_ls_json_is_cheap_by_default(self, pcli: PcliRunner) -> None:
        """Bare --json must NOT trigger per-entry get_metadata(): num_cols/flags stay None/''.
        Users who want full schema info in JSON pass `-l --json`."""
        pxt.create_dir('pcli_text', if_exists='ignore')
        pxt.create_table('pcli_text.cheap', {'a': pxt.Int}, if_exists='replace')
        entries = pcli('ls', 'pcli_text', '--json').json['entries']
        row = next(e for e in entries if e['path'] == 'pcli_text/cheap')
        assert row['num_cols'] is None
        assert row['flags'] == ''

    def test_ls_long_json_includes_details(self, pcli: PcliRunner) -> None:
        """-l --json: the explicit opt-in, num_cols/flags get populated."""
        pxt.create_dir('pcli_text', if_exists='ignore')
        t = pxt.create_table('pcli_text.detailed', {'a': pxt.Int}, if_exists='replace')
        t.add_computed_column(b=t.a * 2)
        entries = pcli('ls', 'pcli_text', '-l', '--json').json['entries']
        row = next(e for e in entries if e['path'] == 'pcli_text/detailed')
        assert row['num_cols'] is not None and row['num_cols'] >= 2
        assert 'c' in row['flags']

    def test_ls_long(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        t = pxt.create_table('pcli_text.tl', {'a': pxt.Int}, if_exists='replace')
        t.add_computed_column(b=t.a * 2)
        out = pcli('ls', 'pcli_text', '-l').stdout
        assert 'cols' in out
        assert 'version' in out
        assert 'flags' in out
        # 'c' = has at least one computed column
        assert 'c' in out

    def test_ls_tree(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        pxt.create_dir('pcli_text.sub', if_exists='ignore')
        pxt.create_table('pcli_text.sub.leaf', {'a': pxt.Int}, if_exists='replace')
        out = pcli('ls', 'pcli_text', '--tree').stdout
        assert all(marker in out for marker in ('sub', 'leaf'))
        assert any(box in out for box in ('├──', '└──'))

    def test_ls_counts(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        t = pxt.create_table('pcli_text.c1', {'a': pxt.Int}, if_exists='replace')
        t.insert([{'a': i} for i in range(4)])
        entries = pcli('ls', 'pcli_text', '--counts', '--json').json['entries']
        row = next(e for e in entries if e['path'] == 'pcli_text/c1')
        assert row['num_rows'] == 4

    def test_ls_empty_dir(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text_empty', if_exists='ignore')
        r = pcli('ls', 'pcli_text_empty')
        assert r.returncode == 0

    def test_describe_text(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        pxt.create_table('pcli_text.d', {'a': pxt.Int, 'b': pxt.String}, if_exists='replace')
        out = pcli('describe', 'pcli_text/d').stdout
        # repr(table) lists column names
        assert 'a' in out
        assert 'b' in out

    def test_columns_text(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        pxt.create_table('pcli_text.cols2', {'a': pxt.Int}, if_exists='replace')
        out = pcli('columns', 'pcli_text/cols2').stdout
        assert 'a' in out
        assert 'stored' in out

    def test_computed_text(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        t = pxt.create_table('pcli_text.cmp2', {'a': pxt.Int}, if_exists='replace')
        t.add_computed_column(doubled=t.a * 2)
        out = pcli('computed', 'pcli_text/cmp2').stdout
        assert 'doubled' in out

    def test_idxs_text(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        pxt.create_table('pcli_text.i', {'a': pxt.Int}, if_exists='replace')
        # btree indexes auto-created for PK-less tables only via embedding; ensure command runs
        r = pcli('idxs', 'pcli_text/i')
        assert r.returncode == 0

    def test_rows_text(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        t = pxt.create_table('pcli_text.r', {'a': pxt.Int}, if_exists='replace')
        t.insert([{'a': 11}, {'a': 22}])
        out = pcli('rows', 'pcli_text/r').stdout
        assert '11' in out
        assert '22' in out

    def test_rows_text_with_nulls(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        t = pxt.create_table('pcli_text.rn', {'a': pxt.Int, 's': pxt.String}, if_exists='replace')
        t.insert([{'a': 1, 's': None}])
        out = pcli('rows', 'pcli_text/rn').stdout
        # null cells render as empty strings, not the literal 'None'
        assert 'None' not in out

    def test_get_text(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        t = pxt.create_table(
            'pcli_text.gtxt', {'k': pxt.Required[pxt.Int], 'v': pxt.String}, primary_key='k', if_exists='replace'
        )
        t.insert([{'k': 7, 'v': 'seven'}])
        out = pcli('get', 'pcli_text/gtxt', '7').stdout
        assert 'seven' in out

    def test_get_pk_coercion(self, pcli: PcliRunner) -> None:
        # Exercises _coerce's int -> float -> JSON literal -> string fallback ladder.
        pxt.create_dir('pcli_text', if_exists='ignore')
        t = pxt.create_table(
            'pcli_text.gcoerce', {'k': pxt.Required[pxt.Float], 'v': pxt.String}, primary_key='k', if_exists='replace'
        )
        t.insert([{'k': 1.5, 'v': 'one-and-a-half'}])
        # float-looking value -> coerced to float
        out = pcli('get', 'pcli_text/gcoerce', '1.5', '--json').json
        assert out['row']['v'] == 'one-and-a-half'

    def test_get_pk_coercion_string_fallback(self, pcli: PcliRunner) -> None:
        # A token that's neither int/float nor a JSON literal falls through to str.
        pxt.create_dir('pcli_text', if_exists='ignore')
        t = pxt.create_table(
            'pcli_text.gstr', {'k': pxt.Required[pxt.String], 'v': pxt.Int}, primary_key='k', if_exists='replace'
        )
        t.insert([{'k': 'alpha', 'v': 1}])
        out = pcli('get', 'pcli_text/gstr', 'alpha', '--json').json
        assert out['row']['v'] == 1

    def test_get_pk_coercion_json_literal(self, pcli: PcliRunner) -> None:
        # JSON literal '"42"' forces a string PK that would otherwise be parsed as int.
        pxt.create_dir('pcli_text', if_exists='ignore')
        t = pxt.create_table(
            'pcli_text.gjson', {'k': pxt.Required[pxt.String], 'v': pxt.Int}, primary_key='k', if_exists='replace'
        )
        t.insert([{'k': '42', 'v': 99}])
        out = pcli('get', 'pcli_text/gjson', '"42"', '--json').json
        assert out['row']['v'] == 99

    def test_errors_json_empty(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        pxt.create_table('pcli_text.errjson', {'k': pxt.Required[pxt.Int]}, primary_key='k', if_exists='replace')
        out = pcli('errors', 'pcli_text/errjson', '--json').json
        assert out == []

    def test_rm_dry_run(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text_rmdr', if_exists='ignore')
        r = pcli('rm', 'pcli_text_rmdr', '-n')
        assert 'would remove' in r.stdout
        # still exists
        assert pxt.get_dir_tree() is not None  # smoke: command succeeded with no side effect

    def test_rm_dry_run_recursive(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text_rmdr2', if_exists='ignore')
        r = pcli('rm', 'pcli_text_rmdr2', '-n', '-r')
        assert 'would remove' in r.stdout
        assert 'recursive' in r.stdout

    def test_ls_counts_text(self, pcli: PcliRunner) -> None:
        # --counts without --json: exercises the text-formatter counts column.
        pxt.create_dir('pcli_text', if_exists='ignore')
        t = pxt.create_table('pcli_text.lctext', {'a': pxt.Int}, if_exists='replace')
        t.insert([{'a': i} for i in range(2)])
        out = pcli('ls', 'pcli_text', '--counts').stdout
        assert 'rows' in out
        assert '2' in out

    def test_columns_no_path(self, pcli: PcliRunner) -> None:
        # No-path branch walks _all_tables() and exercises columns/idxs try/except continue.
        pxt.create_dir('pcli_text', if_exists='ignore')
        pxt.create_table('pcli_text.global_cols', {'a': pxt.Int}, if_exists='replace')
        entries = pcli('columns', '--json').json
        names = {(e['table'], e['column']) for e in entries}
        assert ('pcli_text/global_cols', 'a') in names

    def test_idxs_no_path(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        pxt.create_table('pcli_text.global_idxs', {'a': pxt.Int}, if_exists='replace')
        # success regardless of catalog content; exercises _all_tables walk.
        r = pcli('idxs', '--json')
        assert r.returncode == 0

    def test_get_text_missing(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        pxt.create_table('pcli_text.gmiss', {'k': pxt.Required[pxt.Int]}, primary_key='k', if_exists='replace')
        out = pcli('get', 'pcli_text/gmiss', '999').stdout
        assert 'no row found' in out

    def test_history_text(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        t = pxt.create_table('pcli_text.h', {'a': pxt.Int}, if_exists='replace')
        t.insert([{'a': 1}])
        out = pcli('history', 'pcli_text/h').stdout
        assert 'version' in out
        assert 'change_type' in out

    def test_status_text(self, pcli: PcliRunner) -> None:
        out = pcli('status').stdout
        assert 'pxt_version' in out
        assert 'daemon_pid' in out
        assert 'home' in out
        # the '-' placeholder appears for unset fields when sizes weren't requested
        assert '-' in out

    def test_status_text_sizes(self, pcli: PcliRunner) -> None:
        out = pcli('status', '--sizes').stdout
        # _fmt_size renders a B/KB/MB suffix once sizes are populated
        assert any(unit in out for unit in ('B)', 'KB)', 'MB)', 'GB)'))

    def test_env_text(self, pcli: PcliRunner) -> None:
        out = pcli('env').stdout
        assert 'config_file' in out
        assert 'PIXELTABLE_DB=' in out
        # _CREDENTIAL_VARS split into set/unset summaries (one of the two is always present)
        assert 'credentials' in out

    def test_count_text(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        t = pxt.create_table('pcli_text.ct', {'a': pxt.Int}, if_exists='replace')
        t.insert([{'a': i} for i in range(3)])
        assert pcli('count', 'pcli_text/ct').stdout.strip() == '3'

    def test_errors_text_empty(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        pxt.create_table('pcli_text.errok', {'k': pxt.Required[pxt.Int]}, primary_key='k', if_exists='replace')
        # no computed columns, no errors -> empty output, success
        r = pcli('errors', 'pcli_text/errok')
        assert r.returncode == 0
        assert r.stdout.strip() == ''

    def test_errors_text_populated(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        t = pxt.create_table('pcli_text.errbad', {'k': pxt.Required[pxt.Int]}, primary_key='k', if_exists='replace')
        t.add_computed_column(b=_boom_on_zero(t.k), on_error='ignore')
        t.insert([{'k': 0}, {'k': 1}], on_error='ignore')
        out = pcli('errors', 'pcli_text/errbad').stdout
        assert 'k: 0' in out
        assert 'b' in out

    def test_drop_text(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        pxt.create_table('pcli_text.drp', {'a': pxt.Int}, if_exists='replace')
        out = pcli('drop', 'pcli_text/drp', '-f').stdout
        assert 'dropped' in out

    def test_rm_text(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text_rm', if_exists='ignore')
        out = pcli('rm', 'pcli_text_rm', '-f').stdout
        assert 'removed' in out

    def test_rename_text(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        pxt.create_table('pcli_text.rn_old', {'a': pxt.Int}, if_exists='replace')
        out = pcli('rename', 'pcli_text/rn_old', 'rn_new').stdout
        assert 'renamed' in out

    def test_rename_dry_run(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        pxt.create_table('pcli_text.rndr', {'a': pxt.Int}, if_exists='replace')
        r = pcli('rename', 'pcli_text/rndr', 'rndr2', '-n')
        assert 'would rename' in r.stdout
        # still exists under old name
        assert pxt.get_table('pcli_text/rndr', if_not_exists='ignore') is not None

    def test_rename_rejects_invalid_leaf(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        pxt.create_table('pcli_text.rnbad', {'a': pxt.Int}, if_exists='replace')
        r = pcli('rename', 'pcli_text/rnbad', 'a/b', check=False)
        assert r.returncode != 0
        assert 'leaf name' in r.stderr

    def test_mv_text(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        pxt.create_dir('pcli_text.mvsrc', if_exists='ignore')
        pxt.create_dir('pcli_text.mvdst', if_exists='ignore')
        pxt.create_table('pcli_text.mvsrc.x', {'a': pxt.Int}, if_exists='replace')
        out = pcli('mv', 'pcli_text/mvsrc/x', 'pcli_text/mvdst').stdout
        assert 'moved' in out

    def test_mv_to_root(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        pxt.create_table('pcli_text.toroot', {'a': pxt.Int}, if_exists='replace')
        pcli('mv', 'pcli_text/toroot', '/')
        assert pxt.get_table('toroot', if_not_exists='ignore') is not None

    def test_mv_dry_run(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        pxt.create_table('pcli_text.mvdr', {'a': pxt.Int}, if_exists='replace')
        r = pcli('mv', 'pcli_text/mvdr', 'pcli_text', '-n')
        assert 'would move' in r.stdout

    def test_revert_text(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        t = pxt.create_table('pcli_text.rv', {'a': pxt.Int}, if_exists='replace')
        t.insert([{'a': 1}])
        out = pcli('revert', 'pcli_text/rv', '-f').stdout
        assert 'reverted' in out
        assert '->' in out

    def test_revert_dry_run(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_text', if_exists='ignore')
        pxt.create_table('pcli_text.rvdr', {'a': pxt.Int}, if_exists='replace')
        r = pcli('revert', 'pcli_text/rvdr', '-n')
        assert 'would revert' in r.stdout


class TestPcliErrorPaths:
    """Server- and validator-side error responses surfaced through the CLI."""

    def test_ls_nonexistent_path(self, pcli: PcliRunner) -> None:
        r = pcli('ls', 'does_not_exist', check=False)
        assert r.returncode != 0
        assert 'does not exist' in r.stderr.lower() or 'not found' in r.stderr.lower()

    def test_describe_nonexistent(self, pcli: PcliRunner) -> None:
        r = pcli('describe', 'does_not_exist', check=False)
        assert r.returncode != 0

    def test_count_on_dir(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_err_dir', if_exists='ignore')
        r = pcli('count', 'pcli_err_dir', check=False)
        assert r.returncode != 0

    def test_rows_n_zero(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_err', if_exists='ignore')
        pxt.create_table('pcli_err.t0', {'a': pxt.Int}, if_exists='replace')
        r = pcli('rows', 'pcli_err/t0', '-n', '0', check=False)
        assert r.returncode != 0
        assert 'n must be > 0' in r.stderr

    def test_rows_unknown_col(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_err', if_exists='ignore')
        pxt.create_table('pcli_err.tcol', {'a': pxt.Int}, if_exists='replace')
        r = pcli('rows', 'pcli_err/tcol', '--cols', 'nope', check=False)
        assert r.returncode != 0
        assert 'unknown columns' in r.stderr

    def test_get_unknown_col(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_err', if_exists='ignore')
        t = pxt.create_table('pcli_err.gcol', {'k': pxt.Required[pxt.Int]}, primary_key='k', if_exists='replace')
        t.insert([{'k': 1}])
        r = pcli('get', 'pcli_err/gcol', '1', '--cols', 'missing', check=False)
        assert r.returncode != 0
        assert 'unknown columns' in r.stderr

    def test_get_pk_count_mismatch(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_err', if_exists='ignore')
        pxt.create_table('pcli_err.gpk', {'k': pxt.Required[pxt.Int]}, primary_key='k', if_exists='replace')
        r = pcli('get', 'pcli_err/gpk', '1', '2', check=False)
        assert r.returncode != 0
        assert 'expected 1 PK' in r.stderr

    def test_revert_steps_too_large(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_err', if_exists='ignore')
        pxt.create_table('pcli_err.rvbig', {'a': pxt.Int}, if_exists='replace')
        r = pcli('revert', 'pcli_err/rvbig', '--steps', '999', '-f', check=False)
        assert r.returncode != 0
        assert 'cannot revert' in r.stderr

    def test_revert_steps_invalid(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_err', if_exists='ignore')
        pxt.create_table('pcli_err.rvneg', {'a': pxt.Int}, if_exists='replace')
        r = pcli('revert', 'pcli_err/rvneg', '--steps', '0', '-f', check=False)
        assert r.returncode != 0
        assert '--steps must be >= 1' in r.stderr

    def test_path_rejects_dot_separator(self, pcli: PcliRunner) -> None:
        r = pcli('describe', 'a.b', check=False)
        assert r.returncode != 0
        assert "'/'" in r.stderr or 'separator' in r.stderr

    def test_path_rejects_leading_slash(self, pcli: PcliRunner) -> None:
        r = pcli('describe', '/x', check=False)
        assert r.returncode != 0
        assert 'relative' in r.stderr

    def test_path_rejects_trailing_slash(self, pcli: PcliRunner) -> None:
        r = pcli('describe', 'x/', check=False)
        assert r.returncode != 0
        assert "must not end with '/'" in r.stderr

    def test_path_rejects_double_slash(self, pcli: PcliRunner) -> None:
        r = pcli('describe', 'a//b', check=False)
        assert r.returncode != 0
        assert 'empty components' in r.stderr

    def test_errors_unknown_col(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_err', if_exists='ignore')
        pxt.create_table('pcli_err.errcol', {'k': pxt.Required[pxt.Int]}, primary_key='k', if_exists='replace')
        r = pcli('errors', 'pcli_err/errcol', '--col', 'nope', check=False)
        assert r.returncode != 0
        assert 'unknown column' in r.stderr

    def test_rows_cols_trailing_comma(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_err', if_exists='ignore')
        pxt.create_table('pcli_err.tc1', {'a': pxt.Int}, if_exists='replace')
        r = pcli('rows', 'pcli_err/tc1', '--cols', 'a,', check=False)
        assert r.returncode != 0
        assert 'must not be empty' in r.stderr

    def test_rows_cols_leading_comma(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_err', if_exists='ignore')
        pxt.create_table('pcli_err.tc2', {'a': pxt.Int}, if_exists='replace')
        r = pcli('rows', 'pcli_err/tc2', '--cols', ',a', check=False)
        assert r.returncode != 0
        assert 'must not be empty' in r.stderr

    def test_rows_cols_double_comma(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_err', if_exists='ignore')
        pxt.create_table('pcli_err.tc3', {'a': pxt.Int, 'b': pxt.Int}, if_exists='replace')
        r = pcli('rows', 'pcli_err/tc3', '--cols', 'a,,b', check=False)
        assert r.returncode != 0
        assert 'must not be empty' in r.stderr

    def test_get_pk_empty_rejected(self, pcli: PcliRunner) -> None:
        """An empty PK token almost certainly indicates a typo; reject rather than silently
        returning 'no row found' for PK=''."""
        pxt.create_dir('pcli_err', if_exists='ignore')
        pxt.create_table('pcli_err.gpkempty', {'k': pxt.Required[pxt.Int]}, primary_key='k', if_exists='replace')
        r = pcli('get', 'pcli_err/gpkempty', '', check=False)
        assert r.returncode != 0
        assert 'empty or whitespace' in r.stderr

    def test_get_pk_whitespace_rejected(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_err', if_exists='ignore')
        pxt.create_table('pcli_err.gpkws', {'k': pxt.Required[pxt.Int]}, primary_key='k', if_exists='replace')
        r = pcli('get', 'pcli_err/gpkws', '   ', check=False)
        assert r.returncode != 0
        assert 'empty or whitespace' in r.stderr

    def test_get_composite_pk_one_empty_rejected(self, pcli: PcliRunner) -> None:
        """Only one slot empty is enough to reject; the user gets a clear error before
        a partial composite lookup runs against the server."""
        pxt.create_dir('pcli_err', if_exists='ignore')
        pxt.create_table(
            'pcli_err.gpkc',
            {'a': pxt.Required[pxt.Int], 'b': pxt.Required[pxt.String]},
            primary_key=['a', 'b'],
            if_exists='replace',
        )
        r = pcli('get', 'pcli_err/gpkc', '1', '', check=False)
        assert r.returncode != 0
        assert 'empty or whitespace' in r.stderr

    def test_get_cols_trailing_comma(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_err', if_exists='ignore')
        pxt.create_table('pcli_err.gc', {'k': pxt.Required[pxt.Int]}, primary_key='k', if_exists='replace')
        r = pcli('get', 'pcli_err/gc', '1', '--cols', 'k,', check=False)
        assert r.returncode != 0
        assert 'must not be empty' in r.stderr

    def test_errors_col_not_stored_computed(self, pcli: PcliRunner) -> None:
        pxt.create_dir('pcli_err', if_exists='ignore')
        pxt.create_table(
            'pcli_err.errcol2', {'k': pxt.Required[pxt.Int], 'plain': pxt.String}, primary_key='k', if_exists='replace'
        )
        r = pcli('errors', 'pcli_err/errcol2', '--col', 'plain', check=False)
        assert r.returncode != 0
        assert 'stored computed column' in r.stderr

    def test_unknown_command(self, pcli: PcliRunner) -> None:
        r = pcli('not_a_command', check=False)
        assert r.returncode == 2
        assert 'unknown command' in r.stderr

    def test_top_level_help(self, pcli: PcliRunner) -> None:
        r = pcli('--help')
        assert r.returncode == 0
        # one line per command
        assert all(name in r.stdout for name in ('health', 'ls', 'shell'))

    def test_no_args(self, pcli: PcliRunner) -> None:
        r = pcli(check=False)
        assert r.returncode == 2
        assert 'usage' in r.stdout.lower() or 'usage' in r.stderr.lower()

    def test_subcommand_missing_required_arg(self, pcli: PcliRunner) -> None:
        # 'rows' requires a path positional; argparse should reject and print epilog
        r = pcli('rows', check=False)
        assert r.returncode == 2
        assert 'usage' in r.stderr.lower()
        # epilog (examples block) is appended on error
        assert 'Examples' in r.stderr

    def test_drop_dir_via_rm_only(self, pcli: PcliRunner) -> None:
        # `pcli drop` on a directory is refused by the server (drop is for tables/views).
        pxt.create_dir('pcli_err_dir2', if_exists='ignore')
        r = pcli('drop', 'pcli_err_dir2', '-f', check=False)
        assert r.returncode != 0

    def test_dashboard_health_endpoint(self, pcli_daemon: int) -> None:
        # Legacy dashboard probe — not reachable through the CLI; hit it directly.
        import urllib.request

        with urllib.request.urlopen(f'http://127.0.0.1:{pcli_daemon}/api/pixeltable-health') as r:
            body = json.loads(r.read())
        assert body == {'status': 'ok'}

    def test_revert_server_rejects_invalid_steps(self, pcli_daemon: int) -> None:
        """Client preflight catches steps<1, but the server has its own check that fires
        if anything bypasses the CLI (e.g. a future programmatic caller)."""
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            f'http://127.0.0.1:{pcli_daemon}/pcli/v0/revert',
            data=json.dumps({'path': 'whatever', 'steps': 0}).encode(),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(req)
        assert ei.value.code == 400
        body = json.loads(ei.value.read())
        assert 'steps must be >= 1' in body['detail']

    def test_get_with_valid_cols(self, pcli: PcliRunner) -> None:
        """Exercises the --cols branch of /pcli/v0/get with column names that exist."""
        pxt.create_dir('pcli_text', if_exists='ignore')
        t = pxt.create_table(
            'pcli_text.gcols',
            {'k': pxt.Required[pxt.Int], 'a': pxt.Int, 'b': pxt.String},
            primary_key='k',
            if_exists='replace',
        )
        t.insert([{'k': 1, 'a': 100, 'b': 'hello'}])
        out = pcli('get', 'pcli_text/gcols', '1', '--cols', 'a', '--json').json
        assert out['row'] == {'a': 100}

    def test_rows_and_get_with_image_column(self, pcli: PcliRunner) -> None:
        """Image cells must render as `<Image WxH MODE>` rather than dumping raw bytes
        or blowing up. Exercises both routes.rows() and routes.get() PIL branches."""
        from tests.utils import get_image_files

        img_paths = get_image_files()[:2]
        pxt.create_dir('pcli_img', if_exists='ignore')
        t = pxt.create_table(
            'pcli_img.t',
            {'k': pxt.Required[pxt.Int], 'img': pxt.Image, 'tag': pxt.String},
            primary_key='k',
            if_exists='replace',
        )
        t.insert([{'k': i, 'img': p, 'tag': f't{i}'} for i, p in enumerate(img_paths)])

        # rows --json: image cell is a placeholder string, not base64 / not omitted
        rows_json = pcli('rows', 'pcli_img/t', '-n', '5', '--json').json
        assert len(rows_json) == 2
        assert all(r['img'].startswith('<Image ') and r['img'].endswith('>') for r in rows_json)
        assert {r['tag'] for r in rows_json} == {'t0', 't1'}

        # rows plain text: tab-separated, image placeholder appears unmodified
        rows_text = pcli('rows', 'pcli_img/t').stdout
        assert '<Image ' in rows_text
        # no PIL repr or raw bytes leaked
        assert 'PIL.' not in rows_text
        assert '\\x' not in rows_text

        # get text: image cell rendered the same way
        get_text = pcli('get', 'pcli_img/t', '0').stdout
        assert 'img\t<Image ' in get_text
        assert 'tag\tt0' in get_text

        # get --json: same placeholder, valid JSON
        get_json = pcli('get', 'pcli_img/t', '1', '--json').json
        assert get_json['row']['img'].startswith('<Image ')
        assert get_json['row']['tag'] == 't1'

    def test_ls_counts_dirs_only(self, pcli: PcliRunner) -> None:
        """--counts on a directory whose children are all sub-directories: _fill_counts
        sees an empty targets list and short-circuits."""
        pxt.create_dir('pcli_dir_only', if_exists='ignore')
        pxt.create_dir('pcli_dir_only.sub1', if_exists='ignore')
        pxt.create_dir('pcli_dir_only.sub2', if_exists='ignore')
        r = pcli('ls', 'pcli_dir_only', '--counts', '--json')
        entries = r.json['entries']
        assert all(e['kind'] == 'dir' for e in entries)
        assert all('num_rows' not in e or e['num_rows'] is None for e in entries)
