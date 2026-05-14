"""Smoke tests: drive state via the pxt API, validate via pcli."""

import pixeltable as pxt

from .conftest import PcliRunner


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
        assert out['pxt_version']
        assert out['pid'] > 0

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
