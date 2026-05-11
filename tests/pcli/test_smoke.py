"""Smoke tests: drive state via the pxt API, validate via pcli."""

import pixeltable as pxt


class TestPcliSmoke:
    def test_health(self, pcli) -> None:
        out = pcli('health').json
        assert out['ok'] is True
        assert out['pid'] > 0

    def test_ls_reflects_api_state(self, pcli) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        pxt.create_table('pcli_smoke.t', {'x': pxt.Int, 'y': pxt.String}, if_exists='ignore')

        entries = pcli('ls', 'pcli_smoke', '--json').json['entries']
        paths = {e['path'] for e in entries}
        assert 'pcli_smoke/t' in paths

        pxt.drop_table('pcli_smoke.t')
        entries = pcli('ls', 'pcli_smoke', '--json').json['entries']
        paths = {e['path'] for e in entries}
        assert 'pcli_smoke/t' not in paths

    def test_describe(self, pcli) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        pxt.create_table('pcli_smoke.describe_me', {'a': pxt.Int}, if_exists='ignore')

        out = pcli('describe', 'pcli_smoke/describe_me', '--json').json
        assert 'a' in out['columns']

    def test_rows(self, pcli) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        t = pxt.create_table('pcli_smoke.rows', {'n': pxt.Int, 's': pxt.String}, if_exists='replace')
        t.insert([{'n': i, 's': f'row{i}'} for i in range(5)])

        out = pcli('rows', 'pcli_smoke/rows', '-n', '3', '--json').json
        assert len(out) == 3

        out2 = pcli('rows', 'pcli_smoke/rows', '-n', '10', '--cols', 'n', '--json').json
        assert sorted(r['n'] for r in out2) == [0, 1, 2, 3, 4]

    def test_columns(self, pcli) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        pxt.create_table('pcli_smoke.cols', {'a': pxt.Int, 'b': pxt.String}, if_exists='replace')

        entries = pcli('columns', 'pcli_smoke/cols', '--json').json
        names = {e['column'] for e in entries}
        assert names == {'a', 'b'}

    def test_idxs_no_embedding(self, pcli) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        pxt.create_table('pcli_smoke.no_idx', {'a': pxt.Int}, if_exists='replace')

        entries = pcli('idxs', 'pcli_smoke/no_idx', '--json').json
        assert all(e['index_type'] != 'embedding' for e in entries)

    def test_history(self, pcli) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        t = pxt.create_table('pcli_smoke.hist', {'a': pxt.Int}, if_exists='replace')
        t.insert([{'a': 1}])

        out = pcli('history', 'pcli_smoke/hist', '--json').json
        assert len(out) >= 2

    def test_count(self, pcli) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        t = pxt.create_table('pcli_smoke.cnt', {'a': pxt.Int}, if_exists='replace')
        t.insert([{'a': i} for i in range(7)])

        out = pcli('count', 'pcli_smoke/cnt', '--json').json
        assert out['count'] == 7

        # plain output is just the integer
        plain = pcli('count', 'pcli_smoke/cnt').stdout.strip()
        assert plain == '7'

    def test_get_by_pk(self, pcli) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        t = pxt.create_table('pcli_smoke.pk_tbl', {'k': pxt.Required[pxt.Int], 'v': pxt.String},
                             primary_key='k', if_exists='replace')
        t.insert([{'k': 1, 'v': 'one'}, {'k': 2, 'v': 'two'}])

        out = pcli('get', 'pcli_smoke/pk_tbl', '2', '--json').json
        assert out['pk_columns'] == ['k']
        assert out['row']['v'] == 'two'

        # missing row
        out = pcli('get', 'pcli_smoke/pk_tbl', '99', '--json').json
        assert out['row'] is None

    def test_get_requires_pk(self, pcli) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        pxt.create_table('pcli_smoke.no_pk_get', {'a': pxt.Int}, if_exists='replace')

        r = pcli('get', 'pcli_smoke/no_pk_get', '1', check=False)
        assert r.returncode != 0
        assert 'primary key' in r.stderr.lower()

    def test_status(self, pcli) -> None:
        out = pcli('status', '--json').json
        assert out['pxt_version']
        assert out['pid'] > 0

    def test_env(self, pcli) -> None:
        out = pcli('env', '--json').json
        # PIXELTABLE_DB is always set by init_env
        assert 'PIXELTABLE_DB' in out['env_vars']

    def test_errors_requires_pk(self, pcli) -> None:
        pxt.create_dir('pcli_smoke', if_exists='ignore')
        pxt.create_table('pcli_smoke.no_pk', {'a': pxt.Int}, if_exists='replace')

        r = pcli('errors', 'pcli_smoke/no_pk', check=False)
        assert r.returncode != 0
        assert 'primary key' in r.stderr.lower()
