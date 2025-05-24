import pixeltable as pxt
from pixeltable.catalog import Catalog
from pixeltable.catalog.globals import IfExistsParam
from pixeltable.catalog.path import Path
from tests.utils import reload_catalog


class TestReplica:
    def test_replica(self, test_tbl: pxt.Table) -> None:
        """
        Isolated test for the replica creation functionality.
        """
        pure_snapshot = pxt.create_snapshot('pure_snapshot', test_tbl)
        snapshot_view = pxt.create_snapshot('snapshot_view', test_tbl, additional_columns={'extra': pxt.Int})

        with Catalog.get().begin_xact(for_write=False):
            md1 = Catalog.get().load_replica_md(pure_snapshot)
            md2 = Catalog.get().load_replica_md(snapshot_view)

        assert len(md1) == 2
        assert len(md2) == 2

        for i, md in enumerate(md1):
            print(f'\n{i}: {md}')

        pxt.drop_table(test_tbl, force=True)
        reload_catalog()

        Catalog.get().create_replica(Path('replica_1'), md1, if_exists=IfExistsParam.ERROR)
        Catalog.get().create_replica(Path('replica_2'), md2, if_exists=IfExistsParam.ERROR)
        reload_catalog()

        t1 = pxt.get_table('replica_1')
        t2 = pxt.get_table('replica_2')
        assert len(t1._tbl_version_path.get_tbl_versions()) == 1
        assert len(t2._tbl_version_path.get_tbl_versions()) == 2

        assert t1.get_metadata()['is_snapshot']
        assert t1.get_metadata()['is_replica']
        assert t2.get_metadata()['is_snapshot']
        assert t2.get_metadata()['is_replica']

    def test_complex_replica(self, reset_db: None) -> None:
        """
        This test involves various more complicated arrangements of tables and snapshots.

        base_tbl > v1, v2
        v1 > s11, s12
        v2 > v3 > s31
        s11 > v4 > v5 > s51 > v6 > s61
        """
        t = pxt.create_table('base_tbl', {'c1': pxt.Int})  # Base table
        t.insert({'c1': i} for i in range(10))
        assert t._tbl_version.get().version == 1
        v1 = pxt.create_view('v1', t, additional_columns={'c2': pxt.Int})
        v1.update({'c2': v1.c1 * 10})
        t.insert({'c1': i} for i in range(10, 20))
        assert t._tbl_version.get().version == 2
        assert v1._tbl_version.get().version == 2
        s11 = pxt.create_snapshot('s11', v1)
        v1.update({'c2': v1.c1 * 10})
        assert t._tbl_version.get().version == 2
        assert v1._tbl_version.get().version == 3
        s12 = pxt.create_snapshot('s12', v1)

        v2 = pxt.create_view('v2', t, additional_columns={'c2': pxt.String})
        v2.update({'c2': 'xyz'})
        v3 = pxt.create_view('v3', v2, additional_columns={'c3': pxt.Int})
        v3.update({'c3': v3.c1 * 100})
        t.insert({'c1': i} for i in range(20, 30))
        assert t._tbl_version.get().version == 3
        assert v2._tbl_version.get().version == 2
        assert v3._tbl_version.get().version == 2
        s31 = pxt.create_snapshot('s31', v3, additional_columns={'c31': pxt.Int})

        v4 = pxt.create_view('v4', s11, additional_columns={'c4': pxt.Float})
        v5 = pxt.create_view('v5', v4, additional_columns={'c5': pxt.Bool})

        s51 = pxt.create_snapshot('s51', v5, additional_columns={'c51': pxt.Json})

        v6 = pxt.create_view('v6', s51, additional_columns={'c6': pxt.Json})
        s61 = pxt.create_snapshot('s61', v6)

        with Catalog.get().begin_xact(for_write=False):
            s11_md = Catalog.get().load_replica_md(s11)
            s12_md = Catalog.get().load_replica_md(s12)
            s31_md = Catalog.get().load_replica_md(s31)
            s51_md = Catalog.get().load_replica_md(s51)
            s61_md = Catalog.get().load_replica_md(s61)

        pxt.drop_table('base_tbl', force=True)
        reload_catalog()

        for i, md in enumerate(s11_md):
            print(f'\n{i}: {md}')
        _r11 = Catalog.get().create_replica(Path('replica_s11'), s11_md, if_exists=IfExistsParam.ERROR)
        _r12 = Catalog.get().create_replica(Path('replica_s12'), s12_md, if_exists=IfExistsParam.ERROR)
        _r31 = Catalog.get().create_replica(Path('replica_s31'), s31_md, if_exists=IfExistsParam.ERROR)

        # Intentionally create r61 first, before r51; this way we address both cases for snapshot-over-snapshot:
        # Base snapshot inserted first (r61 after r31); base snapshot inserted last (r51 after r61).
        Catalog.get().create_replica(Path('replica_s61'), s61_md, if_exists=IfExistsParam.ERROR)
        r61 = pxt.get_table('replica_s61')
        Catalog.get().create_replica(Path('replica_s51'), s51_md, if_exists=IfExistsParam.ERROR)
        r51 = pxt.get_table('replica_s51')

        with Catalog.get().begin_xact(for_write=False):
            assert len(r51._base_tables) == 4
            assert len(r61._base_tables) == 6
