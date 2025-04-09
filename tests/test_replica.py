import pixeltable as pxt
from pixeltable.catalog import Catalog
from pixeltable.catalog.globals import IfExistsParam
from pixeltable.catalog.path import Path
from pixeltable.env import Env
from tests.utils import reload_catalog


class TestReplica:
    def test_replica(self, test_tbl: pxt.Table) -> None:
        """
        Isolated test for the replica creation functionality.
        """
        pure_snapshot = pxt.create_snapshot('pure_snapshot', test_tbl)
        snapshot_view = pxt.create_snapshot('snapshot_view', test_tbl, additional_columns={'extra': pxt.Int})

        with Env.get().begin_xact():
            md1 = Catalog.get().load_tbl_hierarchy_md(pure_snapshot)
            md2 = Catalog.get().load_tbl_hierarchy_md(snapshot_view)

        assert len(md1) == 2
        assert len(md2) == 2

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

    def test_complex_replica(self, reset_db) -> None:
        """
        This test involves various more complicated arrangements of tables and snapshots.
        """
        t = pxt.create_table('base_tbl', {'c1': pxt.Int})  # Base table
        t.insert({'c1': i} for i in range(10))
        assert t._tbl_version.get().version == 1
        v1 = pxt.create_view('v1', t, additional_columns={'c2': pxt.Int})
        v1.update({'c2': v1.c1 * 10})
        t.insert({'c1': i} for i in range(10, 20))
        assert t._tbl_version.get().version == 2
        assert v1._tbl_version.get().version == 2
        s1 = pxt.create_snapshot('s1', v1)
        v1.update({'c2': v1.c1 * 10})
        assert t._tbl_version.get().version == 2
        assert v1._tbl_version.get().version == 3
        s2 = pxt.create_snapshot('s2', v1)

        v2 = pxt.create_view('v2', t, additional_columns={'c2': pxt.String})
        v2.update({'c2': 'xyz'})
        v3 = pxt.create_view('v3', v2, additional_columns={'c3': pxt.Int})
        v3.update({'c3': v3.c1 * 100})
        t.insert({'c1': i} for i in range(20, 30))
        assert t._tbl_version.get().version == 3
        assert v2._tbl_version.get().version == 2
        assert v3._tbl_version.get().version == 2
        s3 = pxt.create_snapshot('s3', v3)

        with Env.get().begin_xact():
            s1_md = Catalog.get().load_tbl_hierarchy_md(s1)
            s2_md = Catalog.get().load_tbl_hierarchy_md(s2)
            s3_md = Catalog.get().load_tbl_hierarchy_md(s3)

        pxt.drop_table('base_tbl', force=True)
        reload_catalog()

        for i, md in enumerate(s1_md):
            print(f'\n{i}: {md}')
        r1 = Catalog.get().create_replica(Path('replica_s1'), s1_md, if_exists=IfExistsParam.ERROR)
        r2 = Catalog.get().create_replica(Path('replica_s2'), s2_md, if_exists=IfExistsParam.ERROR)
        r3 = Catalog.get().create_replica(Path('replica_s3'), s3_md, if_exists=IfExistsParam.ERROR)
