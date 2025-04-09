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
