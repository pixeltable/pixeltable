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
        test_snapshot = pxt.create_snapshot('test_snapshot', test_tbl)

        with Env.get().begin_xact():
            md = Catalog.get().load_tbl_ancestors_md(test_snapshot._tbl_version_path)

        pxt.drop_table(test_tbl, force=True)
        reload_catalog()

        Catalog.get().create_replica(Path('test_replica'), md, if_exists=IfExistsParam.ERROR)
        reload_catalog()

        t = pxt.get_table('test_replica')
