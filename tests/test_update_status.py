from dataclasses import asdict

import pytest
from IPython.lib.pretty import pretty

import pixeltable as pxt
from pixeltable.catalog import RowCountStats, UpdateStatus
from pixeltable.metadata.schema import VersionMd, md_from_dict, md_to_dict

from .utils import DatabaseRoot


def test_operation_row_counts(db_root: DatabaseRoot) -> None:
    p = db_root.make_catalog_path
    table = pxt.create_table(p('counts'), {'n': pxt.Int})
    view = pxt.create_view(p('counts_view'), table)
    view.add_computed_column(doubled=view.n * 2)
    inserted = table.insert([{'n': 1}, {'n': 2}])
    assert inserted.row_count_stats == RowCountStats(inserted_rows=2)
    assert inserted.cascade_row_count_stats == RowCountStats(inserted_rows=2)
    updated = table.where(table.n == 1).update({'n': 3})
    assert updated.row_count_stats == RowCountStats(updated_rows=1)
    assert updated.cascade_row_count_stats == RowCountStats(updated_rows=1)
    deleted = table.delete(table.n == 3)
    assert deleted.row_count_stats == RowCountStats(deleted_rows=1)
    assert deleted.cascade_row_count_stats == RowCountStats(deleted_rows=1)


def test_row_counts() -> None:
    status = UpdateStatus(
        row_count_stats=RowCountStats(inserted_rows=2, updated_rows=3, deleted_rows=4, num_excs=1),
        cascade_row_count_stats=RowCountStats(inserted_rows=5, updated_rows=6, deleted_rows=7, num_excs=2),
    )
    assert status.num_rows == 27
    assert status.num_excs == 3
    assert status.pxt_rows_updated == 9
    assert pretty(status) == '7 rows inserted, 11 rows deleted, 9 rows updated, 3 exceptions.'
    converted = status.insert_to_update()
    assert converted.row_count_stats == RowCountStats(updated_rows=5, deleted_rows=4, num_excs=1)
    assert converted.cascade_row_count_stats == RowCountStats(updated_rows=11, deleted_rows=7, num_excs=2)


@pytest.mark.parametrize('legacy_names', [False, True])
def test_history_row_counts(legacy_names: bool) -> None:
    stats = RowCountStats(inserted_rows=2, updated_rows=3, deleted_rows=4, num_excs=1)
    version = VersionMd(
        tbl_id='test',
        created_at=0,
        version=1,
        schema_version=1,
        update_status=UpdateStatus(row_count_stats=stats, cascade_row_count_stats=stats),
    )
    md = asdict(version)
    if legacy_names:
        for field in ('row_count_stats', 'cascade_row_count_stats'):
            md['update_status'][field] = {'ins_rows': 2, 'upd_rows': 3, 'del_rows': 4, 'num_excs': 1}
    restored = md_from_dict(VersionMd, md)
    assert restored == version
    assert md_to_dict(restored) == asdict(version)
