# ruff: noqa: F821
# ruff: noqa: N806
# ruff: noqa: E731
# ruff: noqa: RUF012

from __future__ import annotations

import threading
from typing import Any, Callable

import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.catalog.model import EmbeddingIndex
from pixeltable.utils.fault_injection import FaultLocation

from .coordinator import MultiThreadedScenario
from .fault_injection import BlockFault
from .utils import dummy_embedding, pxt_raises

ROOT = ''


def schema_changed(tbl_name: str) -> str:
    """The drift error update_all() raises when tbl_name changed since the diff."""
    return rf"Table '{tbl_name}' saw schema changes since update_all\(\) computed its changes"


# a dropped table and one dropped and recreated at the same path are indistinguishable by id: both are gone
TABLE_GONE = r"Table 'test_table' was dropped or replaced since update_all\(\) computed its changes"


def _run_with_concurrent_apply(
    update_all_fn: Callable[[], object], concurrent_fn: Callable[[], object], *, timeout: float = 60.0
) -> None:
    """Run update_all_fn() on thread 0; it blocks just before applying its diff. Run concurrent_fn() to completion on
    thread 1, unblock thread 0, and let its apply proceed against the drifted catalog. Re-raises the first exception
    from either thread."""
    block = BlockFault()
    (
        MultiThreadedScenario()
        .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_UPDATE_FROM_MODEL_BEFORE_APPLY, fault=block)
        .then_run_until(thread_id=0, name='update_all (under test)', event=block.reached, fn=update_all_fn)
        .then_run(thread_id=1, name='concurrent change', fn=concurrent_fn)
        .then_unblock(thread_id=1, fault=block)
        .execute(timeout=timeout)
    )


def _run_workers(target: Callable[[int], None], n_threads: int) -> list[tuple[int, BaseException]]:
    """Run target(tid) on n_threads workers; return any exceptions raised."""
    errors: list[tuple[int, BaseException]] = []
    lock = threading.Lock()

    def runner(tid: int) -> None:
        try:
            target(tid)
        except BaseException as e:
            with lock:
                errors.append((tid, e))

    threads = [threading.Thread(target=runner, args=(i,)) for i in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return errors


@pytest.mark.local('fault-injection/concurrency test of update_all() vs concurrent schema change')
class TestConcurrentModelUpdate:
    # ---------------------------------------------------------------------------------------------------------------
    # Group A: thread 0 ADDS a column/index while a concurrent change drifts the catalog.
    # ---------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize('family', ['update_all', 'single_op'])
    def test_add_column_name_collision(self, uses_db: None, fault_injection: None, family: str) -> None:
        """A1: thread 0 adds column x; a concurrent change adds a column also named x."""
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        TM.create_all(ROOT)
        Base.insert([{'id': 1, 'value': 1.0}, {'id': 2, 'value': 2.0}])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            x = value + 1

        if family == 'update_all':
            TMc = pxt.model_base()

            class BaseC(TMc, name='test_table'):
                id: pxt.Int
                value: pxt.Float | None
                x = value + 100

            concurrent = lambda: TMc.update_all(ROOT)
        else:
            t = Base.table
            concurrent = lambda: t.add_computed_column(x=t.value + 100)

        with pxt_raises(excs.ErrorCode.CONCURRENT_MODIFICATION, match=schema_changed('test_table')):
            _run_with_concurrent_apply(lambda: TM2.update_all(ROOT), concurrent)

    def test_partial_apply_names_the_migrations(self, uses_db: None, fault_injection: None) -> None:
        """A create that loses a race reports the migrations that already committed, not just its own failure."""
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int

        TM.create_all(ROOT)

        # migrate test_table and create new_table in one update_all()
        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        class NewTable(TM2, name='new_table'):
            id: pxt.Int

        # someone else creates new_table while the migration is in flight, so only the create fails
        concurrent = lambda: pxt.create_table(f'{ROOT}new_table', {'id': pxt.Int})

        with pxt_raises(excs.ErrorCode.CONCURRENT_MODIFICATION, match=r"already migrated: 'test_table'"):
            _run_with_concurrent_apply(lambda: TM2.update_all(ROOT), concurrent)

        # the migration is committed, so the error must not read as though nothing happened
        assert 'value' in pxt.get_table(f'{ROOT}test_table').columns()

    @pytest.mark.parametrize('family', ['update_all', 'single_op'])
    def test_add_index_name_collision(self, uses_db: None, fault_injection: None, family: str) -> None:
        """A2: thread 0 adds index ix; a concurrent change adds an index also named ix."""
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            text: pxt.String | None

        TM.create_all(ROOT)
        Base.insert([{'id': 1, 'text': 'one'}, {'id': 2, 'text': 'two'}])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            text: pxt.String | None

            __indexes__ = [EmbeddingIndex(text, embedding=dummy_embedding.using(n=768))]

        if family == 'update_all':
            TMc = pxt.model_base()

            class BaseC(TMc, name='test_table'):
                id: pxt.Int
                text: pxt.String | None

                __indexes__ = [EmbeddingIndex(text, embedding=dummy_embedding.using(n=512))]

            concurrent = lambda: TMc.update_all(ROOT)
        else:
            t = Base.table
            concurrent = lambda: t.add_embedding_index('text', idx_name='ix', embedding=dummy_embedding.using(n=512))

        with pxt_raises(excs.ErrorCode.CONCURRENT_MODIFICATION, match=schema_changed('test_table')):
            _run_with_concurrent_apply(lambda: TM2.update_all(ROOT), concurrent)

    @pytest.mark.parametrize('family', ['update_all', 'single_op'])
    def test_view_add_shadows_base_add(self, uses_db: None, fault_injection: None, family: str) -> None:
        """A3: thread 0 adds a view column x; a concurrent change adds a column named x to the BASE.

        A view column shadows a base column of the same name, so the end state here is well-defined and is what
        running the two changes in sequence produces. The update aborts anyway: the base's schema version is part of
        what the view's diff was resolved against, and a version match is the only thing separating this from a base
        change that would invalidate the diff.
        """
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        class View(TM, name='test_view', base=Base):
            vc1 = Base.value + 1

        TM.create_all(ROOT)
        Base.insert([{'id': 1, 'value': 1.0}, {'id': 2, 'value': 2.0}])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        class ViewV2(TM2, name='test_view', base=BaseV2):
            vc1 = BaseV2.value + 1
            x = BaseV2.value + 5

        if family == 'update_all':
            TMc = pxt.model_base()

            class BaseC(TMc, name='test_table'):
                id: pxt.Int
                value: pxt.Float | None
                x = value + 100

            concurrent = lambda: TMc.update_all(ROOT)
        else:
            bt = Base.table
            concurrent = lambda: bt.add_computed_column(x=bt.value + 100)

        with pxt_raises(excs.ErrorCode.CONCURRENT_MODIFICATION, match=schema_changed('test_view')):
            _run_with_concurrent_apply(lambda: TM2.update_all(ROOT), concurrent)

    @pytest.mark.parametrize('family', ['update_all', 'single_op'])
    def test_add_col_on_dropped_col(self, uses_db: None, fault_injection: None, family: str) -> None:
        """A4a: thread 0 adds a computed column referencing extra; a concurrent change drops extra."""
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            extra: pxt.Float | None

        TM.create_all(ROOT)
        Base.insert([{'id': 1, 'value': 1.0, 'extra': 10.0}, {'id': 2, 'value': 2.0, 'extra': 20.0}])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            extra: pxt.Float | None
            x = extra + 1

        if family == 'update_all':
            TMc = pxt.model_base()

            class BaseC(TMc, name='test_table'):
                id: pxt.Int
                value: pxt.Float | None

            concurrent = lambda: TMc.update_all(ROOT, allow_destructive=True)
        else:
            t = Base.table
            concurrent = lambda: t.drop_column('extra')

        with pxt_raises(excs.ErrorCode.CONCURRENT_MODIFICATION, match=schema_changed('test_table')):
            _run_with_concurrent_apply(lambda: TM2.update_all(ROOT), concurrent)

    @pytest.mark.parametrize('retyped_as', [pxt.String | None, pxt.Int | None], ids=['incompatible', 'compatible'])
    def test_add_view_col_on_retyped_base_col(self, uses_db: None, fault_injection: None, retyped_as: Any) -> None:
        """
        Scenario:
        - thread 0 adds a view column computed from a base column
        - thread 1 drops that base column and re-adds it under the same name with a different type.

        Only the base's schema version moves, so the view's own version cannot detect this. vc2 resolves its
        reference to value by name at apply time and binds to the replacement column: a physically different column
        than the one the diff was computed against, holding no data.
        """
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        class View(TM, name='test_view', base=Base):
            vc1 = Base.id + 1

        TM.create_all(ROOT)
        Base.insert([{'id': 1, 'value': 1.0}, {'id': 2, 'value': 2.0}])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        class ViewV2(TM2, name='test_view', base=BaseV2):
            vc1 = BaseV2.id + 1
            vc2 = BaseV2.value + 1.0

        t = Base.table

        def retype_base_col() -> None:
            t.drop_column('value')
            t.add_column(value=retyped_as)

        with pxt_raises(excs.ErrorCode.CONCURRENT_MODIFICATION, match=schema_changed('test_view')):
            _run_with_concurrent_apply(lambda: TM2.update_all(ROOT), retype_base_col)
        assert 'vc2' not in pxt.get_table('test_view').columns()

    @pytest.mark.parametrize('family', ['update_all', 'single_op'])
    def test_add_index_on_dropped_col(self, uses_db: None, fault_injection: None, family: str) -> None:
        """A5: thread 0 adds an index on text; a concurrent change drops text."""
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            text: pxt.String | None

        TM.create_all(ROOT)
        Base.insert([{'id': 1, 'text': 'one'}, {'id': 2, 'text': 'two'}])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            text: pxt.String | None

            __indexes__ = [EmbeddingIndex(text, embedding=dummy_embedding.using(n=768))]

        if family == 'update_all':
            TMc = pxt.model_base()

            class BaseC(TMc, name='test_table'):
                id: pxt.Int

            concurrent = lambda: TMc.update_all(ROOT, allow_destructive=True)
        else:
            t = Base.table
            concurrent = lambda: t.drop_column('text')

        with pxt_raises(excs.ErrorCode.CONCURRENT_MODIFICATION, match=schema_changed('test_table')):
            _run_with_concurrent_apply(lambda: TM2.update_all(ROOT), concurrent)

    @pytest.mark.parametrize('family', ['update_all', 'single_op'])
    def test_conflicting_drop_column(self, uses_db: None, fault_injection: None, family: str) -> None:
        """B1: thread 0 drops column x; a concurrent change already dropped x."""
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            x: pxt.Float | None

        TM.create_all(ROOT)
        Base.insert([{'id': 1, 'value': 1.0, 'x': 1.0}, {'id': 2, 'value': 2.0, 'x': 2.0}])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        if family == 'update_all':
            TMc = pxt.model_base()

            class BaseC(TMc, name='test_table'):
                id: pxt.Int
                value: pxt.Float | None

            concurrent = lambda: TMc.update_all(ROOT, allow_destructive=True)
        else:
            t = Base.table
            concurrent = lambda: t.drop_column('x')

        with pxt_raises(excs.ErrorCode.CONCURRENT_MODIFICATION, match=schema_changed('test_table')):
            _run_with_concurrent_apply(lambda: TM2.update_all(ROOT, allow_destructive=True), concurrent)

    @pytest.mark.parametrize('family', ['update_all', 'single_op'])
    def test_conflicting_drop_index(self, uses_db: None, fault_injection: None, family: str) -> None:
        """B2: thread 0 drops index ix; a concurrent change already dropped ix."""
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            text: pxt.String | None

            __indexes__ = [EmbeddingIndex(text, embedding=dummy_embedding.using(n=768), name='ix')]

        TM.create_all(ROOT)
        Base.insert([{'id': 1, 'text': 'one'}, {'id': 2, 'text': 'two'}])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            text: pxt.String | None

        if family == 'update_all':
            TMc = pxt.model_base()

            class BaseC(TMc, name='test_table'):
                id: pxt.Int
                text: pxt.String | None

            concurrent = lambda: TMc.update_all(ROOT, allow_destructive=True)
        else:
            t = Base.table
            concurrent = lambda: t.drop_embedding_index(idx_name='ix')

        with pxt_raises(excs.ErrorCode.CONCURRENT_MODIFICATION, match=schema_changed('test_table')):
            _run_with_concurrent_apply(lambda: TM2.update_all(ROOT, allow_destructive=True), concurrent)

    def test_drop_renamed_column(self, uses_db: None, fault_injection: None) -> None:
        """B3 (single_op only): thread 0 drops column x; a concurrent change renames x -> y.

        The dropped name no longer exists, and skipping the drop would leave y as an undeclared column; the update
        must fail rather than silently leave the catalog inconsistent with the model.
        """
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            x: pxt.Float | None

        TM.create_all(ROOT)
        Base.insert([{'id': 1, 'value': 1.0, 'x': 1.0}, {'id': 2, 'value': 2.0, 'x': 2.0}])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        t = Base.table
        with pxt_raises(excs.ErrorCode.CONCURRENT_MODIFICATION, match=schema_changed('test_table')):
            _run_with_concurrent_apply(
                lambda: TM2.update_all(ROOT, allow_destructive=True), lambda: t.rename_column('x', 'y')
            )

    @pytest.mark.parametrize('family', ['update_all', 'single_op'])
    def test_drop_column_with_new_dependent(self, uses_db: None, fault_injection: None, family: str) -> None:
        """B4: thread 0 drops column x; a concurrent change adds a dependent that references x."""
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            x: pxt.Float | None

        class View(TM, name='test_view', base=Base):
            vc1 = Base.value + 1

        TM.create_all(ROOT)
        Base.insert([{'id': 1, 'value': 1.0, 'x': 1.0}, {'id': 2, 'value': 2.0, 'x': 2.0}])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        class ViewV2(TM2, name='test_view', base=BaseV2):
            vc1 = BaseV2.value + 1

        if family == 'update_all':
            TMc = pxt.model_base()

            class BaseCM(TMc, name='test_table'):
                id: pxt.Int
                value: pxt.Float | None
                x: pxt.Float | None

            class ViewCM(TMc, name='test_view', base=BaseCM):
                vc1 = BaseCM.value + 1
                dep = BaseCM.x + 1

            concurrent = lambda: TMc.update_all(ROOT)
        else:
            v = View.table
            concurrent = lambda: v.add_computed_column(dep=Base.table.x + 1)

        # Dropping x must be blocked while a live column depends on it. The dependent lives in the view, so the
        # base table's schema version is unchanged and the CAS doesn't fire; the dependents check does.
        with pxt_raises(
            excs.ErrorCode.UNSUPPORTED_OPERATION,
            match=r"Column 'x' was removed from the model for 'test_table', but cannot be dropped "
            r'because the following depend on it:\ndep',
        ):
            _run_with_concurrent_apply(lambda: TM2.update_all(ROOT, allow_destructive=True), concurrent)

    def test_drop_index_with_new_dependent(self, uses_db: None, fault_injection: None) -> None:
        """B5 (single_op only): thread 0 drops index ix; a concurrent change adds a view column that references it."""
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            text: pxt.String | None

            __indexes__ = [EmbeddingIndex(text, embedding=dummy_embedding.using(n=768))]

        class View(TM, name='test_view', base=Base):
            vc1 = Base.id + 1

        TM.create_all(ROOT)
        Base.insert([{'id': 1, 'text': 'one'}, {'id': 2, 'text': 'two'}])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            text: pxt.String | None

        class ViewV2(TM2, name='test_view', base=BaseV2):
            vc1 = BaseV2.id + 1

        v = View.table
        with pxt_raises(
            excs.ErrorCode.UNSUPPORTED_OPERATION,
            match=r"Index 'idx0' was removed from the model for 'test_table', but cannot be dropped "
            r'because the following depend on it:\ndep',
        ):
            _run_with_concurrent_apply(
                lambda: TM2.update_all(ROOT, allow_destructive=True),
                lambda: v.add_computed_column(dep=Base.table.text.embedding()),
            )

    # ---------------------------------------------------------------------------------------------------------------
    # Group C: table-level drift (single-op only).
    # ---------------------------------------------------------------------------------------------------------------

    def test_drop_table(self, uses_db: None, fault_injection: None) -> None:
        """C1: thread 0 updates a table; a concurrent change drops the table entirely."""
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        TM.create_all(ROOT)
        Base.insert([{'id': 1, 'value': 1.0}])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            x = value + 1

        with pxt_raises(excs.ErrorCode.CONCURRENT_MODIFICATION, match=TABLE_GONE):
            _run_with_concurrent_apply(lambda: TM2.update_all(ROOT), lambda: pxt.drop_table('test_table'))

    def test_drop_recreate_table(self, uses_db: None, fault_injection: None) -> None:
        """C2: thread 0 drops column x; a concurrent change drops and recreates the table (new tbl_id) without x."""
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            x: pxt.Float | None

        TM.create_all(ROOT)
        Base.insert([{'id': 1, 'value': 1.0, 'x': 1.0}])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        def recreate() -> None:
            pxt.drop_table('test_table')
            pxt.create_table('test_table', {'id': pxt.Int, 'value': pxt.Float | None})

        with pxt_raises(excs.ErrorCode.CONCURRENT_MODIFICATION, match=TABLE_GONE):
            _run_with_concurrent_apply(lambda: TM2.update_all(ROOT, allow_destructive=True), recreate)

    def test_move_base(self, uses_db: None, fault_injection: None) -> None:
        """C3: thread 0 updates a view; a concurrent change moves the view's base table to a new path."""
        pxt.create_dir('other')
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        class View(TM, name='test_view', base=Base):
            vc1 = Base.value + 1

        TM.create_all(ROOT)
        Base.insert([{'id': 1, 'value': 1.0}])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        class ViewV2(TM2, name='test_view', base=BaseV2):
            vc1 = BaseV2.value + 1
            vc2 = BaseV2.value + 2

        # Moving the base doesn't change the view's own schema, so the view update should still apply coherently.
        _run_with_concurrent_apply(lambda: TM2.update_all(ROOT), lambda: pxt.move('test_table', 'other.test_table'))
        assert 'vc2' in ViewV2.get_metadata()['columns']

    def test_create_table_conflict(self, uses_db: None, fault_injection: None) -> None:
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        TM.create_all(ROOT)
        Base.insert([{'id': 1, 'value': 1.0}])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            x = value + 1

        class NewTable(TM2, name='new_table'):
            id: pxt.Int
            value: pxt.Float | None

        create_new_table = lambda: pxt.create_table('new_table', {'id': pxt.Int})
        with pxt_raises(excs.ErrorCode.CONCURRENT_MODIFICATION, match=r"'new_table' was created concurrently"):
            _run_with_concurrent_apply(lambda: TM2.update_all(ROOT), create_new_table)
        assert pxt.get_table('new_table').columns() == ['id']

    # ---------------------------------------------------------------------------------------------------------------
    # Group D: concurrency that doesn't conflict at the column level. A change to a table under update still moves its
    # schema version, so the CAS aborts; only concurrency that leaves the version alone (data changes, reads) succeeds.
    # ---------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize('family', ['update_all', 'single_op'])
    def test_disjoint_adds(self, uses_db: None, fault_injection: None, family: str) -> None:
        """D1: thread 0 adds column x; a concurrent change adds a disjoint column y to the same table."""
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        TM.create_all(ROOT)
        Base.insert([{'id': 1, 'value': 1.0}, {'id': 2, 'value': 2.0}])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            x = value + 1

        if family == 'update_all':
            TMc = pxt.model_base()

            class BaseC(TMc, name='test_table'):
                id: pxt.Int
                value: pxt.Float | None
                y = value + 2

            concurrent = lambda: TMc.update_all(ROOT)
        else:
            t = Base.table
            concurrent = lambda: t.add_computed_column(y=t.value + 2)

        with pxt_raises(excs.ErrorCode.CONCURRENT_MODIFICATION, match=schema_changed('test_table')):
            _run_with_concurrent_apply(lambda: TM2.update_all(ROOT), concurrent)
        cols = pxt.get_table('test_table').columns()
        assert 'x' not in cols and 'y' in cols

    def test_add_column_with_concurrent_insert(self, uses_db: None, fault_injection: None) -> None:
        """D2: thread 0 adds a computed column; a concurrent change inserts rows (data, not schema)."""
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        TM.create_all(ROOT)
        Base.insert([{'id': 1, 'value': 1.0}, {'id': 2, 'value': 2.0}])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            doubled = value * 2

        t = Base.table
        _run_with_concurrent_apply(
            lambda: TM2.update_all(ROOT), lambda: t.insert([{'id': 3, 'value': 3.0}, {'id': 4, 'value': 4.0}])
        )
        res = t.order_by(t.id).select(t.id, t.doubled).collect()
        assert res['doubled'] == [2.0, 4.0, 6.0, 8.0]

    @pytest.mark.parametrize('family', ['update_all', 'single_op'])
    def test_disjoint_drops(self, uses_db: None, fault_injection: None, family: str) -> None:
        """D3: thread 0 drops column x; a concurrent change drops a disjoint column y from the same table."""
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            x: pxt.Float | None
            y: pxt.Float | None

        TM.create_all(ROOT)
        Base.insert([{'id': 1, 'x': 1.0, 'y': 1.0}, {'id': 2, 'x': 2.0, 'y': 2.0}])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            y: pxt.Float | None

        if family == 'update_all':
            TMc = pxt.model_base()

            class BaseC(TMc, name='test_table'):
                id: pxt.Int
                x: pxt.Float | None

            concurrent = lambda: TMc.update_all(ROOT, allow_destructive=True)
        else:
            t = Base.table
            concurrent = lambda: t.drop_column('y')

        with pxt_raises(excs.ErrorCode.CONCURRENT_MODIFICATION, match=schema_changed('test_table')):
            _run_with_concurrent_apply(lambda: TM2.update_all(ROOT, allow_destructive=True), concurrent)
        cols = pxt.get_table('test_table').columns()
        assert 'x' in cols and 'y' not in cols

    def test_disjoint_tables(self, uses_db: None, fault_injection: None) -> None:
        """D4: thread 0 runs update_all() on table T; a concurrent update_all() targets a disjoint table U."""
        TM = pxt.model_base()

        class TableT(TM, name='table_t'):
            id: pxt.Int
            value: pxt.Float | None

        class TableU(TM, name='table_u'):
            id: pxt.Int
            value: pxt.Float | None

        TM.create_all(ROOT)
        TableT.insert([{'id': 1, 'value': 1.0}])
        TableU.insert([{'id': 1, 'value': 1.0}])

        TM2 = pxt.model_base()

        class TableT2(TM2, name='table_t'):
            id: pxt.Int
            value: pxt.Float | None
            x = value + 1

        class TableU2(TM2, name='table_u'):
            id: pxt.Int
            value: pxt.Float | None

        TMc = pxt.model_base()

        class TableTc(TMc, name='table_t'):
            id: pxt.Int
            value: pxt.Float | None

        class TableUc(TMc, name='table_u'):
            id: pxt.Int
            value: pxt.Float | None
            y = value + 2

        # thread 0 updates only table_t; the concurrent update_all touches only table_u.
        _run_with_concurrent_apply(lambda: TM2.update_all(ROOT), lambda: TMc.update_all(ROOT))
        assert 'x' in TableT2.get_metadata()['columns']
        assert 'y' in TableUc.get_metadata()['columns']

    def test_concurrent_readers(self, uses_db: None, fault_injection: None) -> None:
        """D5: thread 0 runs update_all(); a concurrent reader repeatedly queries the table."""
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        TM.create_all(ROOT)
        Base.insert([{'id': i, 'value': float(i)} for i in range(5)])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            x = value + 1

        t = Base.table

        def read() -> None:
            for _ in range(10):
                rows = t.select(t.id, t.value).collect()
                assert len(rows) == 5

        _run_with_concurrent_apply(lambda: TM2.update_all(ROOT), read)
        assert 'x' in BaseV2.get_metadata()['columns']

    def test_idempotent_update(self, uses_db: None, fault_injection: None) -> None:
        """D6: thread 0 and a concurrent actor run the SAME additive update_all(); one applies, the other aborts."""
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        TM.create_all(ROOT)
        Base.insert([{'id': 1, 'value': 1.0}, {'id': 2, 'value': 2.0}])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            x = value + 1

        TMc = pxt.model_base()

        class BaseC(TMc, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            x = value + 1

        with pxt_raises(excs.ErrorCode.CONCURRENT_MODIFICATION, match=schema_changed('test_table')):
            _run_with_concurrent_apply(lambda: TM2.update_all(ROOT), lambda: TMc.update_all(ROOT))
        # the concurrent actor's x is the only one; thread 0's abort added nothing on top of it
        t = pxt.get_table('test_table')
        assert 'x' in t.columns()
        res = t.order_by(t.id).select(t.x).collect()
        assert res['x'] == [2.0, 3.0]

    @pytest.mark.parametrize('family', ['update_all', 'single_op'])
    def test_disjoint_index_change(self, uses_db: None, fault_injection: None, family: str) -> None:
        """D7: thread 0 adds a column; a concurrent change adds a disjoint index to the same table."""
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            text: pxt.String | None

        TM.create_all(ROOT)
        Base.insert([{'id': 1, 'value': 1.0, 'text': 'one'}, {'id': 2, 'value': 2.0, 'text': 'two'}])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            text: pxt.String | None
            x = value + 1

        if family == 'update_all':
            TMc = pxt.model_base()

            class BaseC(TMc, name='test_table'):
                id: pxt.Int
                value: pxt.Float | None
                text: pxt.String | None

                __indexes__ = [EmbeddingIndex(text, embedding=dummy_embedding.using(n=768), name='ix')]

            concurrent = lambda: TMc.update_all(ROOT)
        else:
            t = Base.table
            concurrent = lambda: t.add_embedding_index('text', idx_name='ix', embedding=dummy_embedding.using(n=768))

        with pxt_raises(excs.ErrorCode.CONCURRENT_MODIFICATION, match=schema_changed('test_table')):
            _run_with_concurrent_apply(lambda: TM2.update_all(ROOT), concurrent)
        md = pxt.get_table('test_table').get_metadata()
        assert 'x' not in md['columns']
        assert 'ix' in md['indexes']

    # ---------------------------------------------------------------------------------------------------------------
    # Group E: genuine-race stress (no fault injection).
    # ---------------------------------------------------------------------------------------------------------------

    def test_race_disjoint_tables(self, uses_db: None) -> None:
        """E1: N threads each run update_all() against a distinct table; every table gets its added column."""
        n_threads = 4
        for tid in range(n_threads):
            pxt.create_table(f'table_{tid}', {'id': pxt.Int, 'value': pxt.Float | None})
            pxt.get_table(f'table_{tid}').insert([{'id': 1, 'value': 1.0}])

        def worker(tid: int) -> None:
            TMi = pxt.model_base()

            class Model(TMi, name=f'table_{tid}'):
                id: pxt.Int
                value: pxt.Float | None
                added = value + 1

            TMi.update_all(ROOT)

        errors = _run_workers(worker, n_threads=n_threads)
        assert errors == [], f'errors: {errors[:3]}'

        for tid in range(n_threads):
            assert 'added' in pxt.get_table(f'table_{tid}').get_metadata()['columns']

    def test_race_identical_update_all(self, uses_db: None) -> None:
        """E2: N threads race the SAME additive update_all(); the column is added once, every loser aborts cleanly."""
        TM = pxt.model_base()

        class Base(TM, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        TM.create_all(ROOT)
        Base.insert([{'id': 1, 'value': 1.0}, {'id': 2, 'value': 2.0}])

        TM2 = pxt.model_base()

        class BaseV2(TM2, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            x = value + 1

        n_threads = 4
        errors = _run_workers(lambda _tid: TM2.update_all(ROOT), n_threads=n_threads)
        # a thread whose diff predates the winner's commit loses the CAS; one that diffs afterwards finds nothing to
        # do. Neither produces a duplicate-add error, and the first thread to reach the apply always commits.
        assert all(
            isinstance(e, excs.Error) and e.error_code is excs.ErrorCode.CONCURRENT_MODIFICATION for _, e in errors
        ), f'errors: {errors[:3]}'
        assert len(errors) < n_threads, 'every thread aborted'

        cols = BaseV2.get_metadata()['columns']
        assert 'x' in cols
        t = BaseV2.table
        res = t.order_by(t.id).select(t.x).collect()
        assert res['x'] == [2.0, 3.0]
