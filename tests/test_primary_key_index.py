"""Tests for the unique constraint on primary key columns."""

import random
import string

import pixeltable as pxt
from pixeltable.index.btree import BtreeIndex
from .utils import pxt_raises, reload_catalog, validate_update_status, DatabaseRoot


class TestPrimaryKeyIndex:
    def test_single_pk(self, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        """Single-column PK: rejects duplicates, allows re-insert after delete, survives reload."""
        p = db_root.make_catalog_path
        t = pxt.create_table(
            p('test_pk'),
            {'id': pxt.Int, 'name': pxt.String | None},
            primary_key='id',
            _is_data_versioned=is_data_versioned,
        )
        validate_update_status(t.insert([{'id': 1, 'name': 'alice'}, {'id': 2, 'name': 'bob'}]), expected_rows=2)

        # Duplicate PK is rejected with a clear error
        with pxt_raises(pxt.ErrorCode.CONSTRAINT_VIOLATION, match='Duplicate primary key'):
            t.insert([{'id': 1, 'name': 'charlie'}])
        assert t.count() == 2
        assert t.where(t.id == 1).collect()['name'] == ['alice']

        # Delete row, then re-insert same PK: the deleted row no longer occupies the key
        t.delete(where=t.id == 1)
        assert t.count() == 1
        validate_update_status(t.insert([{'id': 1, 'name': 'charlie'}]), expected_rows=1)
        result = t.order_by(t.id).collect()
        assert result['id'] == [1, 2]
        assert result['name'] == ['charlie', 'bob']

        # A schema change on a non-PK column rebuilds sa_tbl; the PK must survive it
        t.add_column(extra={'type': pxt.Int | None})
        with pxt_raises(pxt.ErrorCode.CONSTRAINT_VIOLATION, match='Duplicate primary key'):
            t.insert([{'id': 1, 'name': 'dupe'}])
        t.drop_column('extra')
        with pxt_raises(pxt.ErrorCode.CONSTRAINT_VIOLATION, match='Duplicate primary key'):
            t.insert([{'id': 1, 'name': 'dupe'}])

        # Still enforced after catalog reload
        reload_catalog()
        t = pxt.get_table(p('test_pk'))
        with pxt_raises(pxt.ErrorCode.CONSTRAINT_VIOLATION, match='Duplicate primary key'):
            t.insert([{'id': 1, 'name': 'dupe'}])
        validate_update_status(t.insert([{'id': 3, 'name': 'dave'}]), expected_rows=1)
        assert t.count() == 3

    def test_nullable_pk_rejected(self, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        """A nullable primary key is rejected however it is declared: Postgres admits duplicate NULLs."""
        p = db_root.make_catalog_path
        msg = r"Primary key column 'id' cannot be nullable"

        # via the primary_key argument
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match=msg):
            pxt.create_table(p('t0'), {'id': pxt.Int | None}, primary_key='id', _is_data_versioned=is_data_versioned)

        # via a column spec's 'primary_key' key
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match=msg):
            pxt.create_table(
                p('t1'), {'id': {'type': pxt.Int | None, 'primary_key': True}}, _is_data_versioned=is_data_versioned
            )

    def test_unknown_pk_column(self, db_root: DatabaseRoot) -> None:
        """The primary_key argument must name columns of the schema."""
        p = db_root.make_catalog_path
        with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND, match=r"Primary key column 'nonexistent' not found"):
            pxt.create_table(p('t0'), {'id': pxt.Int}, primary_key=['id', 'nonexistent'])

    def test_pk_via_column_spec(self, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        """A primary key can be declared by a column spec."""
        p = db_root.make_catalog_path
        t = pxt.create_table(
            p('test_pk'), {'id': {'type': pxt.Int, 'primary_key': True}}, _is_data_versioned=is_data_versioned
        )
        assert t.get_metadata()['columns']['id']['is_primary_key']
        validate_update_status(t.insert([{'id': 1}]), expected_rows=1)
        with pxt_raises(pxt.ErrorCode.CONSTRAINT_VIOLATION, match='Duplicate primary key'):
            t.insert([{'id': 1}])

    def test_composite_pk(self, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        """Composite PK: partial matches are fine, exact matches are rejected, delete-reinsert works."""
        p = db_root.make_catalog_path
        t = pxt.create_table(
            p('test_pk'),
            {'a': pxt.Int, 'b': pxt.String, 'val': pxt.Int | None},
            primary_key=['a', 'b'],
            _is_data_versioned=is_data_versioned,
        )
        validate_update_status(
            t.insert([{'a': 1, 'b': 'x', 'val': 10}, {'a': 1, 'b': 'y', 'val': 20}]), expected_rows=2
        )

        # Same 'a' with different 'b' is fine — only exact composite match is rejected
        with pxt_raises(pxt.ErrorCode.CONSTRAINT_VIOLATION, match='Duplicate primary key'):
            t.insert([{'a': 1, 'b': 'x', 'val': 30}])
        assert t.count() == 2

        # Delete and re-insert the same composite key
        t.delete(where=(t.a == 1) & (t.b == 'x'))
        assert t.count() == 1
        validate_update_status(t.insert([{'a': 1, 'b': 'x', 'val': 99}]), expected_rows=1)
        assert t.where((t.a == 1) & (t.b == 'x')).collect()['val'] == [99]

    def test_string_pk_truncation(self, db_root: DatabaseRoot) -> None:
        """String PK index uses left(col, MAX_STRING_LEN). Strings identical in first MAX_STRING_LEN chars collide."""
        p = db_root.make_catalog_path
        t = pxt.create_table(p('test_pk'), {'key': pxt.String, 'val': pxt.Int | None}, primary_key='key')
        base = 'a' * BtreeIndex.MAX_STRING_LEN

        validate_update_status(t.insert([{'key': base + '_suffix1', 'val': 1}]), expected_rows=1)

        # Different string, but first MAX_STRING_LEN chars are identical -- index treats them as duplicates
        with pxt_raises(pxt.ErrorCode.CONSTRAINT_VIOLATION, match='Duplicate primary key'):
            t.insert([{'key': base + '_suffix2', 'val': 2}])

        # String that differs within the first MAX_STRING_LEN chars is fine
        different_prefix = 'b' + 'a' * (BtreeIndex.MAX_STRING_LEN - 1)
        validate_update_status(t.insert([{'key': different_prefix + '_suffix1', 'val': 3}]), expected_rows=1)
        assert t.count() == 2

    def test_batch_with_duplicate_fails_atomically(
        self, db_root: DatabaseRoot, is_data_versioned: bool
    ) -> None:
        """A batch containing a duplicate fails and does not persist any rows from the batch."""
        p = db_root.make_catalog_path
        t = pxt.create_table(
            p('test_pk'),
            {'id': pxt.Int, 'v': pxt.String | None},
            primary_key='id',
            _is_data_versioned=is_data_versioned,
        )
        validate_update_status(t.insert([{'id': 1, 'v': 'a'}]), expected_rows=1)

        with pxt_raises(pxt.ErrorCode.CONSTRAINT_VIOLATION, match='Duplicate primary key'):
            t.insert([{'id': 2, 'v': 'b'}, {'id': 1, 'v': 'c'}])

        # Original data is unchanged
        assert t.count() == 1
        assert t.collect()['id'] == [1]
        assert t.collect()['v'] == ['a']

    def test_pk_too_long(self, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        """The combined length of PK values exceeds Postgres's limit."""
        p = db_root.make_catalog_path
        schema = {f'k{i}': pxt.String for i in range(11)}
        pk_cols = [f'k{i}' for i in range(11)]
        t = pxt.create_table(p('test_pk'), schema, primary_key=pk_cols, _is_data_versioned=is_data_versioned)

        row = {f'k{i}': 'a' * BtreeIndex.MAX_STRING_LEN for i in range(11)}
        with pxt_raises(pxt.ErrorCode.CONSTRAINT_VIOLATION, match='Primary key value too large for index'):
            t.insert([row])

        if not is_data_versioned:
            # This scenario only raises for operational tables because they do not truncate PK values
            # Note: the value has to be incompressible to exceed the limit
            rng = random.Random(0)
            long_str = ''.join(rng.choices(string.ascii_letters + string.digits, k=4000))
            row = {f'k{i}': long_str if i == 0 else '' for i in range(11)}
            with pxt_raises(pxt.ErrorCode.CONSTRAINT_VIOLATION, match='Primary key value too large for index'):
                t.insert([row])

    def test_batch_update_with_pk_index(self, db_root: DatabaseRoot) -> None:
        """batch_update works correctly with the PK index: updates expire the old version."""
        p = db_root.make_catalog_path
        t = pxt.create_table(p('test_pk'), {'id': pxt.Int, 'val': pxt.Int | None}, primary_key='id')
        validate_update_status(t.insert([{'id': 1, 'val': 10}, {'id': 2, 'val': 20}]), expected_rows=2)

        # Update existing row — old version gets v_max set, new version is live
        validate_update_status(t.batch_update([{'id': 1, 'val': 99}]), expected_rows=1)
        assert t.where(t.id == 1).collect()['val'] == [99]
        assert t.count() == 2

        # The PK is still taken by the live row
        with pxt_raises(pxt.ErrorCode.CONSTRAINT_VIOLATION, match='Duplicate primary key'):
            t.insert([{'id': 1, 'val': 50}])

    def test_prohibited_pk_col_ops(self, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        p = db_root.make_catalog_path
        t = pxt.create_table(
            p('test'),
            {'id0': pxt.Int, 'id1': pxt.Int},
            primary_key=['id0', 'id1'],
            _is_data_versioned=is_data_versioned,
        )
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Cannot drop primary key column'):
            t.drop_column(t.id0)

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Cannot add primary key column'):
            t.add_column(new={'type': pxt.Int, 'primary_key': True})
