"""Tests for the partial unique B-tree index on primary key columns."""

import pixeltable as pxt
from pixeltable.index.btree import BtreeIndex
from tests.utils import pxt_raises, reload_catalog, validate_update_status


class TestPrimaryKeyIndex:
    def test_single_pk(self, uses_db: None) -> None:
        """Single-column PK: rejects duplicates, allows re-insert after delete, survives reload."""
        t = pxt.create_table('test_pk', {'id': pxt.Required[pxt.Int], 'name': pxt.String}, primary_key='id')
        validate_update_status(t.insert([{'id': 1, 'name': 'alice'}, {'id': 2, 'name': 'bob'}]), expected_rows=2)

        # Duplicate PK is rejected with a clear error
        with pxt_raises(pxt.ErrorCode.CONSTRAINT_VIOLATION, match='Duplicate primary key'):
            t.insert([{'id': 1, 'name': 'charlie'}])
        assert t.count() == 2
        assert t.where(t.id == 1).collect()['name'] == ['alice']

        # Delete row, then re-insert same PK — partial index only covers live rows
        t.delete(where=t.id == 1)
        assert t.count() == 1
        validate_update_status(t.insert([{'id': 1, 'name': 'charlie'}]), expected_rows=1)
        result = t.order_by(t.id).collect()
        assert result['id'] == [1, 2]
        assert result['name'] == ['charlie', 'bob']

        # Index still enforced after catalog reload
        reload_catalog()
        t = pxt.get_table('test_pk')
        with pxt_raises(pxt.ErrorCode.CONSTRAINT_VIOLATION, match='Duplicate primary key'):
            t.insert([{'id': 1, 'name': 'dupe'}])
        validate_update_status(t.insert([{'id': 3, 'name': 'dave'}]), expected_rows=1)
        assert t.count() == 3

    def test_composite_pk(self, uses_db: None) -> None:
        """Composite PK: partial matches are fine, exact matches are rejected, delete-reinsert works."""
        t = pxt.create_table(
            'test_pk',
            {'a': pxt.Required[pxt.Int], 'b': pxt.Required[pxt.String], 'val': pxt.Int},
            primary_key=['a', 'b'],
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

    def test_string_pk_truncation(self, uses_db: None) -> None:
        """String PK index uses left(col, MAX_STRING_LEN). Strings identical in first MAX_STRING_LEN chars collide."""
        t = pxt.create_table('test_pk', {'key': pxt.Required[pxt.String], 'val': pxt.Int}, primary_key='key')
        base = 'a' * BtreeIndex.MAX_STRING_LEN

        validate_update_status(t.insert([{'key': base + '_suffix1', 'val': 1}]), expected_rows=1)

        # Different string, but first MAX_STRING_LEN chars are identical -- index treats them as duplicates
        with pxt_raises(pxt.ErrorCode.CONSTRAINT_VIOLATION, match='Duplicate primary key'):
            t.insert([{'key': base + '_suffix2', 'val': 2}])

        # String that differs within the first MAX_STRING_LEN chars is fine
        different_prefix = 'b' + 'a' * (BtreeIndex.MAX_STRING_LEN - 1)
        validate_update_status(t.insert([{'key': different_prefix + '_suffix1', 'val': 3}]), expected_rows=1)
        assert t.count() == 2

    def test_batch_with_duplicate_fails_atomically(self, uses_db: None) -> None:
        """A batch containing a duplicate fails and does not persist any rows from the batch."""
        t = pxt.create_table('test_pk', {'id': pxt.Required[pxt.Int], 'v': pxt.String}, primary_key='id')
        validate_update_status(t.insert([{'id': 1, 'v': 'a'}]), expected_rows=1)

        with pxt_raises(pxt.ErrorCode.CONSTRAINT_VIOLATION, match='Duplicate primary key'):
            t.insert([{'id': 2, 'v': 'b'}, {'id': 1, 'v': 'c'}])

        # Original data is unchanged
        assert t.count() == 1
        assert t.collect()['id'] == [1]
        assert t.collect()['v'] == ['a']

    def test_pk_index_row_too_large(self, uses_db: None) -> None:
        """Many PK columns can exceed the btree max row size; error message should be user-friendly."""
        schema = {f'k{i}': pxt.Required[pxt.String] for i in range(11)}
        pk_cols = [f'k{i}' for i in range(11)]
        t = pxt.create_table('test_pk', schema, primary_key=pk_cols)

        row = {f'k{i}': 'a' * BtreeIndex.MAX_STRING_LEN for i in range(11)}
        with pxt_raises(pxt.ErrorCode.CONSTRAINT_VIOLATION, match='Primary key value too large for index'):
            t.insert([row])

    def test_batch_update_with_pk_index(self, uses_db: None) -> None:
        """batch_update works correctly with the PK index: updates expire the old version."""
        t = pxt.create_table('test_pk', {'id': pxt.Required[pxt.Int], 'val': pxt.Int}, primary_key='id')
        validate_update_status(t.insert([{'id': 1, 'val': 10}, {'id': 2, 'val': 20}]), expected_rows=2)

        # Update existing row — old version gets v_max set, new version is live
        validate_update_status(t.batch_update([{'id': 1, 'val': 99}]), expected_rows=1)
        assert t.where(t.id == 1).collect()['val'] == [99]
        assert t.count() == 2

        # The PK is still taken by the live row
        with pxt_raises(pxt.ErrorCode.CONSTRAINT_VIOLATION, match='Duplicate primary key'):
            t.insert([{'id': 1, 'val': 50}])
