"""Indexes on views: an embedding index on a base column, kept up to date by inserts and updates of the base."""

import numpy as np

import pixeltable as pxt


@pxt.udf
def embed(s: str) -> pxt.Array[(3,), pxt.Float]:
    v = np.array([len(s), s.count('a'), 1.0], dtype=np.float32)
    return v / np.linalg.norm(v)


class TestViewIndexes:
    def test_embedding_index_on_base_column(self) -> None:
        t = pxt.create_table('idx_base', {'s': pxt.String})
        t.insert([{'s': 'aaa'}, {'s': 'bcd'}, {'s': 'a'}])
        v = pxt.create_view('idx_view', t.where(t.s != 'a'))
        v.add_embedding_index('s', string_embed=embed)
        # the view's reference to the base column finds the view's index
        sim = v.s.similarity(string='aa')
        assert v.order_by(sim, asc=False).select(v.s).collect()['s'] == ['aaa', 'bcd']
        t.insert([{'s': 'aaaa'}])
        t.update({'s': 'zzz'}, where=t.s == 'aaa')
        assert v.order_by(sim, asc=False).select(v.s).limit(1).collect()['s'] == ['aaaa']
        assert list(v.get_metadata()['indexes']) == ['idx0']
        v.drop_embedding_index(idx_name='idx0')
        assert v.get_metadata()['indexes'] == {}
        # revert() restores a dropped index
        v.revert()
        assert list(v.get_metadata()['indexes']) == ['idx0']
