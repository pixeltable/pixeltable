from typing import List
import sqlalchemy as sql
import pytest

import pixeltable as pt
from pixeltable import catalog
from pixeltable.type_system import StringType, BoolType, IntType, ImageType, ArrayType, ColumnType, FloatType, VideoType
from pixeltable.exprs import Expr, CompoundPredicate, FunctionCall, Literal, InlineDict, InlineArray, ColumnRef
from pixeltable.exprs import RELATIVE_PATH_ROOT as R
from pixeltable.functions import udf_call, dict_map, cast, sum, count
from pixeltable.functions.pil.image import blend
from pixeltable import exceptions as exc
from pixeltable import exprs
from pixeltable.function import FunctionRegistry


class TestExprs:
    def test_basic(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        assert t['c1'].equals(t.c1)
        assert t['c7']['*'].f5.equals(t.c7['*'].f5)

        assert isinstance(t.c1 == None, Expr)
        assert isinstance(t.c1 < 'a', Expr)
        assert isinstance(t.c1 <= 'a', Expr)
        assert isinstance(t.c1 == 'a', Expr)
        assert isinstance(t.c1 != 'a', Expr)
        assert isinstance(t.c1 > 'a', Expr)
        assert isinstance(t.c1 >= 'a', Expr)
        assert isinstance((t.c1 == 'a') & (t.c2 < 5), Expr)
        assert isinstance((t.c1 == 'a') | (t.c2 < 5), Expr)
        assert isinstance(~(t.c1 == 'a'), Expr)

    def test_compound_predicates(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        # compound predicates that can be fully evaluated in SQL
        e = ((t.c1 == 'test string') & (t.c2 > 50)).sql_expr()
        assert len(e.clauses) == 2

        e = ((t.c1 == 'test string') & (t.c2 > 50) & (t.c3 < 1.0)).sql_expr()
        assert len(e.clauses) == 3
        e = ((t.c1 == 'test string') | (t.c2 > 50)).sql_expr()
        assert len(e.clauses) == 2
        e = ((t.c1 == 'test string') | (t.c2 > 50) | (t.c3 < 1.0)).sql_expr()
        assert len(e.clauses) == 3
        e = (~(t.c1 == 'test string')).sql_expr()
        assert isinstance(e, sql.sql.expression.BinaryExpression)

        # compound predicates with Python functions
        @pt.function(return_type=BoolType(), param_types=[StringType()])
        def udf(_: str) -> bool:
            return True
        @pt.function(return_type=BoolType(), param_types=[IntType()])
        def udf2(_: int) -> bool:
            return True

        # & can be split
        p = (t.c1 == 'test string') & udf(t.c1)
        assert p.sql_expr() is None
        sql_pred, other_pred = p.extract_sql_predicate()
        assert isinstance(sql_pred, sql.sql.expression.BinaryExpression)
        assert isinstance(other_pred, FunctionCall)

        p = (t.c1 == 'test string') & udf(t.c1) & (t.c2 > 50)
        assert p.sql_expr() is None
        sql_pred, other_pred = p.extract_sql_predicate()
        assert len(sql_pred.clauses) == 2
        assert isinstance(other_pred, FunctionCall)

        p = (t.c1 == 'test string') & udf(t.c1) & (t.c2 > 50) & udf2(t.c2)
        assert p.sql_expr() is None
        sql_pred, other_pred = p.extract_sql_predicate()
        assert len(sql_pred.clauses) == 2
        assert isinstance(other_pred, CompoundPredicate)

        # | cannot be split
        p = (t.c1 == 'test string') | udf(t.c1)
        assert p.sql_expr() is None
        sql_pred, other_pred = p.extract_sql_predicate()
        assert sql_pred is None
        assert isinstance(other_pred, CompoundPredicate)

    def test_filters(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        _ = t[t.c1 == 'test string'].show()
        print(_)
        _ = t[t.c2 > 50].show()
        print(_)
        _ = t[t.c1n == None].show()
        print(_)
        _ = t[t.c1n != None].show(0)
        print(_)

    def test_exception_handling(self, test_tbl: catalog.Table) -> None:
        t = test_tbl

        # error in expr that's handled in SQL
        with pytest.raises(exc.Error):
            _ = t[(t.c2 + 1) / t.c2].show()

        # error in expr that's handled in Python
        with pytest.raises(exc.Error):
            _ = t[(t.c6.f2 + 1) / (t.c2 - 10)].show()

        # the same, but with an inline function
        @pt.function(return_type=FloatType(), param_types=[IntType(), IntType()])
        def f(a: int, b: int) -> float:
            return a / b
        with pytest.raises(exc.Error):
            _ = t[f(t.c2 + 1, t.c2)].show()

        # error in agg.init()
        class Aggregator:
            def __init__(self):
                self.sum = 1 / 0
            @classmethod
            def make_aggregator(cls):
                return cls()
            def update(self, val):
                pass
            def value(self):
                return 1
        agg = pt.make_aggregate_function(
            IntType(), [IntType()], Aggregator.make_aggregator, Aggregator.update, Aggregator.value)
        with pytest.raises(exc.Error):
            _ = t[agg(t.c2)].show()

        # error in agg.update()
        class Aggregator:
            def __init__(self):
                self.sum = 0
            @classmethod
            def make_aggregator(cls):
                return cls()
            def update(self, val):
                self.sum += 1 / val
            def value(self):
                return 1
        agg = pt.make_aggregate_function(
            IntType(), [IntType()], Aggregator.make_aggregator, Aggregator.update, Aggregator.value)
        with pytest.raises(exc.Error):
            _ = t[agg(t.c2 - 10)].show()

        # error in agg.value()
        class Aggregator:
            def __init__(self):
                self.sum = 0
            @classmethod
            def make_aggregator(cls):
                return cls()
            def update(self, val):
                self.sum += val
            def value(self):
                return 1 / self.sum
        agg = pt.make_aggregate_function(
            IntType(), [IntType()], Aggregator.make_aggregator, Aggregator.update, Aggregator.value)
        with pytest.raises(exc.Error):
            _ = t[t.c2 <= 2][agg(t.c2 - 1)].show()

    def test_error_props(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        with pytest.raises(exc.Error):
            _ = t[t.c1.errortype].show()
        with pytest.raises(exc.Error):
            _ = t[t.c1.errormsg].show()

    @pytest.mark.skip(reason='not implemented')
    def test_nullable(self, test_client: pt.Client) -> None:
        # create table with two int columns
        cols = [
            pt.Column('c1', IntType(nullable=True)),
            pt.Column('c2', IntType(nullable=True))
        ]
        t = test_client.create_table('test', cols)

        # computed column that doesn't allow nulls
        t.add_column(pt.Column('c3', IntType(), computed_with=lambda c1, c2: c1 + c2))
        # function that does allow nulls
        @pt.function(return_type=IntType(), param_types=[IntType(), IntType(nullable=True)])
        def f(a: int, b: int) -> int:
            if b is None:
                return a
            return a + b
        t.add_column(pt.Column('c4', IntType(), computed_with=f(t.c1, t.c2)))

        # data that tests all combinations of nulls
        data = [(1, 1), (1, None), (None, 1), (None, None)]
        t.insert_rows(data)
        result = t[t.c3, t.c4].show(0)
        assert result[0, 0] == 2
        assert result[0, 1] == 2
        assert result[1, 0] == None
        assert result[1, 1] == 1
        assert result[2, 0] == None
        assert result[2, 1] == None
        assert result[3, 0] == None
        assert result[3, 1] == None

    def test_arithmetic_exprs(self, test_tbl: catalog.Table) -> None:
        t = test_tbl

        _ = t[t.c2, t.c6.f3, t.c2 + t.c6.f3, (t.c2 + t.c6.f3) / (t.c6.f3 + 1)].show()
        _ = t[t.c2 + t.c2].show()
        for op1, op2 in [(t.c2, t.c2), (t.c3, t.c3)]:
            _ = t[op1 + op2].show()
            _ = t[op1 - op2].show()
            _ = t[op1 * op2].show()
            _ = t[op1 > 0][op1 / op2].show()

        # non-numeric types
        for op1, op2 in [
            (t.c1, t.c2), (t.c1, 1), (t.c2, t.c1), (t.c2, 'a'),
            (t.c1, t.c3), (t.c1, 1.0), (t.c3, t.c1), (t.c3, 'a')
        ]:
            with pytest.raises(exc.Error):
                _ = t[op1 + op2]
            with pytest.raises(exc.Error):
                _ = t[op1 - op2]
            with pytest.raises(exc.Error):
                _ = t[op1 * op2]
            with pytest.raises(exc.Error):
                _ = t[op1 / op2]

        # TODO: test division; requires predicate
        for op1, op2 in [(t.c6.f2, t.c6.f2), (t.c6.f3, t.c6.f3)]:
            _ = t[op1 + op2].show()
            _ = t[op1 - op2].show()
            _ = t[op1 * op2].show()
            with pytest.raises(exc.Error):
                _ = t[op1 / op2].show()

        for op1, op2 in [
            (t.c6.f1, t.c6.f2), (t.c6.f1, t.c6.f3), (t.c6.f1, 1), (t.c6.f1, 1.0),
            (t.c6.f2, t.c6.f1), (t.c6.f3, t.c6.f1), (t.c6.f2, 'a'), (t.c6.f3, 'a'),
        ]:
            with pytest.raises(exc.Error):
                _ = t[op1 + op2].show()
            with pytest.raises(exc.Error):
                _ = t[op1 - op2].show()
            with pytest.raises(exc.Error):
                _ = t[op1 * op2].show()


    def test_inline_dict(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        df = t[[{'a': t.c1, 'b': {'c': t.c2}, 'd': 1, 'e': {'f': 2}}]]
        result = df.show()
        print(result)

    def test_inline_array(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        result = t[[ [[t.c2, 1], [t.c2, 2]] ]].show()
        t = result.col_types[0]
        assert t.is_array_type()
        assert isinstance(t, ArrayType)
        assert t.shape == (2, 2)
        assert t.dtype == ColumnType.Type.INT

    def test_json_mapper(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        # top-level is dict
        df = t[t.c6.f5['*'] >> (R + 1)]
        res = df.show()
        print(res)
        _ = t[t.c7['*'].f5 >> [R[3], R[2], R[1], R[0]]]
        _ = _.show()
        print(_)
        # target expr contains global-scope dependency
        df = t[
            t.c6.f5['*'] >> (R * t.c6.f5[1])
        ]
        res = df.show()
        print(res)

    def test_dicts(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        # top-level is dict
        _ = t[t.c6.f1]
        _ = _.show()
        print(_)
        # predicate on dict field
        _ = t[t.c6.f2 < 2].show()
        #_ = t[t.c6.f2].show()
        #_ = t[t.c6.f5].show()
        _ = t[t.c6.f6.f8].show()
        _ = t[cast(t.c6.f6.f8, ArrayType((4,), FloatType()))].show()

        # top-level is array
        #_ = t[t.c7['*'].f1].show()
        #_ = t[t.c7['*'].f2].show()
        #_ = t[t.c7['*'].f5].show()
        _ = t[t.c7['*'].f6.f8].show()
        _ = t[t.c7[0].f6.f8].show()
        _ = t[t.c7[:2].f6.f8].show()
        _ = t[t.c7[::-1].f6.f8].show()
        _ = t[cast(t.c7['*'].f6.f8, ArrayType((2, 4), FloatType()))].show()
        print(_)

    def test_arrays(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        t.add_column(catalog.Column('array_col', computed_with=[[t.c2, 1], [1, t.c2]]))
        _ = t[t.array_col].show()
        print(_)
        _ = t[t.array_col[:, 0]].show()
        print(_)

    def test_select_list(self, img_tbl) -> None:
        t = img_tbl
        result = t[t.img].show(n=100)
        _ = result._repr_html_()
        df = t[t.img, udf_call(lambda img: img.rotate(60), ImageType(), tbl=t)]
        _ = df.show(n=100)._repr_html_()
        df = t[[t.img, t.img.rotate(60)]]
        _ = df.show(n=100)._repr_html_()

        with pytest.raises(exc.Error):
            _ = t[t.img.rotate]

    def test_img_members(self, img_tbl) -> None:
        t = img_tbl
        # make sure the limit is applied in Python, not in the SELECT
        result = t[t.img.height > 200][t.img].show(n=3)
        assert len(result) == 3
        result = t[t.img.crop((10, 10, 60, 60))].show(n=100)
        result = t[t.img.crop((10, 10, 60, 60)).resize((100, 100))].show(n=100)
        result = t[t.img.crop((10, 10, 60, 60)).resize((100, 100)).convert('L')].show(n=100)
        result = t[t.img.getextrema()].show(n=100)
        result = t[t.img, t.img.height, t.img.rotate(90)].show(n=100)
        _ = result._repr_html_()

    def test_img_functions(self, img_tbl) -> None:
        t = img_tbl
        result = t[blend(t.img, t.img.rotate(90), 0.5)].show(100)
        print(result)
        from pixeltable.functions.image_embedding import openai_clip
        result = t[openai_clip(t.img)].show(10)
        print(result)
        _ = result._repr_html_()
        _ = t.img.entropy() > 1
        _ = _.extract_sql_predicate()
        _ = (t.img.entropy() > 1) & (t.split == 'train')
        _ = _.extract_sql_predicate()
        _ = (t.img.entropy() > 1) & (t.split == 'train') & (t.split == 'val')
        _ = _.extract_sql_predicate()
        _ = (t.split == 'train') & (t.img.entropy() > 1) & (t.split == 'val') & (t.img.entropy() < 0)
        _ = _.extract_sql_predicate()
        _ = t[(t.split == 'train') & (t.category == 'n03445777')][t.img].show()
        print(_)
        result = t[t.img.width > 1].show()
        print(result)
        result = t[(t.split == 'val') & (t.img.entropy() > 1) & (t.category == 'n03445777')].show()
        print(result)
        result = t[
            (t.split == 'train') & (t.img.entropy() > 1) & (t.split == 'val') & (t.img.entropy() < 0)
        ][t.img, t.split].show()
        print(result)

    def test_categoricals_map(self, img_tbl) -> None:
        t = img_tbl
        m = t[t.category].categorical_map()
        _ = t[dict_map(t.category, m)].show()
        print(_)

    def test_similarity(self, indexed_img_tbl: catalog.Table) -> None:
        t = indexed_img_tbl
        _ = t.show(30)
        probe = t[t.img, t.category].show(1)
        img = probe[0, 0]
        result = t[t.img.nearest(img)].show(10)
        assert len(result) == 10
        # nearest() with one SQL predicate and one Python predicate
        result = t[t.img.nearest(img) & (t.category == probe[0, 1]) & (t.img.width > 1)].show(10)
        # TODO: figure out how to verify results
        #assert len(result) == 3

        result = t[t.img.matches('musical instrument')].show(10)
        assert len(result) == 10
        # matches() with one SQL predicate and one Python predicate
        french_horn_category = 'n03394916'
        result = t[
            t.img.matches('musical instrument') & (t.category == french_horn_category) & (t.img.width > 1)
        ].show(10)
        #assert len(result) == 6

    # TODO: this doesn't work when combined with test_similarity(), for some reason the data table for img_tbl
    # doesn't get created; why?
    def test_similarity2(self, img_tbl: catalog.Table) -> None:
        t = img_tbl
        probe = t[t.img].show(1)
        img = probe[0, 0]

        with pytest.raises(exc.Error):
            _ = t[t.img.nearest(img)].show(10)
        with pytest.raises(exc.Error):
            _ = t[t.img.matches('musical instrument')].show(10)

    def test_serialization(
            self, test_tbl: catalog.Table, test_tbl_exprs: List[exprs.Expr],
            img_tbl: catalog.Table, img_tbl_exprs: List[exprs.Expr]) -> None:
        t = test_tbl
        for e in test_tbl_exprs:
            e_serialized = e.serialize()
            e_deserialized = Expr.deserialize(e_serialized, t)
            assert e.equals(e_deserialized)

        img_t = img_tbl
        for e in img_tbl_exprs:
            e_serialized = e.serialize()
            e_deserialized = Expr.deserialize(e_serialized, img_t)
            assert e.equals(e_deserialized)

    def test_print(self, test_tbl_exprs: List[exprs.Expr], img_tbl_exprs: List[exprs.Expr]) -> None:
        _ = FunctionRegistry.get().library_fns
        for e in test_tbl_exprs:
            _ = str(e)
            print(_)
        for e in img_tbl_exprs:
            _ = str(e)
            print(_)

    def test_subexprs(self, img_tbl: catalog.Table) -> None:
        t = img_tbl
        e = t.img
        subexprs = [s for s in e.subexprs()]
        assert len(subexprs) == 1
        e = t.img.rotate(90).resize((224, 224))
        subexprs = [s for s in e.subexprs()]
        assert len(subexprs) == 3
        subexprs = [s for s in e.subexprs() if isinstance(s, ColumnRef)]
        assert len(subexprs) == 1
        assert t.img.equals(subexprs[0])

    def test_window_fns(self, test_client: pt.Client, test_tbl: catalog.Table) -> None:
        cl = test_client
        t = test_tbl
        _ = t[sum(t.c2, group_by=t.c4, order_by=t.c3)].show(100)

        # conflicting ordering requirements
        with pytest.raises(exc.Error):
            _ = t[sum(t.c2, group_by=t.c4, order_by=t.c3), sum(t.c2, group_by=t.c3, order_by=t.c4)].show(100)
        with pytest.raises(exc.Error):
            _ = t[sum(t.c2, group_by=t.c4, order_by=t.c3), sum(t.c2, group_by=t.c3, order_by=t.c4)].show(100)

        # backfill works
        t.add_column(catalog.Column('c9', computed_with=sum(t.c2, group_by=t.c4, order_by=t.c3)))
        _ = t.c9.col.has_window_fn_call()

        # ordering conflict between frame extraction and window fn
        cols = [
            catalog.Column('video', VideoType(nullable=False)),
            catalog.Column('frame', ImageType(nullable=False)),
            catalog.Column('frame_idx', IntType(nullable=False)),
            catalog.Column('c2', IntType(nullable=False)),
        ]
        vt = cl.create_table(
            'video_test', cols, extract_frames_from='video', extracted_frame_col='frame',
            extracted_frame_idx_col='frame_idx', extracted_fps=0)
        # compatible ordering
        _ = vt[vt.frame, sum(vt.c2, group_by=vt.video, order_by=vt.frame_idx)].show(100)
        with pytest.raises(exc.Error):
            # incompatible ordering
            _ = vt[vt.frame, sum(vt.c2, group_by=vt.frame_idx, order_by=vt.video)].show(100)

        c2 = catalog.Column('c2', IntType(nullable=False))
        c3 = catalog.Column('c3', FloatType(nullable=False))
        c4 = catalog.Column('c4', BoolType(nullable=False))
        new_t = cl.create_table('insert_test', [c2, c3, c4])
        new_t.add_column(catalog.Column(
            'c2_sum', computed_with=sum(new_t.c2, group_by=new_t.c4, order_by=new_t.c3)))
        data_df = t[t.c2, t.c4, t.c3].show(0).to_pandas()
        new_t.insert_pandas(data_df)
        _ = new_t.show(0)
        print(_)

    def test_aggregates(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        _ = t[t.c2 % 2, sum(t.c2), count(t.c2), sum(t.c2) + count(t.c2), sum(t.c2) + (t.c2 % 2)]\
            .group_by(t.c2 % 2).show()

        # check that aggregates don't show up in the wrong places
        with pytest.raises(exc.Error):
            # aggregate in where clause
            _ = t[sum(t.c2) > 0][sum(t.c2)].group_by(t.c2 % 2).show()
        with pytest.raises(exc.Error):
            # aggregate in group_by clause
            _ = t[sum(t.c2)].group_by(sum(t.c2)).show()
        with pytest.raises(exc.Error):
            # mixing aggregates and non-aggregates
            _ = t[sum(t.c2) + t.c2].group_by(t.c2 % 2).show()
        with pytest.raises(exc.Error):
            # nested aggregates
            _ = t[sum(count(t.c2))].group_by(t.c2 % 2).show()
        a = 10
