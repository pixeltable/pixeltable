import json
import math
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import PIL.Image
import pytest
import sqlalchemy as sql

import pixeltable as pxt
import pixeltable.func as func
from pixeltable import catalog
from pixeltable import exceptions as excs
from pixeltable import exprs
from pixeltable.exprs import RELATIVE_PATH_ROOT as R
from pixeltable.exprs import ColumnRef, Expr
from pixeltable.functions import cast
from pixeltable.functions.globals import count, sum
from pixeltable.iterators import FrameIterator
from pixeltable.type_system import ArrayType, BoolType, ColumnType, FloatType, IntType, StringType, VideoType

from .utils import get_image_files, reload_catalog, skip_test_if_not_installed, validate_update_status


class TestExprs:
    @pxt.udf(return_type=FloatType(), param_types=[IntType(), IntType()])
    def div_0_error(a: int, b: int) -> float:
        return a / b

    # function that does allow nulls
    @pxt.udf(return_type=FloatType(nullable=True),
            param_types=[FloatType(nullable=False), FloatType(nullable=True)])
    def null_args_fn(a: int, b: int) -> int:
        if b is None:
            return a
        return a + b

    # error in agg.init()
    @pxt.uda(update_types=[IntType()], value_type=IntType())
    class init_exc(pxt.Aggregator):
        def __init__(self):
            self.sum = 1 / 0
        def update(self, val):
            pass
        def value(self):
            return 1

    # error in agg.update()
    @pxt.uda(update_types=[IntType()], value_type=IntType())
    class update_exc(pxt.Aggregator):
        def __init__(self):
            self.sum = 0
        def update(self, val):
            self.sum += 1 / val
        def value(self):
            return 1

    # error in agg.value()
    @pxt.uda(update_types=[IntType()], value_type=IntType())
    class value_exc(pxt.Aggregator):
        def __init__(self):
            self.sum = 0
        def update(self, val):
            self.sum += val
        def value(self):
            return 1 / self.sum

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
        with pytest.raises(AttributeError) as excinfo:
            _ = t.does_not_exist
        assert 'unknown' in str(excinfo.value).lower()

    def test_compound_predicates(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        # compound predicates that can be fully evaluated in SQL
        _ = t.where((t.c1 == 'test string') & (t.c6.f1 > 50)).collect()
        _ = t.where((t.c1 == 'test string') & (t.c2 > 50)).collect()
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

        with pytest.raises(TypeError) as exc_info:
            _ = t.where((t.c1 == 'test string') or (t.c6.f1 > 50)).collect()
        assert 'cannot be used in conjunction with python boolean operators' in str(exc_info.value).lower()

        # # compound predicates with Python functions
        # @pt.udf(return_type=BoolType(), param_types=[StringType()])
        # def udf(_: str) -> bool:
        #     return True
        # @pt.udf(return_type=BoolType(), param_types=[IntType()])
        # def udf2(_: int) -> bool:
        #     return True

        # TODO: find a way to test this
        # # & can be split
        # p = (t.c1 == 'test string') & udf(t.c1)
        # assert p.sql_expr() is None
        # sql_pred, other_pred = p.extract_sql_predicate()
        # assert isinstance(sql_pred, sql.sql.expression.BinaryExpression)
        # assert isinstance(other_pred, FunctionCall)
        #
        # p = (t.c1 == 'test string') & udf(t.c1) & (t.c2 > 50)
        # assert p.sql_expr() is None
        # sql_pred, other_pred = p.extract_sql_predicate()
        # assert len(sql_pred.clauses) == 2
        # assert isinstance(other_pred, FunctionCall)
        #
        # p = (t.c1 == 'test string') & udf(t.c1) & (t.c2 > 50) & udf2(t.c2)
        # assert p.sql_expr() is None
        # sql_pred, other_pred = p.extract_sql_predicate()
        # assert len(sql_pred.clauses) == 2
        # assert isinstance(other_pred, CompoundPredicate)
        #
        # # | cannot be split
        # p = (t.c1 == 'test string') | udf(t.c1)
        # assert p.sql_expr() is None
        # sql_pred, other_pred = p.extract_sql_predicate()
        # assert sql_pred is None
        # assert isinstance(other_pred, CompoundPredicate)

    def test_filters(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        _ = t.where(t.c1 == 'test string').show()
        print(_)
        _ = t.where(t.c2 > 50).show()
        print(_)
        _ = t.where(t.c1n == None).show()
        print(_)
        _ = t.where(t.c1n != None).show(0)
        print(_)

    def test_exception_handling(self, test_tbl: catalog.Table) -> None:
        t = test_tbl

        # error in expr that's handled in SQL
        with pytest.raises(excs.Error):
            _ = t[(t.c2 + 1) / t.c2].show()

        # error in expr that's handled in Python
        with pytest.raises(excs.Error):
            _ = t[(t.c6.f2 + 1) / (t.c2 - 10)].show()

        # the same, but with an inline function
        with pytest.raises(excs.Error):
            _ = t[self.div_0_error(t.c2 + 1, t.c2)].show()

        # error in agg.init()
        with pytest.raises(excs.Error) as exc_info:
            _ = t[self.init_exc(t.c2)].show()
        assert 'division by zero' in str(exc_info.value)

        # error in agg.update()
        with pytest.raises(excs.Error):
            _ = t[self.update_exc(t.c2 - 10)].show()

        # error in agg.value()
        with pytest.raises(excs.Error):
            _ = t[t.c2 <= 2][self.value_exc(t.c2 - 1)].show()

    def test_props(self, test_tbl: catalog.Table, img_tbl: catalog.Table) -> None:
        t = test_tbl
        # errortype/-msg for computed column
        res = t.select(error=t.c8.errortype).collect()
        assert res.to_pandas()['error'].isna().all()
        res = t.select(error=t.c8.errormsg).collect()
        assert res.to_pandas()['error'].isna().all()

        img_t = img_tbl
        # fileurl
        res = img_t.select(img_t.img.fileurl).show(0).to_pandas()
        stored_urls = set(res.iloc[:, 0])
        assert len(stored_urls) == len(res)
        all_urls = set(urllib.parse.urljoin('file:', urllib.request.pathname2url(path)) for path in get_image_files())
        assert stored_urls <= all_urls

        # localpath
        res = img_t.select(img_t.img.localpath).show(0).to_pandas()
        stored_paths = set(res.iloc[:, 0])
        assert len(stored_paths) == len(res)
        all_paths  = set(get_image_files())
        assert stored_paths <= all_paths

        # errortype/-msg for image column
        res = img_t.select(error=img_t.img.errortype).collect().to_pandas()
        assert res['error'].isna().all()
        res = img_t.select(error=img_t.img.errormsg).collect().to_pandas()
        assert res['error'].isna().all()

        for c in [t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7]:
            # errortype/errormsg only applies to stored computed and media columns
            with pytest.raises(excs.Error) as excinfo:
                _ = t.select(c.errortype).show()
            assert 'only valid for' in str(excinfo.value)
            with pytest.raises(excs.Error) as excinfo:
                _ = t.select(c.errormsg).show()
            assert 'only valid for' in str(excinfo.value)

            # fileurl/localpath only applies to media columns
            with pytest.raises(excs.Error) as excinfo:
                _ = t.select(t.c1.fileurl).show()
            assert 'only valid for' in str(excinfo.value)
            with pytest.raises(excs.Error) as excinfo:
                _ = t.select(t.c1.localpath).show()
            assert 'only valid for' in str(excinfo.value)

        # fileurl/localpath doesn't apply to unstored computed img columns
        img_t.add_column(c9=img_t.img.rotate(30), stored=False)
        with pytest.raises(excs.Error) as excinfo:
            _ = img_t.select(img_t.c9.localpath).show()
        assert 'computed unstored' in str(excinfo.value)

    def test_null_args(self, reset_db) -> None:
        # create table with two int columns
        schema = {'c1': FloatType(nullable=True), 'c2': FloatType(nullable=True)}
        t = pxt.create_table('test', schema)

        # computed column that doesn't allow nulls
        t.add_column(c3=lambda c1, c2: c1 + c2, type=FloatType(nullable=False))
        t.add_column(c4=self.null_args_fn(t.c1, t.c2))

        # data that tests all combinations of nulls
        data = [{'c1': 1.0, 'c2': 1.0}, {'c1': 1.0, 'c2': None}, {'c1': None, 'c2': 1.0}, {'c1': None, 'c2': None}]
        status = t.insert(data, fail_on_exception=False)
        assert status.num_rows == len(data)
        assert status.num_excs >= len(data) - 1
        result = t.select(t.c3, t.c4).collect()
        assert result['c3'] == [2.0, None, None, None]
        assert result['c4'] == [2.0, 1.0, None, None]

    def test_arithmetic_exprs(self, test_tbl: catalog.Table) -> None:
        t = test_tbl

        # Add nullable int and float columns
        t.add_column(c2n=IntType(nullable=True))
        t.add_column(c3n=FloatType(nullable=True))
        t.where(t.c2 % 7 != 0).update({'c2n': t.c2, 'c3n': t.c3})

        _ = t[t.c2, t.c6.f3, t.c2 + t.c6.f3, (t.c2 + t.c6.f3) / (t.c6.f3 + 1)].show()
        _ = t[t.c2 + t.c2].show()
        for op1, op2 in [(t.c2, t.c2), (t.c3, t.c3), (t.c2, t.c2n), (t.c2n, t.c2)]:
            _ = t.select(op1 + op2).show()
            _ = t.select(op1 - op2).show()
            _ = t.select(op1 * op2).show()
            _ = t.where(op1 > 0).select(op1 / op2).show()
            _ = t.where(op1 > 0).select(op1 % op2).show()
            _ = t.where(op1 > 0).select(op1 // op2).show()

        # non-numeric types
        for op1, op2 in [
            (t.c1, t.c2), (t.c1, 1), (t.c2, t.c1), (t.c2, 'a'),
            (t.c1, t.c3), (t.c1, 1.0), (t.c3, t.c1), (t.c3, 'a')
        ]:
            with pytest.raises(excs.Error):
                _ = t[op1 + op2]
            with pytest.raises(excs.Error):
                _ = t[op1 - op2]
            with pytest.raises(excs.Error):
                _ = t[op1 * op2]
            with pytest.raises(excs.Error):
                _ = t[op1 / op2]
            with pytest.raises(excs.Error):
                _ = t[op1 % op2]
            with pytest.raises(excs.Error):
                _ = t[op1 // op2]

        # TODO: test division; requires predicate
        for op1, op2 in [(t.c6.f2, t.c6.f2), (t.c6.f3, t.c6.f3)]:
            _ = t[op1 + op2].show()
            _ = t[op1 - op2].show()
            _ = t[op1 * op2].show()
            with pytest.raises(excs.Error):
                _ = t[op1 / op2].show()

        for op1, op2 in [
            (t.c6.f1, t.c6.f2), (t.c6.f1, t.c6.f3), (t.c6.f1, 1), (t.c6.f1, 1.0),
            (t.c6.f2, t.c6.f1), (t.c6.f3, t.c6.f1), (t.c6.f2, 'a'), (t.c6.f3, 'a'),
        ]:
            with pytest.raises(excs.Error):
                _ = t[op1 + op2].show()
            with pytest.raises(excs.Error):
                _ = t[op1 - op2].show()
            with pytest.raises(excs.Error):
                _ = t[op1 * op2].show()

        # Test literal exprs
        results = t.where(t.c2 == 7).select(
            -t.c2,
            t.c2 + 2,
            t.c2 - 2,
            t.c2 * 2,
            t.c2 / 2,
            t.c2 % 2,
            t.c2 // 2,
            2 + t.c2,
            2 - t.c2,
            2 * t.c2,
            2 / t.c2,
            2 % t.c2,
            2 // t.c2,
        ).collect()
        assert list(results[0].values()) == [-7, 9, 5, 14, 3.5, 1, 3, 9, -5, 14, 0.2857142857142857, 2, 0]

        # Test that arithmetic operations give the right answers. We do this two ways:
        # (i) with primitive operators only, to ensure that the arithmetic operations are done in SQL when possible;
        # (ii) with a Python function call interposed, to ensure that the arithmetic operations are always done in Python;
        # (iii) and (iv), as (i) and (ii) but with JsonType expressions.
        primitive_ops = (t.c2, t.c3)
        forced_python_ops = (t.c2.apply(math.floor, col_type=IntType()), t.c3.apply(math.floor, col_type=FloatType()))
        json_primitive_ops = (t.c6.f2, t.c6.f3)
        json_forced_python_ops = (t.c6.f2.apply(math.floor, col_type=IntType()), t.c6.f3.apply(math.floor, col_type=FloatType()))
        for (int_operand, float_operand) in (primitive_ops, forced_python_ops, json_primitive_ops, json_forced_python_ops):
            results = t.where(t.c2 == 7).select(
                add_int=int_operand + (t.c2 - 4),
                sub_int=int_operand - (t.c2 - 4),
                mul_int=int_operand * (t.c2 - 4),
                truediv_int=int_operand / (t.c2 - 4),
                mod_int=int_operand % (t.c2 - 4),
                neg_floordiv_int=(int_operand * -1) // (t.c2 - 4),
                add_float=float_operand + (t.c3 - 4.0),
                sub_float=float_operand - (t.c3 - 4.0),
                mul_float=float_operand * (t.c3 - 4.0),
                truediv_float=float_operand / (t.c3 - 4.0),
                mod_float=float_operand % (t.c3 - 4.0),
                floordiv_float=float_operand // (t.c3 - 4.0),
                neg_floordiv_float=(float_operand * -1) // (t.c3 - 4.0),
                add_int_to_nullable=int_operand + (t.c2n - 4),
                add_float_to_nullable=float_operand + (t.c3n - 4.0),
            ).collect()
            assert list(results[0].values()) == [10, 4, 21, 2.3333333333333335, 1, -3, 10.0, 4.0, 21.0, 2.3333333333333335, 1.0, 2.0, -3.0, None, None], (
                f'Failed with operands: {int_operand}, {float_operand}'
            )

        with pytest.raises(excs.Error) as exc_info:
            t.select(t.c6 + t.c2.apply(math.floor, col_type=IntType())).collect()
        assert '+ requires numeric type, but c6 has type dict' in str(exc_info.value)

    def test_comparison(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        # Test that comparison operations give the right answers. As with arithmetic operations, we do this two ways:
        # (i) with primitive operators only, to ensure that the comparison operations are done in SQL when possible;
        # (ii) with a Python function call interposed, to ensure that the comparison operations are always done in Python.
        comparison_pairs = (
            (t.c1, "test string 10"),       # string-to-string
            (t.c2, 50),                     # int-to-int
            (t.c3, 50.1),                   # float-to-float
            (t.c5, datetime(2024, 7, 2)),   # datetime-to-datetime
        )
        for expr1, expr2 in comparison_pairs:
            forced_expr1 = expr1.apply(lambda x: x, col_type=expr1.col_type)
            for a_expr, b_expr in ((expr1, expr2), (expr2, expr1), (forced_expr1, expr2), (expr2, forced_expr1)):
                results = t.select(
                    a=a_expr,
                    b=b_expr,
                    eq=a_expr == b_expr,
                    ne=a_expr != b_expr,
                    lt=a_expr < b_expr,
                    le=a_expr <= b_expr,
                    gt=a_expr > b_expr,
                    ge=a_expr >= b_expr,
                ).collect()
                a_results = results['a']
                b_results = results['b']
                assert results['eq'] == [a == b for a, b in zip(a_results, b_results)], f'{a_expr} == {b_expr}'
                assert results['ne'] == [a != b for a, b in zip(a_results, b_results)], f'{a_expr} != {b_expr}'
                assert results['lt'] == [a < b for a, b in zip(a_results, b_results)], f'{a_expr} < {b_expr}'
                assert results['le'] == [a <= b for a, b in zip(a_results, b_results)], f'{a_expr} <= {b_expr}'
                assert results['gt'] == [a > b for a, b in zip(a_results, b_results)], f'{a_expr} > {b_expr}'
                assert results['ge'] == [a >= b for a, b in zip(a_results, b_results)], f'{a_expr} >= {b_expr}'

    def test_inline_dict(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        df = t[[{'a': t.c1, 'b': {'c': t.c2}, 'd': 1, 'e': {'f': 2}}]]
        result = df.show()
        print(result)

    def test_inline_array(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        result = t.select([[t.c2, 1], [t.c2, 2]]).show()
        t = next(iter(result.schema.values()))
        assert t.is_array_type()
        assert isinstance(t, ArrayType)
        assert t.shape == (2, 2)
        assert t.dtype == ColumnType.Type.INT

    def test_json_slice(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        t['orig'] = t.c6.f5
        t['slice_all'] = t.c6.f5[:]
        t['slice_to'] = t.c6.f5[:7]
        t['slice_from'] = t.c6.f5[3:]
        t['slice_range'] = t.c6.f5[3:7]
        t['slice_range_step'] = t.c6.f5[3:7:2]
        res = t.collect()
        orig = res['orig']
        assert all(res['slice_all'][i] == orig[i] for i in range(len(orig)))
        assert all(res['slice_to'][i] == orig[i][:7] for i in range(len(orig)))
        assert all(res['slice_from'][i] == orig[i][3:] for i in range(len(orig)))
        assert all(res['slice_range'][i] == orig[i][3:7] for i in range(len(orig)))
        assert all(res['slice_range_step'][i] == orig[i][3:7:2] for i in range(len(orig)))

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
        t.add_column(array_col=pxt.array([[t.c2, 1], [1, t.c2]]))
        _ = t[t.array_col].show()
        print(_)
        _ = t[t.array_col[:, 0]].show()
        print(_)

    def test_in(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        user_cols = [t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7]
        # list of literals
        rows = list(t.where(t.c2.isin([1, 2, 3])).select(*user_cols).collect())
        assert len(rows) == 3

        # list of literals with some incompatible values
        rows = list(t.where(t.c2.isin(['a', datetime.now(), 1, 2, 3])).select(*user_cols).collect())
        assert len(rows) == 3

        # set of literals
        rows = list(t.where(t.c2.isin({1, 2, 3})).select(*user_cols).collect())
        assert len(rows) == 3

        # dict of literals
        rows = list(t.where(t.c2.isin({1: 'a', 2: 'b', 3: 'c'})).select(*user_cols).collect())
        assert len(rows) == 3

        # json expr
        rows = list(t.where(t.c2.isin(t.c6.f5)).select(*user_cols).collect())
        assert len(rows) == 5

        with pytest.raises(excs.Error) as excinfo:
            # not a scalar
            _ = t.where(t.c6.isin([{'a': 1}, {'b': 2}])).collect()
        assert 'only supported for scalar types' in str(excinfo.value)

        with pytest.raises(excs.Error) as excinfo:
            # bad json path returns None
            _ = t.where(t.c2.isin(t.c7.badpath)).collect()
        assert 'must be an Iterable' in str(excinfo.value)

        with pytest.raises(excs.Error) as excinfo:
            # json path returns scalar
            _ = t.where(t.c2.isin(t.c6.f2)).collect()
        assert ', not 0' in str(excinfo.value)

        with pytest.raises(excs.Error) as excinfo:
            # not a scalar
            _ = t.where(t.c2.isin(t.c1)).collect()
        assert 'c1 has type string' in str(excinfo.value)

        status = t.add_column(in_test=t.c2.isin([1, 2, 3]))
        assert status.num_excs == 0
        def inc_pk(rows: list[dict], offset: int) -> None:
            for r in rows:
                r['c2'] += offset
        inc_pk(rows, 1000)
        validate_update_status(t.insert(rows), len(rows))

        # still works after catalog reload
        reload_catalog()
        t = pxt.get_table('test_tbl')
        inc_pk(rows, 1000)
        validate_update_status(t.insert(rows), len(rows))

    def test_astype(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        # Convert int to float
        status = t.add_column(c2_as_float=t.c2.astype(FloatType()))
        assert status.num_excs == 0
        data = t.select(t.c2, t.c2_as_float).collect()
        for row in data:
            assert isinstance(row['c2'], int)
            assert isinstance(row['c2_as_float'], float)
            assert row['c2'] == row['c2_as_float']
        # Compound expression
        status = t.add_column(compound_as_float=(t.c2 + 1).astype(FloatType()))
        assert status.num_excs == 0
        data = t.select(t.c2, t.compound_as_float).collect()
        for row in data:
            assert isinstance(row['compound_as_float'], float)
            assert row['c2'] + 1 == row['compound_as_float']
        # Type conversion error
        status = t.add_column(c2_as_string=t.c2.astype(StringType()))
        assert status.num_excs == t.count()

    def test_astype_str_to_img(self, reset_db) -> None:
        img_files = get_image_files()
        img_files = img_files[:5]
        # store relative paths in the table
        parent_dir = Path(img_files[0]).parent
        assert(all(parent_dir == Path(img_file).parent for img_file in img_files))
        t = pxt.create_table('astype_test', {'rel_path': StringType()})
        validate_update_status(t.insert({'rel_path': Path(f).name} for f in img_files), expected_rows=len(img_files))

        # create a computed image column constructed from the relative paths
        import pixeltable.functions as pxtf
        validate_update_status(
            t.add_column(
                img=pxtf.string.format('{0}/{1}', str(parent_dir), t.rel_path).astype(pxt.ImageType()), stored=True)
        )
        loaded_imgs = t.select(t.img).collect()['img']
        orig_imgs = [PIL.Image.open(f) for f in img_files]
        for orig_img, retrieved_img in zip(orig_imgs, loaded_imgs):
            assert np.array_equal(np.array(orig_img), np.array(retrieved_img))

        # the same for a select list item
        loaded_imgs = (
            t.select(img=pxtf.string.format('{0}/{1}', str(parent_dir), t.rel_path).astype(pxt.ImageType()))
            .collect()['img']
        )
        for orig_img, retrieved_img in zip(orig_imgs, loaded_imgs):
            assert np.array_equal(np.array(orig_img), np.array(retrieved_img))

    def test_apply(self, test_tbl: catalog.Table) -> None:

        t = test_tbl

        # For each column c1, ..., c5, we create a new column ci_as_str that converts it to
        # a string, then check that each row is correctly converted
        # (For c1 this is the no-op string-to-string conversion)
        for col_id in range(1, 6):
            col_name = f'c{col_id}'
            str_col_name = f'c{col_id}_str'
            status = t.add_column(**{str_col_name: t[col_name].apply(str)})
            assert status.num_excs == 0
            data = t.select(t[col_name], t[str_col_name]).collect()
            for row in data:
                assert row[str_col_name] == str(row[col_name])

        # Test a compound expression with apply
        status = t.add_column(c2_plus_1_str=(t.c2 + 1).apply(str))
        assert status.num_excs == 0
        data = t.select(t.c2, t.c2_plus_1_str).collect()
        for row in data:
            assert row['c2_plus_1_str'] == str(row['c2'] + 1)

        # For columns c6, c7, try using json.dumps and json.loads to emit and parse JSON <-> str
        for col_id in range(6, 8):
            col_name = f'c{col_id}'
            str_col_name = f'c{col_id}_str'
            back_to_json_col_name = f'c{col_id}_back_to_json'
            status = t.add_column(**{str_col_name: t[col_name].apply(json.dumps)})
            assert status.num_excs == 0
            status = t.add_column(**{back_to_json_col_name: t[str_col_name].apply(json.loads)})
            assert status.num_excs == 0
            data = t.select(t[col_name], t[str_col_name], t[back_to_json_col_name]).collect()
            for row in data:
                assert row[str_col_name] == json.dumps(row[col_name])
                assert row[back_to_json_col_name] == row[col_name]

        def f1(x):
            return str(x)

        # Now test that a function without a return type throws an exception ...
        with pytest.raises(excs.Error) as exc_info:
            t.c2.apply(f1)
        assert 'Column type of `f1` cannot be inferred.' in str(exc_info.value)

        # ... but works if the type is specified explicitly.
        status = t.add_column(c2_str_f1=t.c2.apply(f1, col_type=StringType()))
        assert status.num_excs == 0

        # Test that the return type of a function can be successfully inferred.
        def f2(x) -> str:
            return str(x)

        status = t.add_column(c2_str_f2=t.c2.apply(f2))
        assert status.num_excs == 0

        # Test various validation failures.

        def f3(x, y) -> str:
            return f'{x}{y}'

        with pytest.raises(excs.Error) as exc_info:
            t.c2.apply(f3)  # Too many required parameters
        assert str(exc_info.value) == 'Function `f3` has multiple required parameters.'

        def f4() -> str:
            return "pixeltable"

        with pytest.raises(excs.Error) as exc_info:
            t.c2.apply(f4)  # No positional parameters
        assert str(exc_info.value) == 'Function `f4` has no positional parameters.'

        def f5(**kwargs) -> str:
            return ""

        with pytest.raises(excs.Error) as exc_info:
            t.c2.apply(f5)  # No positional parameters
        assert str(exc_info.value) == 'Function `f5` has no positional parameters.'

        # Ensure these varargs signatures are acceptable

        def f6(x, **kwargs) -> str:
            return x

        t.c2.apply(f6)

        def f7(x, *args) -> str:
            return x

        t.c2.apply(f7)

        def f8(*args) -> str:
            return ''

        t.c2.apply(f8)

    def test_select_list(self, img_tbl) -> None:
        t = img_tbl
        result = t[t.img].show(n=100)
        _ = result._repr_html_()
        df = t[[t.img, t.img.rotate(60)]]
        _ = df.show(n=100)._repr_html_()

        with pytest.raises(excs.Error):
            _ = t[t.img.rotate]

    def test_img_members(self, img_tbl) -> None:
        t = img_tbl
        # make sure the limit is applied in Python, not in the SELECT
        result = t.where(t.img.height > 200).select(t.img).show(n=3)
        assert len(result) == 3
        result = t.select(t.img.crop((10, 10, 60, 60))).show(n=100)
        result = t.select(t.img.crop((10, 10, 60, 60)).resize((100, 100))).show(n=100)
        result = t.select(t.img.crop((10, 10, 60, 60)).resize((100, 100)).convert('L')).show(n=100)
        result = t.select(t.img.getextrema()).show(n=100)
        result = t.select(t.img, t.img.height, t.img.rotate(90)).show(n=100)
        _ = result._repr_html_()

    def test_ext_imgs(self, reset_db) -> None:
        t = pxt.create_table('img_test', {'img': pxt.ImageType()})
        img_urls = [
            'https://raw.github.com/pixeltable/pixeltable/main/docs/source/data/images/000000000030.jpg',
            'https://raw.github.com/pixeltable/pixeltable/main/docs/source/data/images/000000000034.jpg',
            'https://raw.github.com/pixeltable/pixeltable/main/docs/source/data/images/000000000042.jpg',
            'https://raw.github.com/pixeltable/pixeltable/main/docs/source/data/images/000000000049.jpg',
            'https://raw.github.com/pixeltable/pixeltable/main/docs/source/data/images/000000000057.jpg',
            'https://raw.github.com/pixeltable/pixeltable/main/docs/source/data/images/000000000061.jpg',
            'https://raw.github.com/pixeltable/pixeltable/main/docs/source/data/images/000000000063.jpg',
            'https://raw.github.com/pixeltable/pixeltable/main/docs/source/data/images/000000000064.jpg',
            'https://raw.github.com/pixeltable/pixeltable/main/docs/source/data/images/000000000069.jpg',
            'https://raw.github.com/pixeltable/pixeltable/main/docs/source/data/images/000000000071.jpg',
        ]
        t.insert({'img': url} for url in img_urls)
        # this fails with an assertion
        # TODO: fix it
        #res = t.where(t.img.width < 600).collect()

    def test_img_exprs(self, img_tbl) -> None:
        t = img_tbl
        _ = t.where(t.img.width < 600).collect()
        _ = (t.img.entropy() > 1) & (t.split == 'train')
        _ = (t.img.entropy() > 1) & (t.split == 'train') & (t.split == 'val')
        _ = (t.split == 'train') & (t.img.entropy() > 1) & (t.split == 'val') & (t.img.entropy() < 0)
        result = t.where((t.split == 'train') & (t.category == 'n03445777')).select(t.img).show()
        print(result)
        result = t.where(t.img.width > 1).show()
        print(result)
        result = t.where((t.split == 'val') & (t.img.entropy() > 1) & (t.category == 'n03445777')).show()
        print(result)
        result = t.where(
            (t.split == 'train') & (t.img.entropy() > 1) & (t.split == 'val') & (t.img.entropy() < 0)
        ).select(t.img, t.split).show()
        print(result)

    @pytest.mark.skip(reason='temporarily disabled')
    def test_similarity(self, small_img_tbl) -> None:
        t = small_img_tbl
        _ = t.show(30)
        probe = t.select(t.img, t.category).show(1)
        img = probe[0, 0]
        result = t.where(t.img.nearest(img)).show(10)
        assert len(result) == 10
        # nearest() with one SQL predicate and one Python predicate
        result = t[t.img.nearest(img) & (t.category == probe[0, 1]) & (t.img.width > 1)].show(10)
        # TODO: figure out how to verify results

        with pytest.raises(excs.Error) as exc_info:
            _ = t[t.img.nearest(img)].order_by(t.category).show()
        assert 'cannot be used in conjunction with' in str(exc_info.value)

        result = t[t.img.nearest('musical instrument')].show(10)
        assert len(result) == 10
        # matches() with one SQL predicate and one Python predicate
        french_horn_category = 'n03394916'
        result = t[
            t.img.nearest('musical instrument') & (t.category == french_horn_category) & (t.img.width > 1)
        ].show(10)

        with pytest.raises(excs.Error) as exc_info:
            _ = t[t.img.nearest(5)].show()
        assert 'requires' in str(exc_info.value)

    # TODO: this doesn't work when combined with test_similarity(), for some reason the data table for img_tbl
    # doesn't get created; why?
    def test_similarity2(self, img_tbl: catalog.Table) -> None:
        t = img_tbl
        probe = t[t.img].show(1)
        img = probe[0, 0]

        with pytest.raises(AttributeError):
            _ = t[t.img.nearest(img)].show(10)
        with pytest.raises(AttributeError):
            _ = t[t.img.nearest('musical instrument')].show(10)

    def test_ids(
            self, test_tbl: catalog.Table, test_tbl_exprs: List[exprs.Expr],
            img_tbl: catalog.Table, img_tbl_exprs: List[exprs.Expr]
    ) -> None:
        skip_test_if_not_installed('transformers')
        d: Dict[int, exprs.Expr] = {}
        for e in test_tbl_exprs:
            assert e.id is not None
            d[e.id] = e
        for e in img_tbl_exprs:
            assert e.id is not None
            d[e.id] = e
        assert len(d) == len(test_tbl_exprs) + len(img_tbl_exprs)

    def test_serialization(
            self, test_tbl_exprs: List[exprs.Expr], img_tbl_exprs: List[exprs.Expr]
    ) -> None:
        """Test as_dict()/from_dict() (via serialize()/deserialize()) for all exprs."""
        skip_test_if_not_installed('transformers')
        for e in test_tbl_exprs:
            e_serialized = e.serialize()
            e_deserialized = Expr.deserialize(e_serialized)
            assert e.equals(e_deserialized)

        for e in img_tbl_exprs:
            e_serialized = e.serialize()
            e_deserialized = Expr.deserialize(e_serialized)
            assert e.equals(e_deserialized)

    def test_print(self, test_tbl_exprs: List[exprs.Expr], img_tbl_exprs: List[exprs.Expr]) -> None:
        skip_test_if_not_installed('transformers')
        _ = func.FunctionRegistry.get().module_fns
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
        assert len(subexprs) == 6
        subexprs = [s for s in e.subexprs(expr_class=ColumnRef)]
        assert len(subexprs) == 1
        assert t.img.equals(subexprs[0])

    def test_window_fns(self, reset_db, test_tbl: catalog.Table) -> None:
        t = test_tbl
        _ = t.select(sum(t.c2, group_by=t.c4, order_by=t.c3)).show(100)

        # conflicting ordering requirements
        with pytest.raises(excs.Error):
            _ = t.select(sum(t.c2, group_by=t.c4, order_by=t.c3), sum(t.c2, group_by=t.c3, order_by=t.c4)).show(100)
        with pytest.raises(excs.Error):
            _ = t.select(sum(t.c2, group_by=t.c4, order_by=t.c3), sum(t.c2, group_by=t.c3, order_by=t.c4)).show(100)

        # backfill works
        t.add_column(c9=sum(t.c2, group_by=t.c4, order_by=t.c3))
        _ = t.c9.col.has_window_fn_call()

        # ordering conflict between frame extraction and window fn
        base_t = pxt.create_table('videos', {'video': VideoType(), 'c2': IntType(nullable=False)})
        v = pxt.create_view('frame_view', base_t, iterator=FrameIterator.create(video=base_t.video, fps=0))
        # compatible ordering
        _ = v.select(v.frame, sum(v.frame_idx, group_by=base_t, order_by=v.pos)).show(100)
        with pytest.raises(excs.Error):
            # incompatible ordering
            _ = v.select(v.frame, sum(v.c2, order_by=base_t, group_by=v.pos)).show(100)

        schema = {
            'c2': IntType(nullable=False),
            'c3': FloatType(nullable=False),
            'c4': BoolType(nullable=False),
        }
        new_t = pxt.create_table('insert_test', schema)
        new_t.add_column(c2_sum=sum(new_t.c2, group_by=new_t.c4, order_by=new_t.c3))
        rows = list(t.select(t.c2, t.c4, t.c3).collect())
        new_t.insert(rows)
        _ = new_t.show(0)

    def test_make_list(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        import pixeltable.functions as pxtf

        # create a json column with an InlineDict; the type will have a type spec
        t.add_column(json_col={'a': t.c1, 'b': t.c2})
        res = t.select(out=pxtf.json.make_list(t.json_col)).collect()
        assert len(res) == 1
        val = res[0]['out']
        assert len(val) == t.count()
        res2 = t.select(t.json_col).collect()['json_col']
        # need to use frozensets because dicts are not hashable
        assert set(frozenset(d.items()) for d in val) == set(frozenset(d.items()) for d in res2)

    def test_aggregates(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        _ = t[t.c2 % 2, sum(t.c2), count(t.c2), sum(t.c2) + count(t.c2), sum(t.c2) + (t.c2 % 2)]\
            .group_by(t.c2 % 2).show()

        # check that aggregates don't show up in the wrong places
        with pytest.raises(excs.Error):
            # aggregate in where clause
            _ = t[sum(t.c2) > 0][sum(t.c2)].group_by(t.c2 % 2).show()
        with pytest.raises(excs.Error):
            # aggregate in group_by clause
            _ = t[sum(t.c2)].group_by(sum(t.c2)).show()
        with pytest.raises(excs.Error):
            # mixing aggregates and non-aggregates
            _ = t[sum(t.c2) + t.c2].group_by(t.c2 % 2).show()
        with pytest.raises(excs.Error):
            # nested aggregates
            _ = t[sum(count(t.c2))].group_by(t.c2 % 2).show()

    @pxt.uda(
        init_types=[IntType()], update_types=[IntType()], value_type=IntType(),
        allows_window=True, requires_order_by=False)
    class window_agg:
        def __init__(self, val: int = 0):
            self.val = val
        def update(self, ignore: int) -> None:
            pass
        def value(self) -> int:
            return self.val

    @pxt.uda(
        init_types=[IntType()], update_types=[IntType()], value_type=IntType(),
        requires_order_by=True, allows_window=True)
    class ordered_agg:
        def __init__(self, val: int = 0):
            self.val = val
        def update(self, i: int) -> None:
            pass
        def value(self) -> int:
            return self.val

    @pxt.uda(
        init_types=[IntType()], update_types=[IntType()], value_type=IntType(),
        requires_order_by=False, allows_window=False)
    class std_agg:
        def __init__(self, val: int = 0):
            self.val = val
        def update(self, i: int) -> None:
            pass
        def value(self) -> int:
            return self.val

    def test_udas(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        # init arg is passed along
        assert t.select(out=self.window_agg(t.c2, order_by=t.c2)).collect()[0]['out'] == 0
        assert t.select(out=self.window_agg(t.c2, val=1, order_by=t.c2)).collect()[0]['out'] == 1

        with pytest.raises(excs.Error) as exc_info:
            _ = t.select(self.window_agg(t.c2, val=t.c2, order_by=t.c2)).collect()
        assert 'needs to be a constant' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            # ordering expression not a pixeltable expr
            _ = t.select(self.ordered_agg(1, t.c2)).collect()
        assert 'but instead is a' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            # explicit order_by
            _ = t.select(self.ordered_agg(t.c2, order_by=t.c2)).collect()
        assert 'order_by invalid' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            # order_by for non-window function
            _ = t.select(self.std_agg(t.c2, order_by=t.c2)).collect()
        assert 'does not allow windows' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            # group_by for non-window function
            _ = t.select(self.std_agg(t.c2, group_by=t.c4)).collect()
        assert 'group_by invalid' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            # group_by with non-ancestor table
            _ = t.select(t.c2).group_by(t)
        assert 'group_by(): test_tbl is not a base table of test_tbl' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            # group_by with non-singleton table
            _ = t.select(t.c2).group_by(t, t.c2)
        assert 'group_by(): only one table can be specified' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            # missing init type
            @pxt.uda(update_types=[IntType()], value_type=IntType())
            class WindowAgg:
                def __init__(self, val: int = 0):
                    self.val = val
                def update(self, ignore: int) -> None:
                    pass
                def value(self) -> int:
                    return self.val
        assert 'init_types must be a list of' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            # missing update parameter
            @pxt.uda(init_types=[IntType()], update_types=[], value_type=IntType())
            class WindowAgg:
                def __init__(self, val: int = 0):
                    self.val = val
                def update(self) -> None:
                    pass
                def value(self) -> int:
                    return self.val
        assert 'must have at least one parameter' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            # missing update type
            @pxt.uda(init_types=[IntType()], update_types=[IntType()], value_type=IntType())
            class WindowAgg:
                def __init__(self, val: int = 0):
                    self.val = val
                def update(self, i1: int, i2: int) -> None:
                    pass
                def value(self) -> int:
                    return self.val
        assert 'update_types must be a list of' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            # duplicate parameter names
            @pxt.uda(init_types=[IntType()], update_types=[IntType()], value_type=IntType())
            class WindowAgg:
                def __init__(self, val: int = 0):
                    self.val = val
                def update(self, val: int) -> None:
                    pass
                def value(self) -> int:
                    return self.val
        assert 'cannot have parameters with the same name: val' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            # reserved parameter name
            @pxt.uda(init_types=[IntType()], update_types=[IntType()], value_type=IntType())
            class WindowAgg:
                def __init__(self, val: int = 0):
                    self.val = val
                def update(self, order_by: int) -> None:
                    pass
                def value(self) -> int:
                    return self.val
        assert 'order_by is reserved' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            # reserved parameter name
            @pxt.uda(init_types=[IntType()], update_types=[IntType()], value_type=IntType())
            class WindowAgg:
                def __init__(self, val: int = 0):
                    self.val = val
                def update(self, group_by: int) -> None:
                    pass
                def value(self) -> int:
                    return self.val
        assert 'group_by is reserved' in str(exc_info.value).lower()
