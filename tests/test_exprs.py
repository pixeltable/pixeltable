import base64
import json
import math
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import PIL.Image
import pytest
import sqlalchemy as sql

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.functions as pxtf
from pixeltable import catalog, exprs
from pixeltable.exprs import RELATIVE_PATH_ROOT as R, ColumnRef, Expr, Literal
from pixeltable.functions.globals import cast
from pixeltable.iterators import FrameIterator

from .utils import (
    ReloadTester,
    create_all_datatypes_tbl,
    create_scalars_tbl,
    get_image_files,
    reload_catalog,
    skip_test_if_not_installed,
    validate_update_status,
)


class TestExprs:
    @staticmethod
    @pxt.udf
    def div_0_error(a: int, b: int) -> float:
        return a / b

    @staticmethod
    @pxt.udf
    def required_params_fn(a: float, b: float) -> float:
        return a + b

    @staticmethod
    @pxt.udf
    def mixed_params_fn(a: float, b: Optional[float]) -> float:
        if b is None:
            return a
        return a + b

    @staticmethod
    @pxt.udf
    def optional_params_fn(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None:
            return b
        if b is None:
            return a
        return a + b

    # error in agg.init()
    @pxt.uda
    class init_exc(pxt.Aggregator):
        def __init__(self) -> None:
            self.sum = 1 / 0

        def update(self, val: int):
            pass

        def value(self) -> int:
            return 1

    # error in agg.update()
    @pxt.uda
    class update_exc(pxt.Aggregator):
        def __init__(self) -> None:
            self.sum = 0

        def update(self, val: int):
            self.sum += 1 // val

        def value(self) -> int:
            return 1

    # error in agg.value()
    @pxt.uda
    class value_exc(pxt.Aggregator):
        def __init__(self):
            self.sum = 0

        def update(self, val: int):
            self.sum += val

        def value(self) -> float:
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
        sql_elements = pxt.exprs.SqlElementCache()
        e = sql_elements.get(((t.c1 == 'test string') & (t.c2 > 50)))
        assert len(e.clauses) == 2

        e = sql_elements.get(((t.c1 == 'test string') & (t.c2 > 50) & (t.c3 < 1.0)))
        assert len(e.clauses) == 3
        e = sql_elements.get(((t.c1 == 'test string') | (t.c2 > 50)))
        assert len(e.clauses) == 2
        e = sql_elements.get(((t.c1 == 'test string') | (t.c2 > 50) | (t.c3 < 1.0)))
        assert len(e.clauses) == 3
        e = sql_elements.get((~(t.c1 == 'test string')))
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
        _ = t.where(t.c1n != None).collect()
        print(_)

    def test_exception_handling(self, test_tbl: catalog.Table) -> None:
        t = test_tbl

        # TODO(aaron-siegel): I had to comment this out. We can't let division by zero errors
        #     be handled in SQL; this will fail if we have a query whose where clause is a Python
        #     UDF that excludes the rows triggering the error. We'll need to substitute a different
        #     example, or perhaps ensure we avoid SQL errors in the first place.
        # error in expr that's handled in SQL
        # with pytest.raises(excs.Error):
        #     _ = t[(t.c2 + 1) / t.c2].show()

        # error in expr that's handled in Python
        with pytest.raises(excs.Error):
            _ = t.select((t.c6.f2 + 1) / (t.c2 - 10)).show()

        # the same, but with an inline function
        with pytest.raises(excs.Error):
            _ = t.select(self.div_0_error(t.c2 + 1, t.c2)).show()

        # error in agg.init()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.select(self.init_exc(t.c2)).show()
        assert 'division by zero' in str(exc_info.value)

        # error in agg.update()
        with pytest.raises(excs.Error):
            _ = t.select(self.update_exc(t.c2 - 10)).show()

        # error in agg.value()
        with pytest.raises(excs.Error):
            _ = t.where(t.c2 <= 2).select(self.value_exc(t.c2 - 1)).show()

    def test_props(self, test_tbl: catalog.Table, img_tbl: catalog.Table) -> None:
        t = test_tbl
        # errortype/-msg for computed column
        res = t.select(error=t.c8.errortype).collect()
        assert res.to_pandas()['error'].isna().all()
        res = t.select(error=t.c8.errormsg).collect()
        assert res.to_pandas()['error'].isna().all()

        img_t = img_tbl
        # fileurl
        res = img_t.select(img_t.img.fileurl).collect().to_pandas()
        stored_urls = set(res.iloc[:, 0])
        assert len(stored_urls) == len(res)
        all_urls = set(urllib.parse.urljoin('file:', urllib.request.pathname2url(path)) for path in get_image_files())
        assert stored_urls <= all_urls

        # localpath
        res = img_t.select(img_t.img.localpath).collect().to_pandas()
        stored_paths = set(res.iloc[:, 0])
        assert len(stored_paths) == len(res)
        all_paths = set(get_image_files())
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
        img_t.add_computed_column(c9=img_t.img.rotate(30), stored=False)
        with pytest.raises(excs.Error) as excinfo:
            _ = img_t.select(img_t.c9.localpath).show()
        assert 'computed unstored' in str(excinfo.value)
        with pytest.raises(excs.Error) as excinfo:
            _ = img_t.select(img_t.c9.errormsg).show()
        assert 'only valid for' in str(excinfo.value)
        with pytest.raises(excs.Error) as excinfo:
            _ = img_t.select(img_t.c9.errortype).show()
        assert 'only valid for' in str(excinfo.value)

    def test_null_args(self, reset_db) -> None:
        # create table with two columns
        schema = {'c1': pxt.Float, 'c2': pxt.Float}
        t = pxt.create_table('test', schema)

        t.add_computed_column(c3=self.required_params_fn(t.c1, t.c2))
        t.add_computed_column(c4=self.mixed_params_fn(t.c1, t.c2))
        t.add_computed_column(c5=self.optional_params_fn(t.c1, t.c2))

        # data that tests all combinations of nulls
        data: list[dict[str, Any]] = [
            {'c1': 1.0, 'c2': 1.0},
            {'c1': 1.0, 'c2': None},
            {'c1': None, 'c2': 1.0},
            {'c1': None, 'c2': None},
        ]
        validate_update_status(t.insert(data), expected_rows=4)
        result = t.collect()
        assert result['c3'] == [2.0, None, None, None]
        assert result['c4'] == [2.0, 1.0, None, None]
        assert result['c5'] == [2.0, 1.0, 1.0, None]

    def test_arithmetic_exprs(self, test_tbl: catalog.Table) -> None:
        t = test_tbl

        # Add nullable int and float columns
        t.add_column(c2n=pxt.Int)
        t.add_column(c3n=pxt.Float)
        t.where(t.c2 % 7 != 0).update({'c2n': t.c2, 'c3n': t.c3})

        _ = t.select(t.c2, t.c6.f3, t.c2 + t.c6.f3, (t.c2 + t.c6.f3) / (t.c6.f3 + 1)).collect()
        _ = t.select(t.c2 + t.c2).collect()
        for op1, op2 in [(t.c2, t.c2), (t.c3, t.c3), (t.c2, t.c2n), (t.c2n, t.c2)]:
            _ = t.select(op1 + op2).collect()
            _ = t.select(op1 - op2).collect()
            _ = t.select(op1 * op2).collect()
            _ = t.where(op2 > 0).select(op1 / op2).collect()
            _ = t.where(op2 > 0).select(op1 % op2).collect()
            _ = t.where(op2 > 0).select(op1 // op2).collect()

        # non-numeric types
        for op1, op2 in [
            (t.c1, t.c2),
            (t.c1, 1),
            (t.c2, t.c1),
            (t.c2, 'a'),
            (t.c1, t.c3),
            (t.c1, 1.0),
            (t.c3, t.c1),
            (t.c3, 'a'),
        ]:
            with pytest.raises(excs.Error):
                _ = t.select(op1 + op2).collect()
            with pytest.raises(excs.Error):
                _ = t.select(op1 - op2).collect()
            with pytest.raises(excs.Error):
                _ = t.select(op1 * op2).collect()
            with pytest.raises(excs.Error):
                _ = t.select(op1 / op2).collect()
            with pytest.raises(excs.Error):
                _ = t.select(op1 % op2).collect()
            with pytest.raises(excs.Error):
                _ = t.select(op1 // op2).collect()

        # TODO: test division; requires predicate
        for op1, op2 in [(t.c6.f2, t.c6.f2), (t.c6.f3, t.c6.f3)]:
            _ = t.select(op1 + op2).collect()
            _ = t.select(op1 - op2).collect()
            _ = t.select(op1 * op2).collect()
            with pytest.raises(excs.Error):
                _ = t.select(op1 / op2).collect()

        for op1, op2 in [
            (t.c6.f1, t.c6.f2),
            (t.c6.f1, t.c6.f3),
            (t.c6.f1, 1),
            (t.c6.f1, 1.0),
            (t.c6.f2, t.c6.f1),
            (t.c6.f3, t.c6.f1),
            (t.c6.f2, 'a'),
            (t.c6.f3, 'a'),
        ]:
            with pytest.raises(excs.Error):
                _ = t.select(op1 + op2).collect()
            with pytest.raises(excs.Error):
                _ = t.select(op1 - op2).collect()
            with pytest.raises(excs.Error):
                _ = t.select(op1 * op2).collect()

        # Test literal exprs
        results = (
            t.where(t.c2 == 7)
            .select(
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
            )
            .collect()
        )
        assert list(results[0].values()) == [-7, 9, 5, 14, 3.5, 1, 3, 9, -5, 14, 0.2857142857142857, 2, 0]

        # Test that arithmetic operations give the right answers. We do this two ways:
        # (i) with primitive operators only, to ensure that the arithmetic operations are done in SQL when possible;
        # (ii) with a Python function call interposed, to ensure that the arithmetic operations are always done in Python;
        # (iii) and (iv), as (i) and (ii) but with JsonType expressions.
        primitive_ops = (t.c2, t.c3)
        forced_python_ops = (t.c2.apply(math.floor, col_type=pxt.Int), t.c3.apply(math.floor, col_type=pxt.Float))
        json_primitive_ops = (t.c6.f2, t.c6.f3)
        json_forced_python_ops = (
            t.c6.f2.apply(math.floor, col_type=pxt.Int),
            t.c6.f3.apply(math.floor, col_type=pxt.Float),
        )
        for int_operand, float_operand in (
            primitive_ops,
            forced_python_ops,
            json_primitive_ops,
            json_forced_python_ops,
        ):
            results = (
                t.where(t.c2 == 7)
                .select(
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
                )
                .collect()
            )
            assert list(results[0].values()) == [
                10,
                4,
                21,
                2.3333333333333335,
                1,
                -3,
                10.0,
                4.0,
                21.0,
                2.3333333333333335,
                1.0,
                2.0,
                -3.0,
                None,
                None,
            ], f'Failed with operands: {int_operand}, {float_operand}'

        with pytest.raises(excs.Error) as exc_info:
            t.select(t.c6 + t.c2.apply(math.floor, col_type=pxt.Int)).collect()
        assert '+ requires numeric types, but c6 has type dict' in str(exc_info.value)

    def test_comparison(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        # Test that comparison operations give the right answers. As with arithmetic operations, we do this two ways:
        # (i) with primitive operators only, to ensure that the comparison operations are done in SQL when possible;
        # (ii) with a Python function call interposed, to ensure that the comparison operations are always done in Python.
        comparison_pairs = (
            (t.c1, 'test string 10'),  # string-to-string
            (t.c2, 50),  # int-to-int
            (t.c3, 50.1),  # float-to-float
            (t.c5, datetime(2024, 7, 2)),  # datetime-to-datetime
        )
        for expr1, expr2 in comparison_pairs:
            forced_expr1 = expr1.apply(lambda x: x, col_type=expr1.col_type)
            for a_expr, b_expr in ((expr1, expr2), (expr2, expr1), (forced_expr1, expr2), (expr2, forced_expr1)):
                results = t.select(
                    a=a_expr,
                    b=b_expr,
                    eq=a_expr == b_expr,
                    ne=a_expr != b_expr,
                    # One or the other of a_expr or b_expr will always be an Expr, but mypy doesn't understand that
                    lt=a_expr < b_expr,  # type: ignore[operator]
                    le=a_expr <= b_expr,  # type: ignore[operator]
                    gt=a_expr > b_expr,  # type: ignore[operator]
                    ge=a_expr >= b_expr,  # type: ignore[operator]
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
        df = t.select({'a': t.c1, 'b': {'c': t.c2}, 'd': 1, 'e': {'f': 2}})
        result = df.show()
        print(result)

    def test_constant_literals(self, test_tbl: catalog.Table, reload_tester: ReloadTester) -> None:
        t = test_tbl
        t.add_computed_column(cc0=datetime.now())  # timestamp
        t.add_computed_column(cc1=100)  # integer
        t.add_computed_column(cc2='abc')  # string
        t.add_computed_column(cc3=10.4)  # floating point
        t.add_computed_column(cc4=(100, 200))  # tuple of integer
        t.add_computed_column(cc5={'a': 'str100', 'b': 3.14, 'c': [1, 2, 3], 'd': {'e': (0.99, 100.1)}})
        t.add_computed_column(cc6=pxt.array([100.1, 200.1, 300.1]))  # one dimensional floating point array
        t.add_computed_column(cc7=pxt.array(['abc', 'bcd', 'efg']))  # one dimensional string array
        # list if list (integers)
        t.add_computed_column(
            cc8=[[[1, 2, 3], [4, 5, 6]], [[10, 20, 30], [40, 50, 60]], [[100, 200, 300], [400, 500, 600]]]
        )
        # multidimensional string arrays
        t.add_computed_column(
            cc9=pxt.array(
                [
                    [['a1', 'b2', 'c3'], ['a4', 'b5', 'c6']],
                    [['a10', 'b20', 'c30'], ['a40', 'b50', 'c60']],
                    [['a100', 'b200', 'c300'], ['a400', 'b500', 'c600']],
                ]
            )
        )
        results = reload_tester.run_query(
            t.select(t.cc0, t.cc1, t.cc2, t.cc3, t.cc4, t.cc5, t.cc6, t.cc7, t.cc8, t.cc9)
        )
        print(results.schema)
        reload_tester.run_reload_test()

    def test_inline_constants(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        result = t.select([1, 2, 3])
        print(result.show(5))
        assert isinstance(result.select_list[0][0], Literal)

        arr1 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        arr2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)

        # r0 = t.select(None)
        # print(r0.show(5))

        r1 = t.select(pxt.array(arr1))
        print(r1.show(5))

        arr3 = pxt.array([1, 2, 3, 4, 5])
        r2 = t.select(arr3)
        print(r2.show(5))

        arr4 = pxt.array(np.array([1, 2, 3, 4, 5], dtype=np.int64))
        r3 = t.select(arr4)
        print(r3.show(5))

        result = t.select(
            1,
            (100, 100),
            [200, 200],
            # This will produce a Json type literal object
            ['a', 'b', 'c'],
            # This is an np.array, dtype='<U1' : col_type = StringType
            pxt.array(['a', 'b', 'c']),
            # This is an np.array, dtype='<U7' : col_type = StringType
            pxt.array(['abc', 'd', 'efghijk']),
            arr1,
            arr2,
            {'b': [4, 5]},
            {'c': {}},
            {'d': {'d': 6, 'e': [7, 8], 'f': {}, 'g': {'h': 9}}},
        )
        print(result.show(5))
        exprs = [expr[0] for expr in result.select_list]
        for e in exprs:
            assert isinstance(e, Literal)

        result = t.select(
            1, (100, 100), {'a': [t.c1, 3]}, {'b': [4, 5]}, {'c': {'d': 6, 'e': [7, 8], 'f': {}, 'g': {'h': t.c2}}}
        )
        print(result.show(5))
        exprs = [expr[0] for expr in result.select_list]
        assert isinstance(exprs[0], Literal)
        assert isinstance(exprs[1], Literal)
        assert not isinstance(exprs[2], Literal)
        assert isinstance(exprs[3], Literal)
        assert not isinstance(exprs[4], Literal)

        result = t.select(
            1,
            (100, 100),
            {'a': [t.c1, 3]},
            {'b': [4, 5]},
            {'c': {'d': 6, 'e': [7, 8], 'f': (t.c1, t.c3), 'g': {'h': 9}}},
            {'d': t.c1},
        )
        print(result.show(5))
        exprs = [expr[0] for expr in result.select_list]
        assert isinstance(exprs[0], Literal)
        assert isinstance(exprs[1], Literal)
        assert not isinstance(exprs[2], Literal)
        assert isinstance(exprs[3], Literal)
        assert not isinstance(exprs[4], Literal)
        assert not isinstance(exprs[5], Literal)

        result = t.select(pxt.array([[1, 2, 3], [4, 5, 6]]))
        print(result.show(5))
        exprs = [expr[0] for expr in result.select_list]
        assert isinstance(exprs[0], Literal)
        col_type = next(iter(result.schema.values()))
        assert col_type.is_array_type()
        assert isinstance(col_type, pxt.ArrayType)

    def test_inline_array(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        result = t.select(pxt.array([[t.c2, 1], [t.c2, 2]])).show()
        col_type = next(iter(result.schema.values()))
        assert col_type.is_array_type()
        assert isinstance(col_type, pxt.ArrayType)
        assert col_type.shape == (2, 2)
        assert col_type.dtype == pxt.ColumnType.Type.INT

        with pytest.raises(excs.Error) as excinfo:
            _ = t.select(pxt.array([t.c1, t.c2])).collect()
        assert 'element of type `Int` at index 1 is not compatible with type `String` of preceding elements' in str(
            excinfo.value
        )

    def test_json_path(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        t.add_computed_column(attr=t.c6.f5)
        t.add_computed_column(item=t['c6']['f5'])
        t.add_computed_column(index=t['c6'].f5[2])
        t.add_computed_column(slice_all=t.c6.f5[:])
        t.add_computed_column(slice_to=t.c6.f5[:7])
        t.add_computed_column(slice_from=t.c6.f5[3:])
        t.add_computed_column(slice_range=t.c6.f5[3:7])
        t.add_computed_column(slice_range_step=t.c6.f5[3:7:2])
        t.add_computed_column(slice_range_step_item=t['c6'].f5[3:7:2])
        res = t.collect()
        orig = res['attr']
        assert all(res['item'][i] == orig[i] for i in range(len(res)))
        assert all(res['index'][i] == orig[i][2] for i in range(len(res)))
        assert all(res['slice_all'][i] == orig[i] for i in range(len(orig)))
        assert all(res['slice_to'][i] == orig[i][:7] for i in range(len(orig)))
        assert all(res['slice_from'][i] == orig[i][3:] for i in range(len(orig)))
        assert all(res['slice_range'][i] == orig[i][3:7] for i in range(len(orig)))
        assert all(res['slice_range_step'][i] == orig[i][3:7:2] for i in range(len(orig)))
        assert all(res['slice_range_step_item'][i] == orig[i][3:7:2] for i in range(len(orig)))

    def test_json_mapper(self, test_tbl: catalog.Table, reload_tester: ReloadTester) -> None:
        t = test_tbl

        # top-level is dict
        res = reload_tester.run_query(t.select(input=t.c6.f5, output=t.c6.f5['*'] >> (R + 1)))
        for row in res:
            assert row['output'] == [x + 1 for x in row['input']]

        # top-level is list of dicts; subsequent json path element references the dicts
        res = reload_tester.run_query(t.select(input=t.c7, output=t.c7['*'].f5 >> [R[3], R[2], R[1], R[0]]))
        for row in res:
            assert row['output'] == [[d['f5'][3], d['f5'][2], d['f5'][1], d['f5'][0]] for d in row['input']]

        # target expr contains global-scope dependency
        res = reload_tester.run_query(t.select(input=t.c6, output=t.c6.f5['*'] >> (R * t.c6.f5[1])))
        for row in res:
            assert row['output'] == [x * row['input']['f5'][1] for x in row['input']['f5']]

        # test it as a computed column
        validate_update_status(t.add_computed_column(output=t.c6.f5['*'] >> (R * t.c6.f5[1])), 100)
        res2 = reload_tester.run_query(t.select(t.output))
        for row, row2 in zip(res, res2):
            assert row['output'] == row2['output']

        reload_tester.run_reload_test()

    def test_multi_json_mapper(self, reset_db, reload_tester: ReloadTester) -> None:
        # Workflow with multiple JsonMapper instances
        t = pxt.create_table('test', {'jcol': pxt.Json})
        t.add_computed_column(outputx=t.jcol.x['*'] >> (R + 1))
        t.add_computed_column(outputy=t.jcol.y['*'] >> (R + 2))
        t.add_computed_column(outputz=t.jcol.z['*'] >> (R + 3))
        for i in range(8):
            data = {}
            if (i & 1) != 0:
                data['x'] = [1, 2, 3]
            if (i & 2) != 0:
                data['y'] = [4, 5, 6]
            if (i & 4) != 0:
                data['z'] = [7, 8, 9]
            t.insert(jcol=data)
        res = reload_tester.run_query(t.select(t.outputx, t.outputy, t.outputz))
        for i in range(8):
            print(res[i])
            assert res[i]['outputx'] == (None if (i & 1) == 0 else [2, 3, 4])
            assert res[i]['outputy'] == (None if (i & 2) == 0 else [6, 7, 8])
            assert res[i]['outputz'] == (None if (i & 4) == 0 else [10, 11, 12])

        reload_tester.run_reload_test()

    def test_dicts(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        # top-level is dict
        _ = t.select(t.c6.f1)
        _ = _.show()
        print(_)
        # predicate on dict field
        _ = t.select(t.c6.f2 < 2).show()
        # _ = t[t.c6.f2].show()
        # _ = t[t.c6.f5].show()
        _ = t.select(t.c6.f6.f8).show()
        _ = t.select(cast(t.c6.f6.f8, pxt.Array[(4,), pxt.Float])).show()  # type: ignore[misc]

        # top-level is array
        # _ = t[t.c7['*'].f1].show()
        # _ = t[t.c7['*'].f2].show()
        # _ = t.select(t.c7['*'].f5).show()
        _ = t.select(t.c7['*'].f6.f8).show()
        _ = t.select(t.c7[0].f6.f8).show()
        _ = t.select(t.c7[:2].f6.f8).show()
        _ = t.select(t.c7[::-1].f6.f8).show()
        _ = t.select(cast(t.c7['*'].f6.f8, pxt.Array[(2, 4), pxt.Float])).show()  # type: ignore[misc]
        print(_)

    def test_arrays(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        t.add_computed_column(array_col=pxt.array([[t.c2, 1], [5, t.c2]]))

        def selection_equals(expr: Expr, expected: list[np.ndarray]) -> bool:
            return all(np.array_equal(x, y) for x, y in zip(t.select(out=expr).collect()['out'], expected))

        assert selection_equals(t.array_col, [np.array([[i, 1], [5, i]]) for i in range(100)])
        assert selection_equals(t.array_col[1], [np.array([5, i]) for i in range(100)])
        assert selection_equals(t.array_col[:, 0], [np.array([i, 5]) for i in range(100)])

        with pytest.raises(AttributeError) as excinfo:
            t.array_col[1, 'string']
        assert 'Invalid array indices' in str(excinfo.value)

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
        assert 'c1 has type String' in str(excinfo.value)

        status = t.add_computed_column(in_test=t.c2.isin([1, 2, 3]))
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
        validate_update_status(t.add_computed_column(c2_as_float=t.c2.astype(pxt.Float)))
        assert t.c2_as_float.col_type == pxt.FloatType(nullable=False)
        data = t.select(t.c2, t.c2_as_float).collect()
        for row in data:
            assert isinstance(row['c2'], int)
            assert isinstance(row['c2_as_float'], float)
            assert row['c2'] == row['c2_as_float']

        # Compound expression
        validate_update_status(t.add_computed_column(compound_as_float=(t.c2 + 1).astype(pxt.Float)))
        assert t.compound_as_float.col_type == pxt.FloatType(nullable=False)
        data = t.select(t.c2, t.compound_as_float).collect()
        for row in data:
            assert isinstance(row['compound_as_float'], float)
            assert row['c2'] + 1 == row['compound_as_float']

        # Type conversion error
        status = t.add_computed_column(c2_as_string=t.c2.astype(pxt.String), on_error='ignore')
        assert status.num_excs == t.count()
        errormsgs = t.select(out=t.c2_as_string.errormsg).collect()['out']
        assert all('Expected string, got int' in msg for msg in errormsgs), errormsgs

        # Convert a nullable column
        validate_update_status(t.add_column(c2n=pxt.Int))
        t.where(t.c2 % 2 == 0).update({'c2n': t.c2})  # set even values; keep odd values as None
        validate_update_status(t.add_computed_column(c2n_as_float=t.c2n.astype(pxt.Float)))
        assert t.c2n_as_float.col_type == pxt.FloatType(nullable=True)

        # Cast nullable to required
        status = t.add_computed_column(c2n_as_req_float=t.c2n.astype(pxt.Required[pxt.Float]), on_error='ignore')
        assert t.c2n_as_req_float.col_type == pxt.FloatType(nullable=False)
        assert status.num_excs == t.count() // 2  # Just the odd values should error out
        errormsgs = [msg for msg in t.select(out=t.c2n_as_req_float.errormsg).collect()['out'] if msg is not None]
        assert len(errormsgs) == t.count() // 2
        assert all('Expected non-None value' in msg for msg in errormsgs), errormsgs

    def test_astype_str_to_img(self, reset_db) -> None:
        img_files = get_image_files()
        img_files = img_files[:5]
        # store relative paths in the table
        parent_dir = Path(img_files[0]).parent
        assert all(parent_dir == Path(img_file).parent for img_file in img_files)
        t = pxt.create_table('astype_test', {'rel_path': pxt.String})
        validate_update_status(t.insert({'rel_path': Path(f).name} for f in img_files), expected_rows=len(img_files))

        # create a computed image column constructed from the relative paths
        import pixeltable.functions as pxtf

        validate_update_status(
            t.add_computed_column(
                img=pxtf.string.format('{0}/{1}', str(parent_dir), t.rel_path).astype(pxt.Image), stored=True
            )
        )
        loaded_imgs = t.select(t.img).collect()['img']
        orig_imgs = [PIL.Image.open(f) for f in img_files]
        for orig_img, retrieved_img in zip(orig_imgs, loaded_imgs):
            assert np.array_equal(np.array(orig_img), np.array(retrieved_img))

        # the same for a select list item
        loaded_imgs = t.select(
            img=pxtf.string.format('{0}/{1}', str(parent_dir), t.rel_path).astype(pxt.Image)
        ).collect()['img']
        for orig_img, retrieved_img in zip(orig_imgs, loaded_imgs):
            assert np.array_equal(np.array(orig_img), np.array(retrieved_img))

    def test_astype_str_to_img_data_url(self, reset_db) -> None:
        t = pxt.create_table('astype_test', {'url': pxt.String})
        t.add_computed_column(img=t.url.astype(pxt.Image))
        images = get_image_files(include_bad_image=True)[:5]  # bad image is at idx 0
        url_encoded_images = [
            f'data:image/jpeg;base64,{base64.b64encode(open(img, "rb").read()).decode()}' for img in images
        ]

        status = t.insert({'url': url} for url in url_encoded_images[1:])
        validate_update_status(status, expected_rows=4)

        loaded_imgs = t.select(t.img).head(4)['img']
        for image_file, retrieved_img in zip(images[1:], loaded_imgs):
            orig_img = PIL.Image.open(image_file)
            orig_img.load()
            assert orig_img.size == retrieved_img.size

        # Try inserting a non-image
        with pytest.raises(
            excs.ExprEvalError, match='data URL could not be decoded into a valid image: data:text/plain,Hello there.'
        ):
            t.insert(url='data:text/plain,Hello there.')

        # Try inserting a bad image
        with pytest.raises(
            excs.ExprEvalError,
            match='data URL could not be decoded into a valid image: data:image/jpeg;base64,dGhlc2UgYXJlIHNvbWUgYmFkIGp...',
        ):
            t.insert(url=url_encoded_images[0])

    def test_apply(self, test_tbl: catalog.Table) -> None:
        t = test_tbl

        # For each column c1, ..., c5, we create a new column ci_as_str that converts it to
        # a string, then check that each row is correctly converted
        # (For c1 this is the no-op string-to-string conversion)
        for col_id in range(1, 6):
            col_name = f'c{col_id}'
            str_col_name = f'c{col_id}_str'
            status = t.add_computed_column(**{str_col_name: t[col_name].apply(str)})
            assert status.num_excs == 0
            data = t.select(t[col_name], t[str_col_name]).collect()
            for row in data:
                assert row[str_col_name] == str(row[col_name])

        # Test a compound expression with apply
        status = t.add_computed_column(c2_plus_1_str=(t.c2 + 1).apply(str))
        assert status.num_excs == 0
        data = t.select(t.c2, t.c2_plus_1_str).collect()
        for row in data:
            assert row['c2_plus_1_str'] == str(row['c2'] + 1)

        # For columns c6, c7, try using json.dumps and json.loads to emit and parse JSON <-> str
        for col_id in range(6, 8):
            col_name = f'c{col_id}'
            str_col_name = f'c{col_id}_str'
            back_to_json_col_name = f'c{col_id}_back_to_json'
            status = t.add_computed_column(**{str_col_name: t[col_name].apply(json.dumps)})
            assert status.num_excs == 0
            status = t.add_computed_column(**{back_to_json_col_name: t[str_col_name].apply(json.loads)})
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
        status = t.add_computed_column(c2_str_f1=t.c2.apply(f1, col_type=pxt.String))
        assert status.num_excs == 0

        # Test that the return type of a function can be successfully inferred.
        def f2(x) -> str:
            return str(x)

        status = t.add_computed_column(c2_str_f2=t.c2.apply(f2))
        assert status.num_excs == 0

        # Test various validation failures.

        def f3(x, y) -> str:
            return f'{x}{y}'

        with pytest.raises(excs.Error) as exc_info:
            t.c2.apply(f3)  # Too many required parameters
        assert str(exc_info.value) == 'Function `f3` has multiple required parameters.'

        def f4() -> str:
            return 'pixeltable'

        with pytest.raises(excs.Error) as exc_info:
            t.c2.apply(f4)  # No positional parameters
        assert str(exc_info.value) == 'Function `f4` has no positional parameters.'

        def f5(**kwargs) -> str:
            return ''

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
        result = t.select(t.img).show(n=100)
        _ = result._repr_html_()
        df = t.select(t.img, t.img.rotate(60))
        _ = df.show(n=100)._repr_html_()

        with pytest.raises(excs.Error):
            _ = t.select(t.img.rotate)

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
            'https://raw.github.com/pixeltable/pixeltable/main/docs/resources/images/000000000030.jpg',
            'https://raw.github.com/pixeltable/pixeltable/main/docs/resources/images/000000000034.jpg',
            'https://raw.github.com/pixeltable/pixeltable/main/docs/resources/images/000000000042.jpg',
            'https://raw.github.com/pixeltable/pixeltable/main/docs/resources/images/000000000049.jpg',
            'https://raw.github.com/pixeltable/pixeltable/main/docs/resources/images/000000000057.jpg',
            'https://raw.github.com/pixeltable/pixeltable/main/docs/resources/images/000000000061.jpg',
            'https://raw.github.com/pixeltable/pixeltable/main/docs/resources/images/000000000063.jpg',
            'https://raw.github.com/pixeltable/pixeltable/main/docs/resources/images/000000000064.jpg',
            'https://raw.github.com/pixeltable/pixeltable/main/docs/resources/images/000000000069.jpg',
            'https://raw.github.com/pixeltable/pixeltable/main/docs/resources/images/000000000071.jpg',
        ]
        t.insert({'img': url} for url in img_urls)
        # this fails with an assertion
        # TODO: fix it
        # res = t.where(t.img.width < 600).collect()

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
        result = (
            t.where((t.split == 'train') & (t.img.entropy() > 1) & (t.split == 'val') & (t.img.entropy() < 0))
            .select(t.img, t.split)
            .show()
        )
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
        result = t.select(t.img.nearest(img) & (t.category == probe[0, 1]) & (t.img.width > 1)).show(10)
        # TODO: figure out how to verify results

        with pytest.raises(excs.Error) as exc_info:
            _ = t.select(t.img.nearest(img)).order_by(t.category).show()
        assert 'cannot be used in conjunction with' in str(exc_info.value)

        result = t.select(t.img.nearest('musical instrument')).show(10)
        assert len(result) == 10
        # matches() with one SQL predicate and one Python predicate
        french_horn_category = 'n03394916'
        result = t[t.img.nearest('musical instrument') & (t.category == french_horn_category) & (t.img.width > 1)].show(
            10
        )

        with pytest.raises(excs.Error) as exc_info:
            _ = t.select(t.img.nearest(5)).show()
        assert 'requires' in str(exc_info.value)

    # TODO: this doesn't work when combined with test_similarity(), for some reason the data table for img_tbl
    # doesn't get created; why?
    def test_similarity2(
        self, img_tbl: catalog.Table, indexed_img_tbl: catalog.Table, multi_idx_img_tbl: catalog.Table
    ) -> None:
        t = img_tbl
        probe = t.select(t.img).show(1)
        img = probe[0, 0]

        with pytest.raises(AttributeError):
            _ = t.select(t.img.nearest(img)).show(10)
        with pytest.raises(AttributeError):
            _ = t.select(t.img.nearest('musical instrument')).show(10)

        t1 = indexed_img_tbl
        # for a table with a single embedding index, whether we
        # specify the index or not, the similarity expression
        # would use that index. So these exressions should be equivalent.
        sim1 = t1.img.similarity('red truck')
        sim2 = t1.img.similarity('red truck', idx='img_idx0')
        assert sim1.id == sim2.id
        assert sim1.serialize() == sim2.serialize()

        t2 = multi_idx_img_tbl
        # for a table with multiple embedding indexes, the index
        # to use must be specified to the similarity expression.
        # So similarity expressions using different indexes should differ.
        sim1 = t2.img.similarity('red truck', idx='img_idx1')
        sim2 = t2.img.similarity('red truck', idx='img_idx2')
        assert sim1.id != sim2.id
        assert sim1.serialize() != sim2.serialize()

    def test_ids(
        self, test_tbl_exprs: list[exprs.Expr], img_tbl_exprs: list[exprs.Expr], multi_img_tbl_exprs: list[exprs.Expr]
    ) -> None:
        skip_test_if_not_installed('transformers')
        d: dict[int, exprs.Expr] = {}
        for e in test_tbl_exprs + img_tbl_exprs + multi_img_tbl_exprs:
            assert e.id is not None
            d[e.id] = e
        assert len(d) == len(test_tbl_exprs) + len(img_tbl_exprs) + len(multi_img_tbl_exprs)

    def test_serialization(
        self, test_tbl_exprs: list[exprs.Expr], img_tbl_exprs: list[exprs.Expr], multi_img_tbl_exprs: list[exprs.Expr]
    ) -> None:
        """Test as_dict()/from_dict() (via serialize()/deserialize()) for all exprs."""
        skip_test_if_not_installed('transformers')
        for e in test_tbl_exprs + img_tbl_exprs + multi_img_tbl_exprs:
            e_serialized = e.serialize()
            e_deserialized = Expr.deserialize(e_serialized)
            assert e.equals(e_deserialized)

    def test_print(
        self, test_tbl_exprs: list[exprs.Expr], img_tbl_exprs: list[exprs.Expr], multi_img_tbl_exprs: list[exprs.Expr]
    ) -> None:
        skip_test_if_not_installed('transformers')
        _ = pxt.func.FunctionRegistry.get().module_fns
        for e in test_tbl_exprs + img_tbl_exprs + multi_img_tbl_exprs:
            _ = str(e)
            print(_)

    def test_subexprs(self, img_tbl: catalog.Table) -> None:
        t = img_tbl
        e = t.img
        subexprs = [s for s in e.subexprs()]
        assert len(subexprs) == 1
        e = t.img.rotate(90).resize((224, 224))
        subexprs = [s for s in e.subexprs()]
        assert len(subexprs) == 5
        subexprs = [s for s in e.subexprs(expr_class=ColumnRef)]
        assert len(subexprs) == 1
        assert t.img.equals(subexprs[0])

    def test_window_fns(self, reset_db, test_tbl: catalog.Table) -> None:
        t = test_tbl
        _ = t.select(pxtf.sum(t.c2, group_by=t.c4, order_by=t.c3)).show(100)

        # conflicting ordering requirements
        with pytest.raises(excs.Error):
            _ = t.select(
                pxtf.sum(t.c2, group_by=t.c4, order_by=t.c3), pxtf.sum(t.c2, group_by=t.c3, order_by=t.c4)
            ).show(100)
        with pytest.raises(excs.Error):
            _ = t.select(
                pxtf.sum(t.c2, group_by=t.c4, order_by=t.c3), pxtf.sum(t.c2, group_by=t.c3, order_by=t.c4)
            ).show(100)

        # backfill works
        t.add_computed_column(c9=pxtf.sum(t.c2, group_by=t.c4, order_by=t.c3))
        _ = t.c9.col.has_window_fn_call()

        # ordering conflict between frame extraction and window fn
        base_t = pxt.create_table('videos', {'video': pxt.Video, 'c2': pxt.Int})
        v = pxt.create_view('frame_view', base_t, iterator=FrameIterator.create(video=base_t.video, fps=0))
        # compatible ordering
        _ = v.select(v.frame, pxtf.sum(v.frame_idx, group_by=base_t, order_by=v.pos)).show(100)
        with pytest.raises(excs.Error):
            # incompatible ordering
            _ = v.select(v.frame, pxtf.sum(v.c2, order_by=base_t, group_by=v.pos)).show(100)

        schema = {'c2': pxt.Int, 'c3': pxt.Float, 'c4': pxt.Bool}
        new_t = pxt.create_table('insert_test', schema)
        new_t.add_computed_column(c2_sum=pxtf.sum(new_t.c2, group_by=new_t.c4, order_by=new_t.c3))
        rows = list(t.select(t.c2, t.c4, t.c3).collect())
        new_t.insert(rows)
        _ = new_t.collect()

    def test_make_list(self, test_tbl: catalog.Table) -> None:
        t = test_tbl

        # create a json column with an InlineDict; the type will have a type spec
        t.add_computed_column(json_col={'a': t.c1, 'b': t.c2})
        res = t.select(out=pxtf.json.make_list(t.json_col)).collect()
        assert len(res) == 1
        val = res[0]['out']
        assert len(val) == t.count()
        res2 = t.select(t.json_col).collect()['json_col']
        # need to use frozensets because dicts are not hashable
        assert set(frozenset(d.items()) for d in val) == set(frozenset(d.items()) for d in res2)

    def test_agg(self, reset_db) -> None:
        t = create_scalars_tbl(1000)
        df = t.select().collect().to_pandas()

        def series_to_list(series):
            return [int(x) if pd.notna(x) else None for x in series]

        int_sum: Expr = pxtf.sum(t.c_int)
        _ = t.group_by(t.c_int).select(t.c_int, out=int_sum).order_by(int_sum, asc=False).limit(5).collect()

        for pxt_fn, pd_fn in [
            (pxtf.sum, 'sum'),
            (pxtf.mean, 'mean'),
            (pxtf.min, 'min'),
            (pxtf.max, 'max'),
            (pxtf.count, 'count'),
        ]:
            pxt_sql_result = t.group_by(t.c_int).select(out=pxt_fn(t.c_int)).order_by(t.c_int).collect()
            pxt_py_result = (
                t.group_by(t.c_int)
                # apply(): force execution in Python
                .select(out=pxt_fn(t.c_int.apply(lambda x: x, col_type=t.c_int.col_type)))
                .order_by(t.c_int)
                .collect()
            )
            pd_result = (
                df.groupby('c_int', dropna=False)
                .agg(out=('c_int', pd_fn))
                .reset_index()
                .sort_values('c_int', na_position='last')
            )
            if pd_fn != 'sum':
                # pandas doesn't return NaN for sum of NaNs
                assert pxt_sql_result['out'] == series_to_list(pd_result['out']), pd_fn
            assert pxt_py_result['out'] == pxt_sql_result['out'], pd_fn

        # agg with order-by is currently not supported on the Python path
        for pxt_fn, pd_fn in [(pxtf.mean, 'mean'), (pxtf.min, 'min'), (pxtf.max, 'max'), (pxtf.count, 'count')]:
            int_agg = pxt_fn(t.c_int)
            pxt_sql_result = (
                t.where(t.c_int != None)
                .group_by(t.c_int)
                .select(out=int_agg)
                .order_by(int_agg, asc=False)
                .limit(5)
                .collect()
            )
            pd_result = (
                df.groupby('c_int')
                .agg(out=('c_int', pd_fn))
                .nlargest(5, 'out')
                .reset_index()
                .sort_values('out', ascending=False)
            )
            assert pxt_sql_result['out'] == series_to_list(pd_result['out']), pd_fn

        # sum()
        pxt_sql_result = (
            t.where(t.c_int != None).group_by(t.c_int).select(out=pxtf.sum(t.c_int)).order_by(t.c_int).collect()
        )
        pd_result = df.groupby('c_int', dropna=True).agg(out=('c_int', 'sum')).reset_index().sort_values('c_int')
        assert pxt_sql_result['out'] == series_to_list(pd_result['out'])

    def test_agg_errors(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        from pixeltable.functions import count, sum

        # check that aggregates don't show up in the wrong places
        with pytest.raises(excs.Error):
            # aggregate in where clause
            _ = t.group_by(t.c2 % 2).where(sum(t.c2) > 0).select(sum(t.c2)).collect()
        with pytest.raises(excs.Error):
            # aggregate in group_by clause
            _ = t.group_by(sum(t.c2)).select(sum(t.c2)).collect()
        with pytest.raises(excs.Error):
            # mixing aggregates and non-aggregates
            _ = t.group_by(t.c2 % 2).select(sum(t.c2) + t.c2).collect()
        with pytest.raises(excs.Error):
            # nested aggregates
            _ = t.group_by(t.c2 % 2).select(sum(count(t.c2))).collect()

    def test_function_call_errors(self, test_tbl: pxt.Table) -> None:
        t = test_tbl
        with pytest.raises(
            excs.Error, match="Argument 2 in call to 'tests.test_exprs.udf1' is not a valid Pixeltable expression"
        ):
            udf1(t.c2, bool)
        with pytest.raises(
            excs.Error, match="Argument 'eggs' in call to 'tests.test_exprs.udf1' is not a valid Pixeltable expression"
        ):
            udf1(eggs=bool)

    @pxt.uda(allows_window=True, requires_order_by=False)
    class window_agg(pxt.Aggregator):
        def __init__(self, val: int = 0):
            self.val = val

        def update(self, ignore: int) -> None:
            pass

        def value(self) -> int:
            return self.val

    @pxt.uda(requires_order_by=True, allows_window=True)
    class ordered_agg(pxt.Aggregator):
        def __init__(self, val: int = 0):
            self.val = val

        def update(self, i: int) -> None:
            pass

        def value(self) -> int:
            return self.val

    @pxt.uda(requires_order_by=False, allows_window=False)
    class std_agg(pxt.Aggregator):
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
        assert 'must be a constant value' in str(exc_info.value)

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
            # missing update parameter
            @pxt.uda
            class WindowAgg1(pxt.Aggregator):
                def __init__(self, val: int = 0):
                    self.val = val

                def update(self) -> None:
                    pass

                def value(self) -> int:
                    return self.val

        assert 'must have at least one parameter' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            # duplicate parameter names
            @pxt.uda
            class WindowAgg2(pxt.Aggregator):
                def __init__(self, val: int = 0):
                    self.val = val

                def update(self, val: int) -> None:
                    pass

                def value(self) -> int:
                    return self.val

        assert 'cannot have parameters with the same name: val' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            # reserved parameter name
            @pxt.uda
            class WindowAgg3(pxt.Aggregator):
                def __init__(self, val: int = 0):
                    self.val = val

                def update(self, order_by: int) -> None:
                    pass

                def value(self) -> int:
                    return self.val

        assert "'order_by' is a reserved parameter name" in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            # reserved parameter name
            @pxt.uda
            class WindowAgg4(pxt.Aggregator):
                def __init__(self, val: int = 0):
                    self.val = val

                def update(self, group_by: int) -> None:
                    pass

                def value(self) -> int:
                    return self.val

        assert "'group_by' is a reserved parameter name" in str(exc_info.value).lower()

    def test_repr(self, reset_db) -> None:
        t = create_all_datatypes_tbl()
        instances: list[tuple[exprs.Expr, str]] = [
            # ArithmeticExpr
            (t.c_int + 5, 'c_int + 5'),
            (t.c_int - 5, 'c_int - 5'),
            (t.c_int * 5, 'c_int * 5'),
            (t.c_int / 5, 'c_int / 5'),
            (t.c_int % 5, 'c_int % 5'),
            (t.c_int // (t.c_int - 5), 'c_int // (c_int - 5)'),
            # ArraySlice
            (t.c_array[:5, 2], 'c_array[:5, 2]'),
            # ColumnPropertyRef
            (t.c_image.errormsg, 'c_image.errormsg'),
            # Comparison
            (t.c_int == 5, 'c_int == 5'),
            (t.c_int != 5, 'c_int != 5'),
            (t.c_int < 5, 'c_int < 5'),
            (t.c_int <= 5, 'c_int <= 5'),
            (t.c_int > 5, 'c_int > 5'),
            (t.c_int >= 5, 'c_int >= 5'),
            # CompoundPredicate
            ((t.c_int == 5) & (t.c_float > 5), '(c_int == 5) & (c_float > 5)'),
            ((t.c_int == 5) | (t.c_float > 5), '(c_int == 5) | (c_float > 5)'),
            (~(t.c_int == 5), '~(c_int == 5)'),
            # FunctionCall
            (pxtf.string.lower(t.c_string), 'lower(c_string)'),
            (pxtf.image.quantize(t.c_image, kmeans=5), 'quantize(c_image, kmeans=5)'),
            # InPredicate
            (t.c_int.isin([1, 2, 3]), 'c_int.isin([1, 2, 3])'),
            # InlineDict/List
            (
                pxtf.openai.chat_completions([{'system': t.c_string}], model='test'),
                "chat_completions([{'system': c_string}], model='test')",
            ),
            # InlineArray
            (pxt.array([1, 2, t.c_int]), '[1, 2, c_int]'),
            # IsNull
            (t.c_int == None, 'c_int == None'),
            # JsonPath
            (t.c_json.f2.f5[2:4][3], 'c_json.f2.f5[2:4][3]'),
            # JsonPath with relative root (with and without a succeeding path)
            (t.c_json.f2.f5['*'] >> R, 'c_json.f2.f5[*] >> R'),
            (t.c_json.f2.f5['*'] >> R.abcd, 'c_json.f2.f5[*] >> R.abcd'),
            # MethodRef
            (t.c_image.resize((100, 100)), 'c_image.resize([100, 100])'),
            # TypeCast
            (t.c_int.astype(pxt.Float), 'c_int.astype(Float)'),
        ]
        for e, expected_repr in instances:
            assert repr(e) == expected_repr

    def test_string_concat_exprs(self, test_tbl: catalog.Table) -> None:
        # create table with two columns
        schema = {'s1': pxt.String, 's2': pxt.String}
        t = pxt.create_table('test_str_concat', schema)
        t.add_computed_column(s3=t.s1 + '-' + t.s2)
        t.add_computed_column(s4=t.s1 * 3)

        t.insert([{'s1': 'left', 's2': 'right'}, {'s1': 'A', 's2': 'B'}])
        result = t.collect()
        assert result['s3'] == ['left-right', 'A-B']
        assert result['s4'] == ['leftleftleft', 'AAA']

        with pytest.raises(excs.Error) as exc_info:
            _ = t.add_computed_column(invalid_op=t.s1 * 's1')
        assert '* on strings requires int type,' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = t.add_computed_column(invalid_op=t.s1 + 3)
        assert '+ on strings requires string type,' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = t.add_computed_column(invalid_op=t.s1 / t.s2)
        assert 'invalid operation / on strings, only operators + and * are supported' in str(exc_info.value)


@pxt.udf
def udf1(x: int, y: str) -> str:
    return f'{x} {y}'
