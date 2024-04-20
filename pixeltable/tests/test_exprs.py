import json
import urllib.parse
import urllib.request
from typing import List, Dict

import pytest
import sqlalchemy as sql

import pixeltable as pxt
import pixeltable.func as func
from pixeltable import catalog
from pixeltable import exceptions as excs
from pixeltable import exprs
from pixeltable.exprs import Expr, ColumnRef
from pixeltable.exprs import RELATIVE_PATH_ROOT as R
from pixeltable.functions import cast, sum, count
from pixeltable.functions.pil.image import blend
from pixeltable.iterators import FrameIterator
from pixeltable.tests.utils import get_image_files, skip_test_if_not_installed
from pixeltable.type_system import StringType, BoolType, IntType, ArrayType, ColumnType, FloatType, \
    VideoType


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
        img_t.add_column(c9=img_t.img.rotate(30))
        with pytest.raises(excs.Error) as excinfo:
            _ = img_t.select(img_t.c9.localpath).show()
        assert 'computed unstored' in str(excinfo.value)

    def test_null_args(self, test_client: pxt.Client) -> None:
        # create table with two int columns
        schema = {'c1': FloatType(nullable=True), 'c2': FloatType(nullable=True)}
        t = test_client.create_table('test', schema)

        # computed column that doesn't allow nulls
        t.add_column(c3=lambda c1, c2: c1 + c2, type=FloatType(nullable=False))
        t.add_column(c4=self.null_args_fn(t.c1, t.c2))

        # data that tests all combinations of nulls
        data = [{'c1': 1.0, 'c2': 1.0}, {'c1': 1.0, 'c2': None}, {'c1': None, 'c2': 1.0}, {'c1': None, 'c2': None}]
        status = t.insert(data, fail_on_exception=False)
        assert status.num_rows == len(data)
        assert status.num_excs == len(data) - 1
        result = t.select(t.c3, t.c4).collect()
        assert result['c3'] == [2.0, None, None, None]
        assert result['c4'] == [2.0, 1.0, None, None]

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
            with pytest.raises(excs.Error):
                _ = t[op1 + op2]
            with pytest.raises(excs.Error):
                _ = t[op1 - op2]
            with pytest.raises(excs.Error):
                _ = t[op1 * op2]
            with pytest.raises(excs.Error):
                _ = t[op1 / op2]

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


    def test_inline_dict(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        df = t[[{'a': t.c1, 'b': {'c': t.c2}, 'd': 1, 'e': {'f': 2}}]]
        result = df.show()
        print(result)

    def test_inline_array(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        result = t.select([[t.c2, 1], [t.c2, 2]]).show()
        t = result.column_types()[0]
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
        t.add_column(array_col=[[t.c2, 1], [1, t.c2]])
        _ = t[t.array_col].show()
        print(_)
        _ = t[t.array_col[:, 0]].show()
        print(_)

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
        result = t[t.img.height > 200][t.img].show(n=3)
        assert len(result) == 3
        result = t[t.img.crop((10, 10, 60, 60))].show(n=100)
        result = t[t.img.crop((10, 10, 60, 60)).resize((100, 100))].show(n=100)
        result = t[t.img.crop((10, 10, 60, 60)).resize((100, 100)).convert('L')].show(n=100)
        result = t[t.img.getextrema()].show(n=100)
        result = t[t.img, t.img.height, t.img.rotate(90)].show(n=100)
        _ = result._repr_html_()

    def test_img_functions(self, img_tbl) -> None:
        skip_test_if_not_installed('nos')
        t = img_tbl
        from pixeltable.functions.pil.image import resize
        result = t[t.img.resize((224, 224))].show(0)
        result = t[resize(t.img, (224, 224))].show(0)
        result = t[blend(t.img, t.img.rotate(90), 0.5)].show(100)
        print(result)
        from pixeltable.functions.nos.image_embedding import openai_clip
        result = t[openai_clip(t.img.resize((224, 224)))].show(10)
        print(result)
        _ = result._repr_html_()
        _ = t.img.entropy() > 1
        _ = (t.img.entropy() > 1) & (t.split == 'train')
        _ = (t.img.entropy() > 1) & (t.split == 'train') & (t.split == 'val')
        _ = (t.split == 'train') & (t.img.entropy() > 1) & (t.split == 'val') & (t.img.entropy() < 0)
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

    @pytest.mark.skip(reason='temporarily disabled')
    def test_similarity(self, small_img_tbl) -> None:
        skip_test_if_not_installed('nos')
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

        with pytest.raises(excs.Error):
            _ = t[t.img.nearest(img)].show(10)
        with pytest.raises(excs.Error):
            _ = t[t.img.nearest('musical instrument')].show(10)

    def test_ids(
            self, test_tbl: catalog.Table, test_tbl_exprs: List[exprs.Expr],
            img_tbl: catalog.Table, img_tbl_exprs: List[exprs.Expr]
    ) -> None:
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
        for e in test_tbl_exprs:
            e_serialized = e.serialize()
            e_deserialized = Expr.deserialize(e_serialized)
            assert e.equals(e_deserialized)

        for e in img_tbl_exprs:
            e_serialized = e.serialize()
            e_deserialized = Expr.deserialize(e_serialized)
            assert e.equals(e_deserialized)

    def test_print(self, test_tbl_exprs: List[exprs.Expr], img_tbl_exprs: List[exprs.Expr]) -> None:
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
        assert len(subexprs) == 4
        subexprs = [s for s in e.subexprs(expr_class=ColumnRef)]
        assert len(subexprs) == 1
        assert t.img.equals(subexprs[0])

    def test_window_fns(self, test_client: pxt.Client, test_tbl: catalog.Table) -> None:
        cl = test_client
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
        base_t = cl.create_table('videos', {'video': VideoType(), 'c2': IntType(nullable=False)})
        args = {'video': base_t.video, 'fps': 0}
        v = cl.create_view('frame_view', base_t, iterator_class=FrameIterator, iterator_args=args)
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
        new_t = cl.create_table('insert_test', schema=schema)
        new_t.add_column(c2_sum=sum(new_t.c2, group_by=new_t.c4, order_by=new_t.c3))
        rows = list(t.select(t.c2, t.c4, t.c3).collect())
        new_t.insert(rows)
        _ = new_t.show(0)

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
