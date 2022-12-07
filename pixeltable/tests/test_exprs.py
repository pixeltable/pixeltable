import sqlalchemy as sql
import pytest

from pixeltable import catalog
from pixeltable.type_system import StringType, BoolType, IntType, ImageType, Function
from pixeltable.exprs import Expr, CompoundPredicate, FunctionCall
from pixeltable.functions import udf_call, dict_map
from pixeltable.functions.pil.image import blend
from pixeltable.functions.clip import encode_image
from pixeltable import exceptions as exc


class TestExprs:
    def test_basic(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        assert isinstance(t['c1'] < 'a', Expr)
        assert isinstance(t.c1 < 'a', Expr)
        assert isinstance(t['c1'] <= 'a', Expr)
        assert isinstance(t.c1 <= 'a', Expr)
        assert isinstance(t['c1'] == 'a', Expr)
        assert isinstance(t.c1 == 'a', Expr)
        assert isinstance(t['c1'] != 'a', Expr)
        assert isinstance(t.c1 != 'a', Expr)
        assert isinstance(t['c1'] > 'a', Expr)
        assert isinstance(t.c1 > 'a', Expr)
        assert isinstance(t['c1'] >= 'a', Expr)
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
        udf = Function(lambda a: True, BoolType(), [StringType()])
        udf2 = Function(lambda a: True, BoolType(), [IntType()])

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

    def test_basic_filter(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        _ = t[t.c1 == 'test string'].show()
        print(_)
        _ = t[t.c2 > 50].show()
        print(_)

    def test_dicts(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        _ = t[t.c6.detections['*'].iscrowd].show()
        print(_)
        _ = t[t.c6.detections['*'].bounding_box].show()
        print(_)
        _ = t[t.c6.detections['*'].bounding_box[0]].show()
        print(_)

    def test_select_list(self, img_tbl) -> None:
        t = img_tbl
        result = t[t.img].show(n=100)
        _ = result._repr_html_()
        df = t[t.img, udf_call(lambda img: img.rotate(60), ImageType(), tbl=t)]
        _ = df.show(n=100)._repr_html_()
        df = t[[t.img, t.img.rotate(60)]]
        _ = df.show(n=100)._repr_html_()

        with pytest.raises(exc.OperationalError):
            _ = t[t.img.rotate]

    def test_img_members(self, img_tbl) -> None:
        t = img_tbl
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
        result = t[encode_image(t.img)].show(10)
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

        with pytest.raises(exc.OperationalError):
            _ = t[t.img.nearest(img)].show(10)
        with pytest.raises(exc.OperationalError):
            _ = t[t.img.matches('musical instrument')].show(10)
