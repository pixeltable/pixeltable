import sqlalchemy as sql

from pixeltable import catalog
from pixeltable.type_system import ColumnType
from pixeltable.exprs import FunctionCall, Expr, CompoundPredicate
from pixeltable.functions import Function
from pixeltable.functions.pil.image import blend
from pixeltable.functions.clip import encode_image, encode_text
from pixeltable.utils.clip import encode_text


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
        udf = Function(lambda a: True, ColumnType.BOOL, [ColumnType.STRING])
        udf2 = Function(lambda a: True, ColumnType.BOOL, [ColumnType.INT])

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

    def test_select_list(self, test_img_tbl: catalog.Table) -> None:
        t = test_img_tbl
        result = t[t.img].show(n=100)
        _ = result._repr_html_()
        df = t[
            t.img, FunctionCall(lambda img: img.rotate(60), tbl=t, return_type=ColumnType.IMAGE)
        ]
        _ = df.show(n=100)._repr_html_()
        df = t[[t.img, t.img.rotate(60)]]
        _ = df.show(n=100)._repr_html_()

    def test_img_members(self, test_img_tbl: catalog.Table) -> None:
        t = test_img_tbl
        result = t[t.img, t.img.height, t.img.rotate(90)].show(n=100)
        _ = result._repr_html_()

    def test_img_functions(self, test_img_tbl: catalog.Table) -> None:
        t = test_img_tbl
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

    def test_similarity(self, test_img_tbl: catalog.Table) -> None:
        t = test_img_tbl
        data = t.show(30)
        probe = t[t.img, t.category].show(1)
        img = probe[0, 0]
        result = t[t.img.nearest(img, 10)].show(10)
        assert len(result) == 10
        # nearest() with one SQL predicate and one Python predicate
        result = t[t.img.nearest(img, 10) & (t.category == probe[0, 1]) & (t.img.width > 1)].show(10)
        assert len(result) == 3

        result = t[t.img.matches('musical instrument', 10)].show(10)
        assert len(result) == 10
        # matches() with one SQL predicate and one Python predicate
        french_horn_category = 'n03394916'
        result = t[t.img.matches('musical instrument', 10) & (t.category == french_horn_category) & (t.img.width > 1)].show(10)
        assert len(result) == 6
