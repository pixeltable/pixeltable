import pixeltable as pt
from pixeltable.tests.utils import make_tbl, create_table_data
from pixeltable import catalog
from pixeltable.type_system import ColumnType
from pixeltable.exprs import FunctionCall, Expr
from pixeltable.functions.pil.image import blend
from pixeltable.functions.clip import encode_image


class TestExprs:
    def test_basic(self, test_db: None) -> None:
        cl = pt.Client()
        db = cl.create_db('test')
        t1 = make_tbl(db, 'test1', ['c1', 'c2'])
        assert isinstance(t1['c1'] < 'a', Expr)
        assert isinstance(t1.c1 < 'a', Expr)
        assert isinstance(t1['c1'] <= 'a', Expr)
        assert isinstance(t1.c1 <= 'a', Expr)
        assert isinstance(t1['c1'] == 'a', Expr)
        assert isinstance(t1.c1 == 'a', Expr)
        assert isinstance(t1['c1'] != 'a', Expr)
        assert isinstance(t1.c1 != 'a', Expr)
        assert isinstance(t1['c1'] > 'a', Expr)
        assert isinstance(t1.c1 > 'a', Expr)
        assert isinstance(t1['c1'] >= 'a', Expr)
        assert isinstance(t1.c1 >= 'a', Expr)
        assert isinstance((t1.c1 == 'a') & (t1.c2 < 5), Expr)
        assert isinstance((t1.c1 == 'a') | (t1.c2 < 5), Expr)
        assert isinstance(~(t1.c1 == 'a'), Expr)

    def test_basic_filter(self, test_db: None) -> None:
        cl = pt.Client()
        db = cl.create_db('test')
        t1 = make_tbl(db, 'test1', ['c1', 'c2'])
        data = create_table_data(t1)
        t1.insert_pandas(data)
        _ = t1[t1.c1 == 'test string'].show()
        print(_)
        _ = t1[t1.c2 > 50].show()
        print(_)
        _ = (t1.c1 == 'test string') & (t1.c2 > 50)
        _ = _.sql_expr()
        print(_)
        _ = (t1.c1 == 'test string') | (t1.c2 > 50)
        _ = _.sql_expr()
        print(_)
        _ = ~(t1.c1 == 'test string')
        _ = _.sql_expr()
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

