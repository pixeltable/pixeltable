import pixeltable as pt
from pixeltable.tests.utils import make_tbl, create_table_data, read_data_file
from pixeltable import catalog
from pixeltable.type_system import ColumnType
from pixeltable.exprs import FunctionCall, Expr
from pixeltable.functions.pil import a


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

    def test_select_list(self, test_db: None) -> None:
        cl = pt.Client()
        db = cl.create_db('test')
        cols = [
            catalog.Column('file_path', ColumnType.IMAGE, nullable=False),
            catalog.Column('category', ColumnType.STRING, nullable=False),
            catalog.Column('split', ColumnType.STRING, nullable=False),
        ]
        tbl = db.create_table('test', cols)
        df = read_data_file('imagenette2-160', 'manifest.csv')
        tbl.insert_pandas(df)
        result = tbl[tbl.file_path].show(n=100)
        _ = result._repr_html_()
        df = tbl[[tbl.file_path, FunctionCall(lambda file_path: file_path.rotate(60), tbl=tbl, col_type=ColumnType.IMAGE)]]
        _ = df.show(n=100)._repr_html_()
        df = tbl[[tbl.file_path, tbl.file_path.rotate(60)]]
        _ = df.show(n=100)._repr_html_()

    def test_img_members(self, test_db: None) -> None:
        print(a)
        cl = pt.Client()
        db = cl.create_db('test')
        cols = [
            catalog.Column('file_path', ColumnType.IMAGE, nullable=False),
            catalog.Column('category', ColumnType.STRING, nullable=False),
            catalog.Column('split', ColumnType.STRING, nullable=False),
        ]
        tbl = db.create_table('test', cols)
        df = read_data_file('imagenette2-160', 'manifest.csv')
        tbl.insert_pandas(df)
        result = tbl[tbl.file_path, tbl.file_path.height, tbl.file_path.rotate(90)].show(n=100)
        _ = result._repr_html_()

