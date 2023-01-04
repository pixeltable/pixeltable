import pytest

from pixeltable.function import Function, FunctionRegistry
from pixeltable.type_system import IntType
from pixeltable import catalog
import pixeltable as pt
from pixeltable import exceptions as exc


def dummy_fn(i: int) -> int:
    return i

class TestFunction:
    eval_fn = lambda x: x + 1
    func = Function(IntType(), [IntType()], eval_fn=eval_fn)

    def test_anonymous_fn(self, init_db: None) -> None:
        d = self.func.as_dict()
        FunctionRegistry.get().clear_cache()
        deserialized = Function.from_dict(d)
        assert deserialized.eval_fn(1) == 2

    def test_named_fn(self, test_db: catalog.Db) -> None:
        db = test_db
        db.create_function('test_fn', self.func)
        FunctionRegistry.get().clear_cache()
        cl = pt.Client()
        db2 = cl.get_db('test')
        fn2 = db2.load_function('test_fn')
        assert fn2.eval_fn(1) == 2

        with pytest.raises(exc.DuplicateNameError):
            db.create_function('test_fn', self.func)
        with pytest.raises(exc.UnknownEntityError):
            db.create_function('dir1.test_fn', self.func)
        with pytest.raises(exc.Error):
            library_fn = Function(IntType(), [IntType()], module_name=__name__, symbol='dummy_fn')
            db.create_function('library_fn', library_fn)
