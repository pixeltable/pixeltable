import numpy as np
import pandas as pd
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

    def test_serialize_anonymous(self, init_db: None) -> None:
        d = self.func.as_dict()
        FunctionRegistry.get().clear_cache()
        deserialized = Function.from_dict(d)
        assert deserialized.eval_fn(1) == 2

    def test_create(self, test_db: catalog.Db) -> None:
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
            library_fn = Function(IntType(), [IntType()], module_name=__name__, eval_symbol='dummy_fn')
            db.create_function('library_fn', library_fn)

    def test_update(self, test_db: catalog.Db) -> None:
        db = test_db
        db.create_function('test_fn', self.func)
        FunctionRegistry.get().clear_cache()
        cl = pt.Client()
        db2 = cl.get_db('test')
        fn = db2.load_function('test_fn')
        db2.update_function('test_fn', lambda x: x + 2)
        # change visible in previously-loaded Function
        assert fn.eval_fn(1) == 3

    def test_rename(self, test_db: catalog.Db) -> None:
        db = test_db
        db.create_function('test_fn', self.func)

        FunctionRegistry.get().clear_cache()
        cl = pt.Client()
        db2 = cl.get_db('test')
        with pytest.raises(exc.UnknownEntityError):
            db2.rename_function('test_fn2', 'test_fn')
        db2.rename_function('test_fn', 'test_fn2')
        func = db2.load_function('test_fn2')
        assert func.eval_fn(1) == 2

        with pytest.raises(exc.UnknownEntityError):
            _ = db2.load_function('test_fn')

        # move function between directories
        db2.create_dir('functions')
        db2.create_dir('functions2')
        db2.create_function('functions.func1', self.func)
        with pytest.raises(exc.UnknownEntityError):
            db2.rename_function('functions2.func1', 'functions.func1')
        db2.rename_function('functions.func1', 'functions2.func1')

        FunctionRegistry.get().clear_cache()
        cl = pt.Client()
        db3 = cl.get_db('test')
        func = db3.load_function('functions2.func1')
        assert func.eval_fn(1) == 2
        with pytest.raises(exc.UnknownEntityError):
            _ = db3.load_function('functions.func1')

    def test_drop(self, test_db: catalog.Db) -> None:
        db = test_db
        db.create_function('test_fn', self.func)
        FunctionRegistry.get().clear_cache()
        cl = pt.Client()
        db2 = cl.get_db('test')
        db2.drop_function('test_fn')

        with pytest.raises(exc.UnknownEntityError):
            _ = db2.load_function('test_fn')
