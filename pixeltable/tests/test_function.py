from pixeltable.function import Function, Registry
from pixeltable.type_system import IntType


class TestFunction:
    def test_basic(self, init_db: None) -> None:
        eval_fn = lambda x: x + 1
        func = Function(IntType(), [IntType()], eval_fn=eval_fn)
        d = func.as_dict()
        Registry.get().clear_cache()
        deserialized = Function.from_dict(d)
        assert deserialized.eval_fn(1) == 2


