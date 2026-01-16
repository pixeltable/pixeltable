from typing import Callable, overload
from .signature import Signature
from pixeltable import exceptions as excs


class PxtIterator:
    signatures: list[Signature]
    py_fns: list[Callable]

    def __init__(self, signatures: list[Signature], py_fns: list[Callable]):
        assert len(signatures) > 0
        assert len(signatures) == len(py_fns)
        self.signatures = signatures
        self.py_fns = py_fns


@overload
def iterator(decorated_fn: Callable) -> PxtIterator: ...

@overload
def iterator(*, unstored_cols: list[str] | None = None) -> Callable[[Callable], PxtIterator]: ...

def iterator(*args, **kwargs):  # type: ignore[no-untyped-def]
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return make_iterator(decorated_fn=args[0])
    else:
        unstored_cols = kwargs.pop('unstored_cols', None)
        if len(kwargs) > 0:
            raise excs.Error(f'Invalid @iterator decorator kwargs: {", ".join(kwargs.keys())}')
        if len(args) > 0:
            raise excs.Error('Unexpected @iterator decorator arguments.')

        def decorator(decorated_fn: Callable) -> PxtIterator:
            return make_iterator(decorated_fn=decorated_fn, unstored_cols=unstored_cols)

        return decorator


def make_iterator(decorated_fn: Callable, unstored_cols: list[str] | None = None) -> PxtIterator:
    sig = Signature.create(decorated_fn)
    # TODO: Unstored cols
    return PxtIterator([sig], [decorated_fn])
