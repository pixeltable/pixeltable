import re
import warnings
from textwrap import dedent
from typing import Any, Iterator, TypedDict

import numpy as np
import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable import func, type_system as ts
from pixeltable.iterators.base import ComponentIterator
from tests.utils import ReloadTester, reload_catalog


class MyRow(TypedDict):
    icol: int
    scol: str
    acol: pxt.Array[np.float32, (None, 512)] | None


@pxt.iterator
def simple_iterator(x: int, str_text: str = 'string') -> Iterator[MyRow]:
    for i in range(x):
        yield MyRow(icol=i, scol=f'{str_text} {i}', acol=None)


@simple_iterator.validate
def _(bound_args: dict[str, Any]) -> None:
    if 'x' in bound_args and bound_args['x'] < 0:
        raise pxt.Error('Parameter `x` must be non-negative.')
    if 'str_text' not in bound_args:
        raise pxt.Error('Parameter `str_text` must be a constant.')
    if not bound_args['str_text'].isidentifier():
        raise pxt.Error('Parameter `str_text` must be a valid identifier.')


@pxt.iterator
class class_based_iterator(pxt.PxtIterator[MyRow]):
    x: int
    str_text: str
    current: int

    def __init__(self, x: int, str_text: str = 'string') -> None:
        self.x = x
        self.str_text = str_text
        self.current = 0

    def __next__(self) -> MyRow:
        if self.current >= self.x:
            raise StopIteration
        result = MyRow(icol=self.current, scol=f'{self.str_text} {self.current}', acol=None)
        self.current += 1
        return result


@class_based_iterator.validate
def _(bound_args: dict[str, Any]) -> None:
    if 'x' in bound_args and bound_args['x'] < 0:
        raise pxt.Error('Parameter `x` must be non-negative.')
    if 'str_text' not in bound_args:
        raise pxt.Error('Parameter `str_text` must be a constant.')
    if not bound_args['str_text'].isidentifier():
        raise pxt.Error('Parameter `str_text` must be a valid identifier.')


@pxt.iterator(unstored_cols=['icol'])
class iterator_with_seek(pxt.PxtIterator[MyRow]):
    x: int
    str_text: str
    current: int

    def __init__(self, x: int, str_text: str = 'string') -> None:
        self.x = x
        self.str_text = str_text
        self.current = 0

    def __next__(self) -> MyRow:
        if self.current >= self.x:
            raise StopIteration
        result = MyRow(icol=self.current, scol=f'{self.str_text} {self.current}', acol=None)
        self.current += 1
        return result

    def seek(self, pos: int, **kwargs: Any) -> None:
        assert kwargs['scol'] == f'{self.str_text} {pos}'
        self.current = pos

    # Inline validate() method
    @classmethod
    def validate(cls, bound_args: dict[str, Any]) -> None:
        if 'x' in bound_args and bound_args['x'] < 0:
            raise pxt.Error('Parameter `x` must be non-negative.')
        if 'str_text' not in bound_args:
            raise pxt.Error('Parameter `str_text` must be a constant.')
        if not bound_args['str_text'].isidentifier():
            raise pxt.Error('Parameter `str_text` must be a valid identifier.')


class CustomLegacyIterator(ComponentIterator):
    input_text: str
    expand_by: int
    idx: int

    @classmethod
    def input_schema(cls, *args: Any, **kwargs: Any) -> dict[str, ts.ColumnType]:
        return {'text': ts.StringType(), 'expand_by': ts.IntType()}

    @classmethod
    def output_schema(cls, *args: Any, **kwargs: Any) -> tuple[dict[str, ts.ColumnType], list[str]]:
        return {'output_text': ts.StringType(), 'unstored_text': ts.StringType()}, ['unstored_text']

    def __init__(self, text: str, expand_by: int) -> None:
        self.input_text = text
        self.expand_by = expand_by
        self.idx = 0

    def __next__(self) -> dict[str, Any]:
        if self.idx >= self.expand_by:
            raise StopIteration
        result = {
            'output_text': f'stored {self.input_text} {self.idx}',
            'unstored_text': f'unstored {self.input_text} {self.idx}',
        }
        self.idx += 1
        return result

    def close(self) -> None:
        pass

    def set_pos(self, pos: int, **kwargs: Any) -> None:
        assert 0 <= pos < self.expand_by
        self.idx = pos


class TestIterator:
    def test_iterator(self, uses_db: None, reload_tester: ReloadTester) -> None:
        for n, it in enumerate((simple_iterator, class_based_iterator, iterator_with_seek)):
            assert callable(it)
            t = pxt.create_table(f'tbl_{n}', schema={'input': pxt.Int})
            t.insert([{'input': 2}])
            v = pxt.create_view(f'view_{n}', t, iterator=it(t.input))
            t.insert([{'input': 3}, {'input': 5}])
            rs = reload_tester.run_query(v.order_by(v.input, v.pos))
            assert list(rs) == [
                {'acol': None, 'input': 2, 'pos': 0, 'icol': 0, 'scol': 'string 0'},
                {'acol': None, 'input': 2, 'pos': 1, 'icol': 1, 'scol': 'string 1'},
                {'acol': None, 'input': 3, 'pos': 0, 'icol': 0, 'scol': 'string 0'},
                {'acol': None, 'input': 3, 'pos': 1, 'icol': 1, 'scol': 'string 1'},
                {'acol': None, 'input': 3, 'pos': 2, 'icol': 2, 'scol': 'string 2'},
                {'acol': None, 'input': 5, 'pos': 0, 'icol': 0, 'scol': 'string 0'},
                {'acol': None, 'input': 5, 'pos': 1, 'icol': 1, 'scol': 'string 1'},
                {'acol': None, 'input': 5, 'pos': 2, 'icol': 2, 'scol': 'string 2'},
                {'acol': None, 'input': 5, 'pos': 3, 'icol': 3, 'scol': 'string 3'},
                {'acol': None, 'input': 5, 'pos': 4, 'icol': 4, 'scol': 'string 4'},
            ]

            # Test that the iterator-specific validator works at insertion time
            with pytest.raises(pxt.Error, match=r'Parameter `x` must be non-negative.'):
                t.insert([{'input': -1}])

            # Test that the iterator-specific validator works at iterator creation time
            with pytest.raises(pxt.Error, match=r'Parameter `x` must be non-negative.'):
                it(-1)
            with pytest.raises(pxt.Error, match=r'Parameter `str_text` must be a constant.'):
                it(t.input, str_text=pxtf.uuid.uuid7().to_string())
            with pytest.raises(pxt.Error, match=r'Parameter `str_text` must be a valid identifier.'):
                it(t.input, str_text='I am not a valid identifier!')

        reload_tester.run_reload_test()

    @pytest.mark.parametrize('do_reload_catalog', [False, True], ids=['no_reload_catalog', 'reload_catalog'])
    def test_iterator_column_renames(self, uses_db: None, do_reload_catalog: bool) -> None:
        t = pxt.create_table('tbl', schema={'pos': pxt.String, 'input': pxt.Int, 'scol': pxt.Float})
        t.insert([{'pos': 'a', 'input': 5, 'scol': 1.0}, {'pos': 'b', 'input': 3, 'scol': 2.0}])
        v = pxt.create_view('view', t, iterator=simple_iterator(t.input))
        # Try adding a "second round" of iteration
        vv = pxt.create_view('view2', v, iterator=simple_iterator(v.icol))

        reload_catalog(do_reload_catalog)

        assert v._get_schema() == {
            'pos_1': ts.IntType(),
            'icol': ts.IntType(),
            'scol_1': ts.StringType(),
            'acol': ts.ArrayType((None, 512), ts.FloatType(), nullable=True),
            'input': ts.IntType(nullable=True),
            'pos': ts.StringType(nullable=True),
            'scol': ts.FloatType(nullable=True),
        }
        assert vv._get_schema() == {
            'pos_2': ts.IntType(),
            'icol_1': ts.IntType(),
            'scol_2': ts.StringType(),
            'acol_1': ts.ArrayType((None, 512), ts.FloatType(), nullable=True),
            'pos_1': ts.IntType(),
            'icol': ts.IntType(),
            'scol_1': ts.StringType(),
            'acol': ts.ArrayType((None, 512), ts.FloatType(), nullable=True),
            'input': ts.IntType(nullable=True),
            'pos': ts.StringType(nullable=True),
            'scol': ts.FloatType(nullable=True),
        }

    def test_iterator_errors(self, uses_db: None) -> None:
        # Error: class not a subclass of PxtIterator
        with pytest.raises(
            pxt.Error,
            match=r'@pxt.iterator-decorated class `.*not_pxt_iterator` must be a subclass of `pixeltable.PxtIterator`.',
        ):

            @pxt.iterator
            class not_pxt_iterator:
                def __init__(self) -> None:
                    pass

        # Error: class doesn't implement __next__()
        with pytest.raises(
            pxt.Error, match=r'@pxt.iterator-decorated class `.*no_next_method` must implement a `__next__\(\)` method.'
        ):

            @pxt.iterator
            class no_next_method(pxt.PxtIterator[MyRow]):
                def __init__(self) -> None:
                    pass

        # Error: unstored_cols without seek() method
        with pytest.raises(
            pxt.Error, match=r'Iterator `.*no_seek_method` with `unstored_cols` must implement a `seek\(\)` method.'
        ):

            @pxt.iterator(unstored_cols=['icol'])
            class no_seek_method(pxt.PxtIterator[MyRow]):
                def __init__(self) -> None:
                    pass

                def __next__(self) -> MyRow:
                    raise StopIteration

        # Error: validate() not a @classmethod
        with pytest.raises(
            pxt.Error,
            match=r'`validate\(\)` method of @pxt.iterator `.*validate_not_classmethod` must be a @classmethod.',
        ):

            @pxt.iterator
            class validate_not_classmethod(pxt.PxtIterator[MyRow]):
                def __init__(self) -> None:
                    pass

                def __next__(self) -> MyRow:
                    raise StopIteration

                def validate(self, bound_args: dict[str, Any]) -> None:  # type: ignore[override]
                    pass

        # Error: conditional_output_schema() not a @classmethod
        with pytest.raises(
            pxt.Error,
            match=r'`conditional_output_schema\(\)` method of @pxt.iterator '
            r'`.*cos_not_classmethod` must be a @classmethod.',
        ):

            @pxt.iterator
            class cos_not_classmethod(pxt.PxtIterator[MyRow]):
                def __init__(self) -> None:
                    pass

                def __next__(self) -> MyRow:
                    raise StopIteration

                def conditional_output_schema(self, bound_args: dict[str, Any]) -> dict[str, type] | None:  # type: ignore[override]
                    return None

        # Error: __next__() has wrong return type (not dict)
        with pytest.raises(
            pxt.Error,
            match=r'`__next__\(\)` method of @pxt.iterator-decorated class '
            r'`.*wrong_next_return_type` must have return type `dict` or a subclass of `TypedDict`.',
        ):

            @pxt.iterator
            class wrong_next_return_type(pxt.PxtIterator):
                def __init__(self) -> None:
                    pass

                def __next__(self) -> int:
                    raise StopIteration

        # Error: function iterator with wrong return type (not Iterator[dict])
        with pytest.raises(
            pxt.Error,
            match=r'@pxt.iterator-decorated function `.*wrong_return_type\(\)` must have return type '
            r'`Iterator\[dict\]`, or `Iterator\[T\]` for a subclass `T` of `TypedDict`.',
        ):

            @pxt.iterator
            def wrong_return_type(x: int) -> int:
                return x

        # Error: function iterator returning Iterator[int] instead of Iterator[dict]
        with pytest.raises(
            pxt.Error,
            match=r'@pxt.iterator-decorated function `.*iterator_of_ints\(\)` must have return type '
            r'`Iterator\[dict\]`, or `Iterator\[T\]` for a subclass `T` of `TypedDict`.',
        ):

            @pxt.iterator
            def iterator_of_ints(x: int) -> Iterator[int]:
                yield from []

        # Error: TypedDict field has non-convertible type
        class BadFieldType(TypedDict):
            icol: int
            bad_field: object  # object cannot be converted to a Pixeltable type

        with pytest.raises(pxt.Error, match=r"Could not infer Pixeltable type for output field 'bad_field'"):

            @pxt.iterator
            def bad_field_iterator(x: int) -> Iterator[BadFieldType]:
                yield from []

        # Error: plain dict return without conditional_output_schema
        @pxt.iterator
        def plain_dict_iterator(x: int) -> Iterator[dict]:
            yield from []

        t = pxt.create_table('tbl_plain_dict', schema={'input': pxt.Int})
        with pytest.raises(
            pxt.Error,
            match=r'Iterator `.*plain_dict_iterator` must either return a `TypedDict` '
            r'or define a `conditional_output_schema`.',
        ):
            pxt.create_view('view_plain_dict', t, iterator=plain_dict_iterator(t.input))

        # Error: conditional_output_schema returns None
        @pxt.iterator
        def conditional_returns_none(x: int) -> Iterator[dict]:
            yield from []

        @conditional_returns_none.conditional_output_schema
        def _(bound_args: dict[str, Any]) -> dict[str, type] | None:
            return None

        with pytest.raises(
            pxt.Error, match=r'The `conditional_output_schema` for iterator `.*conditional_returns_none` returned None'
        ):
            pxt.create_view('view_cond_none', t, iterator=conditional_returns_none(t.input))

        # Error: duplicate validate() decorator
        @pxt.iterator
        class iterator_with_validate(pxt.PxtIterator[MyRow]):
            def __init__(self) -> None:
                pass

            def __next__(self) -> MyRow:
                raise StopIteration

            @classmethod
            def validate(cls, bound_args: dict[str, Any]) -> None:
                pass

        with pytest.raises(
            pxt.Error, match=r'@pxt.iterator `.*iterator_with_validate` already defines a `validate\(\)` method.'
        ):

            @iterator_with_validate.validate
            def _(bound_args: dict[str, Any]) -> None:
                pass

        # Error: duplicate conditional_output_schema() decorator
        @pxt.iterator
        class iterator_with_cos(pxt.PxtIterator[MyRow]):
            def __init__(self) -> None:
                pass

            def __next__(self) -> MyRow:
                raise StopIteration

            @classmethod
            def conditional_output_schema(cls, bound_args: dict[str, Any]) -> dict[str, type] | None:
                return {'icol': int, 'scol': str}

        with pytest.raises(
            pxt.Error,
            match=r'@pxt.iterator `.*iterator_with_cos` already defines a `conditional_output_schema\(\)` method.',
        ):

            @iterator_with_cos.conditional_output_schema
            def _(bound_args: dict[str, Any]) -> dict[str, type]:
                return {'icol': int}

        # Error: invalid decorator kwargs
        with pytest.raises(pxt.Error, match=r'Invalid @iterator decorator kwargs: invalid_kwarg'):

            @pxt.iterator(invalid_kwarg='value')  # type: ignore[call-overload]
            def invalid_kwarg_iterator(x: int) -> Iterator[MyRow]:
                yield from []

        # Error: unexpected decorator arguments
        with pytest.raises(pxt.Error, match=r'Unexpected @iterator decorator arguments.'):

            @pxt.iterator('unexpected_arg')  # type: ignore[call-overload]
            def unexpected_arg_iterator(x: int) -> Iterator[MyRow]:
                yield from []

        # Error: use of reserved column name
        with pytest.raises(pxt.Error, match=r"'pos' is reserved and cannot be the name of an iterator output."):

            class PosRow(TypedDict):
                pos: int

            @pxt.iterator
            def reserved_pos_iterator(x: int) -> Iterator[PosRow]:
                yield from []

    @pytest.mark.parametrize('as_kwarg', [False, True])
    def test_evolving_iterator(self, as_kwarg: bool, uses_db: None) -> None:
        """
        Tests that code changes to iterators that are backward-compatible with the code pattern in view
        are accepted by Pixeltable.

        The test operates by instantiating a computed column with the UDF `evolving_iterator`, then repeatedly
        monkey-patching `evolving_iterator` with different signatures and checking that they new signatures are
        accepted by Pixeltable.

        The test runs two ways:
        - with the iterator invoked using a positional argument: `evolving_iterator(t.c1)`
        - with the iterator invoked using a keyword argument: `evolving_iterator(a=t.c1)`

        We also test that backward-incompatible changes raise appropriate warnings and errors. Because the
        error messages are lengthy and complex, we match against the entire fully-baked error string, to ensure
        that they remain comprehensible after future refactorings.
        """
        import tests.test_iterator  # noqa: PLW0406

        t = pxt.create_table('test', {'c1': pxt.Int})
        t.insert(c1=5)

        def mimic(it: func.GeneratingFunction) -> None:
            """Monkey-patches `tests.test_function.evolving_udf` with the given function."""
            tests.test_iterator.evolving_iterator = func.GeneratingFunction(
                it.decorated_callable, it.unstored_cols, 'tests.test_iterator.evolving_iterator'
            )

        def reload_and_validate_table(validation_error: str | None = None, has_view: bool = True) -> None:
            reload_catalog()

            # t and v should load without warnings
            t = pxt.get_table('test')
            if has_view:
                v = pxt.get_table('view')

            assert list(t.head()) == [{'c1': 5}]
            if has_view:
                assert list(v.head()) == [
                    {'acol': None, 'c1': 5, 'pos': i, 'icol': i, 'scol': f'prefix {i}'} for i in range(5)
                ]

            # Ensure that inserting or updating raises an error if there is an invalid column
            if validation_error is None:
                with warnings.catch_warnings():
                    warnings.simplefilter('error', pxt.PixeltableWarning)
                    t.insert(c1=6)
                    t.where(t.c1 == 6).update({'c1': 7})
                    t.where(t.c1 == 7).delete()
            else:
                with pytest.raises(pxt.Error, match=error_regex(validation_error)):
                    t.insert(c1=6)
                # TODO: Check for error on update, once update plans are working for iterator views

        def error_regex(msg: str) -> str:
            regex = '\n'.join(
                [
                    re.escape(
                        "Data cannot be updated in the View 'view',\n"
                        "because the iterator defined on 'view' is currently invalid:"
                    ),
                    re.escape(msg),
                ]
            )
            return '(?s)' + regex

        db_params = '(a: pxt.Int | None)' if as_kwarg else '(pxt.Int | None)'
        signature_error = dedent(
            f"""
            The signature stored in the database for a call to `tests.test_iterator.evolving_iterator` no longer
            matches its signature as currently defined in the code. This probably means that the
            code for `tests.test_iterator.evolving_iterator` has changed in a backward-incompatible way.
            Signature of iterator in the database: {db_params} -> ...
            Signature of iterator as currently defined in code: {{params}} -> ...
            """
        ).strip()
        output_schema_mismatch_error = dedent(
            """
            The output schema stored in the database for a call to `tests.test_iterator.evolving_iterator` no longer
            matches its output schema as currently defined in the code. This probably means that the
            code for `tests.test_iterator.evolving_iterator` has changed in a backward-incompatible way.
            The type of output field 'acol' is incompatible
            (expected `pxt.Array[(None, 512), float32] | None`; got `{return_type}`).
            """
        ).strip()

        @pxt.iterator
        def iter_base_version(a: int, prefix: str = 'prefix') -> Iterator[MyRow]:
            for i in range(a):
                yield MyRow(icol=i, scol=f'{prefix} {i}', acol=None)

        mimic(iter_base_version)
        if as_kwarg:
            pxt.create_view('view', t, iterator=tests.test_iterator.evolving_iterator(a=t.c1))
        else:
            pxt.create_view('view', t, iterator=tests.test_iterator.evolving_iterator(t.c1))

        # Change type of an unused optional parameter; this works in all cases
        @pxt.iterator
        def iter_version_2(a: int, prefix: int = 8) -> Iterator[MyRow]:
            for i in range(a):
                yield MyRow(icol=i, scol=f'{prefix} {i}', acol=None)

        mimic(iter_version_2)
        reload_and_validate_table()

        # Rename the parameter; this works only if the iterator was invoked with a positional argument
        @pxt.iterator
        def iter_version_3(c: int, prefix: str = 'prefix') -> Iterator[MyRow]:
            for i in range(c):
                yield MyRow(icol=i, scol=f'{prefix} {i}', acol=None)

        mimic(iter_version_3)
        if as_kwarg:
            reload_and_validate_table(
                validation_error=signature_error.format(params='(c: pxt.Int, prefix: pxt.String)')
            )
        else:
            reload_and_validate_table()

        # Change the parameter from fixed to variadic; this works only if the iterator was invoked with a positional
        # argument
        # @pxt.iterator
        # def iter_version_4(*a: int, prefix: str = 'prefix') -> Iterator[MyRow]:
        #     for i in range(a[0]):
        #         yield MyRow(icol=i, scol=f'{prefix} {i}', acol=None)

        # mimic(iter_version_4)
        # if as_kwarg:
        #     reload_and_validate_table(validation_error=signature_error.format(params='(*a)'))
        # else:
        #     reload_and_validate_table()

        # Narrow the type of an output column; this works in all cases
        class NarrowRow(TypedDict):
            icol: int
            scol: str
            acol: pxt.Array[np.float32, (3, 512)] | None

        @pxt.iterator
        def iter_version_5(a: int, prefix: int = 8) -> Iterator[NarrowRow]:
            for i in range(a):
                yield NarrowRow(icol=i, scol=f'{prefix} {i}', acol=None)

        mimic(iter_version_5)
        reload_and_validate_table()

        # Change the type of the parameter to something incompatible; this fails in all cases
        @pxt.iterator
        def iter_version_6(a: str, prefix: str = 'prefix') -> Iterator[MyRow]:
            for i in range(5):
                yield MyRow(icol=i, scol=f'{prefix} {i}', acol=None)

        mimic(iter_version_6)
        reload_and_validate_table(validation_error=signature_error.format(params='(a: pxt.String, prefix: pxt.String)'))

        # Widen the return type; this fails in all cases
        class WideRow(TypedDict):
            icol: int
            scol: str
            acol: pxt.Array | None

        @pxt.iterator
        def iter_version_7(a: int, prefix: str = 'prefix') -> Iterator[WideRow]:
            for i in range(a):
                yield WideRow(icol=i, scol=f'{prefix} {i}', acol=None)

        mimic(iter_version_7)
        reload_and_validate_table(validation_error=output_schema_mismatch_error.format(return_type='pxt.Array | None'))

        # Add a poison parameter; this works only if the UDF was invoked with a keyword argument
        @pxt.iterator
        def iter_version_8(c: str = '', a: int = 11, prefix: str = 'prefix') -> Iterator[MyRow]:
            for i in range(a):
                yield MyRow(icol=i, scol=f'{prefix} {i}', acol=None)

        mimic(iter_version_8)
        if as_kwarg:
            reload_and_validate_table()
        else:
            reload_and_validate_table(
                validation_error=signature_error.format(params='(c: pxt.String, a: pxt.Int, prefix: pxt.String)')
            )

        # Widen a parameter type; this works in all cases
        # (This also tests scalar -> JSON parameter widening)
        @pxt.iterator
        def iter_version_9(a: pxt.Json, prefix: str = 'prefix') -> Iterator[MyRow]:
            for i in range(a):
                yield MyRow(icol=i, scol=f'{prefix} {i}', acol=None)

        mimic(iter_version_9)
        reload_and_validate_table()

        # Make the function into a non-UDF
        tests.test_iterator.evolving_iterator = lambda x: x  # type: ignore[assignment]
        validation_error = (
            'The iterator `tests.test_iterator.evolving_iterator` cannot be located, because\n'
            'the symbol `tests.test_iterator.evolving_iterator` is no longer a Pixeltable iterator. '
            '(Was the `@pxt.iterator` decorator removed?)'
        )
        reload_and_validate_table(validation_error=validation_error)

        # Remove the function entirely
        del tests.test_iterator.evolving_iterator
        validation_error = (
            'The iterator `tests.test_iterator.evolving_iterator` cannot be located, because\n'
            'the symbol `tests.test_iterator.evolving_iterator` no longer exists. '
            '(Was the iterator moved or renamed?)'
        )
        reload_and_validate_table(validation_error=validation_error)

        # Now drop the view with the broken iterator and make sure the table is still usable
        pxt.drop_table('view')
        reload_and_validate_table(has_view=False)

    def test_retrofit(self, uses_db: None) -> None:
        """
        Tests that legacy iterators defined as subclasses of ComponentIterator can be retrofitted
        into the new GeneratingFunction-based iterator system.
        """
        t = pxt.create_table('test', schema={'input': pxt.String})
        t.insert([{'input': 'balloon'}])
        with pytest.warns(DeprecationWarning, match=r'The `ComponentIterator` class is deprecated'):
            v = pxt.create_view(
                'view_legacy_iterator', t, iterator=CustomLegacyIterator.create(text=t.input, expand_by=4)
            )
        rs = v.order_by(v.input, v.pos).collect()
        assert list(rs) == [
            {'input': 'balloon', 'pos': 0, 'output_text': 'stored balloon 0', 'unstored_text': 'unstored balloon 0'},
            {'input': 'balloon', 'pos': 1, 'output_text': 'stored balloon 1', 'unstored_text': 'unstored balloon 1'},
            {'input': 'balloon', 'pos': 2, 'output_text': 'stored balloon 2', 'unstored_text': 'unstored balloon 2'},
            {'input': 'balloon', 'pos': 3, 'output_text': 'stored balloon 3', 'unstored_text': 'unstored balloon 3'},
        ]

    def test_nested_iterator(self, uses_db: None) -> None:
        n = 5
        t = pxt.create_table('test_nested', schema={'input': pxt.Int})
        t.insert([{'input': n}])

        v1 = pxt.create_view(
            'v1', t, iterator=simple_iterator(t.input), additional_columns={'additional_col_1': pxt.Int}
        )
        assert len(v1.collect()) == n

        v2 = pxt.create_view(
            'v2', v1, iterator=simple_iterator(v1.icol), additional_columns={'additional_col_2': pxt.Int}
        )
        assert len(v2.collect()) == n * (n - 1) // 2

        for _ in range(2):
            assert not v1._tbl_version.get().is_iterator_column(v1.input.col)
            assert not v1.input.col.is_iterator_col
            assert not v2._tbl_version.get().is_iterator_column(v2.input.col)
            assert not v2.input.col.is_iterator_col

            assert v1._tbl_version.get().is_iterator_column(v1.acol.col)
            assert v1.acol.col.is_iterator_col
            assert v2._tbl_version.get().is_iterator_column(v2.acol.col)
            assert v2.acol.col.is_iterator_col

            assert v1.scol.col.is_iterator_col
            assert v1._tbl_version.get().is_iterator_column(v1.scol.col)
            assert v2.scol.col.is_iterator_col
            assert v2._tbl_version.get().is_iterator_column(v2.scol.col)

            assert v1.icol.col.is_iterator_col
            assert v1._tbl_version.get().is_iterator_column(v1.icol.col)
            assert v2.icol.col.is_iterator_col
            assert v2._tbl_version.get().is_iterator_column(v2.icol.col)

            assert not v1.pos.col.is_iterator_col
            assert not v1._tbl_version.get().is_iterator_column(v1.pos.col)
            assert not v2.pos.col.is_iterator_col
            assert not v2._tbl_version.get().is_iterator_column(v2.pos.col)

            assert not v1.additional_col_1.col.is_iterator_col
            assert not v1._tbl_version.get().is_iterator_column(v1.additional_col_1.col)
            assert not v2.additional_col_1.col.is_iterator_col
            assert not v2._tbl_version.get().is_iterator_column(v2.additional_col_1.col)

            assert v2.acol_1.col.is_iterator_col
            assert v2._tbl_version.get().is_iterator_column(v2.acol_1.col)

            assert v2.scol_1.col.is_iterator_col
            assert v2._tbl_version.get().is_iterator_column(v2.scol_1.col)

            assert v2.icol_1.col.is_iterator_col
            assert v2._tbl_version.get().is_iterator_column(v2.icol_1.col)

            assert not v2.pos_1.col.is_iterator_col
            assert not v2._tbl_version.get().is_iterator_column(v2.pos_1.col)

            assert not v2.additional_col_2.col.is_iterator_col
            assert not v2._tbl_version.get().is_iterator_column(v2.additional_col_2.col)

            assert v1._tbl_version.get().iterator_columns() == [v1.icol.col, v1.scol.col, v1.acol.col]
            assert v2._tbl_version.get().iterator_columns() == [v2.icol_1.col, v2.scol_1.col, v2.acol_1.col]

            reload_catalog()
            v1 = pxt.get_table('v1')
            v2 = pxt.get_table('v2')


evolving_iterator: func.GeneratingFunction | None = None
