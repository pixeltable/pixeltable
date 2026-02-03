import re
from textwrap import dedent
from typing import Any, Iterator, TypedDict
import warnings

import pytest

import pixeltable as pxt
from pixeltable import func
import pixeltable.functions as pxtf
from tests.utils import ReloadTester, reload_catalog


class MyRow(TypedDict):
    icol: int
    scol: str


@pxt.iterator
def simple_iterator(x: int, str_text: str = 'string') -> Iterator[MyRow]:
    for i in range(x):
        yield MyRow(icol=i, scol=f'{str_text} {i}')


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
        result = MyRow(icol=self.current, scol=f'{self.str_text} {self.current}')
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
        result = MyRow(icol=self.current, scol=f'{self.str_text} {self.current}')
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
                {'input': 2, 'pos': 0, 'icol': 0, 'scol': 'string 0'},
                {'input': 2, 'pos': 1, 'icol': 1, 'scol': 'string 1'},
                {'input': 3, 'pos': 0, 'icol': 0, 'scol': 'string 0'},
                {'input': 3, 'pos': 1, 'icol': 1, 'scol': 'string 1'},
                {'input': 3, 'pos': 2, 'icol': 2, 'scol': 'string 2'},
                {'input': 5, 'pos': 0, 'icol': 0, 'scol': 'string 0'},
                {'input': 5, 'pos': 1, 'icol': 1, 'scol': 'string 1'},
                {'input': 5, 'pos': 2, 'icol': 2, 'scol': 'string 2'},
                {'input': 5, 'pos': 3, 'icol': 3, 'scol': 'string 3'},
                {'input': 5, 'pos': 4, 'icol': 4, 'scol': 'string 4'},
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
            r'`.*wrong_next_return_type` must have return type `dict` or `MyTypedDict`.',
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
            r'`Iterator\[dict\]` or `Iterator\[MyTypedDict\]`.',
        ):

            @pxt.iterator
            def wrong_return_type(x: int) -> int:
                return x

        # Error: function iterator returning Iterator[int] instead of Iterator[dict]
        with pytest.raises(
            pxt.Error,
            match=r'@pxt.iterator-decorated function `.*iterator_of_ints\(\)` must have return type '
            r'`Iterator\[dict\]` or `Iterator\[MyTypedDict\]`.',
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

    @pytest.mark.parametrize('as_kwarg', [False, True])
    def test_iterator_evolution(self, as_kwarg: bool, uses_db: None) -> None:
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

        def reload_and_validate_table(validation_error: str | None = None) -> None:
            reload_catalog()

            t: pxt.Table
            v: pxt.Table
            # Ensure a warning is generated when the table is accessed, if appropriate
            if validation_error is None:
                with warnings.catch_warnings():  # Ensure no warning is displayed
                    warnings.simplefilter('error', pxt.PixeltableWarning)
                    t = pxt.get_table('test')
                    v = pxt.get_table('view')
            else:
                with pytest.warns(pxt.PixeltableWarning, match=warning_regex(validation_error)):
                    t = pxt.get_table('test')
                    v = pxt.get_table('view')
                with warnings.catch_warnings():  # Ensure the warning is only displayed once
                    warnings.simplefilter('error', pxt.PixeltableWarning)
                    _ = pxt.get_table('test')
                    _ = pxt.get_table('view')

            assert list(t.head()) == [{'c1': 5}]
            assert list(v.head()) == [{'c1': 5, 'pos': i, 'icol': i, 'scol': f'prefix {i}'} for i in range(5)]

            # Ensure that inserting or updating raises an error if there is an invalid column
            if validation_error is None:
                with warnings.catch_warnings():
                    warnings.simplefilter('error', pxt.PixeltableWarning)
                    t.insert(c1=6)
                    t.where(t.c1 == 6).update({'c1': 7})
                    t.where(t.c1 == 5).delete()
            # else:
            #     with pytest.raises(pxt.Error, match=insert_error_regex(validation_error)):
            #         t.insert(c1='abc')
            #     with pytest.raises(pxt.Error, match=update_error_regex(validation_error)):
            #         t.where(t.c1 == 'xyz').update({'c1': 'def'})

        def warning_regex(msg: str) -> str:
            regex = '\n'.join(
                [
                    re.escape("The computed column 'result' in table 'test' is no longer valid."),
                    re.escape(msg),
                    re.escape(
                        'You can continue to query existing data from this column, '
                        'but evaluating it on new data will raise an error.'
                    ),
                ]
            )
            return '(?s)' + regex

        def insert_error_regex(msg: str) -> str:
            regex = '\n'.join(
                [
                    re.escape(
                        "Data cannot be inserted into the Table 'test',\n"
                        "because the column 'result' is currently invalid:"
                    ),
                    re.escape(msg),
                ]
            )
            return '(?s)' + regex

        def update_error_regex(msg: str) -> str:
            regex = '.*'.join(
                [
                    re.escape(
                        "Data cannot be updated in the Table 'test',\nbecause the column 'result' is currently invalid:"
                    ),
                    re.escape(msg),
                ]
            )
            return '(?s)' + regex

        db_params = '(a: pxt.String | None)' if as_kwarg else '(pxt.String | None)'
        signature_error = dedent(
            f"""
            The signature stored in the database for a UDF call to 'tests.test_function.evolving_udf' no longer
            matches its signature as currently defined in the code. This probably means that the
            code for 'tests.test_function.evolving_udf' has changed in a backward-incompatible way.
            Signature of UDF call in the database: {db_params} -> pxt.Array[float32] | None
            Signature of UDF as currently defined in code: {{params}} -> pxt.Array[float32] | None
            """
        ).strip()
        return_type_error = dedent(
            """
            The return type stored in the database for a UDF call to 'tests.test_function.evolving_udf' no longer
            matches its return type as currently defined in the code. This probably means that the
            code for 'tests.test_function.evolving_udf' has changed in a backward-incompatible way.
            Return type of UDF call in the database: Array[float32] | None
            Return type of UDF as currently defined in code: {return_type}
            """
        ).strip()

        @pxt.iterator
        def iter_base_version(a: int, prefix: str = 'prefix') -> Iterator[MyRow]:
            for i in range(a):
                yield MyRow(icol=i, scol=f'{prefix} {i}')

        mimic(iter_base_version)
        if as_kwarg:
            pxt.create_view('view', t, iterator=tests.test_iterator.evolving_iterator(a=t.c1))
        else:
            pxt.create_view('view', t, iterator=tests.test_iterator.evolving_iterator(t.c1))

        # Change type of an unused optional parameter; this works in all cases
        @pxt.iterator
        def iter_version_2(a: int, prefix: int = 8) -> Iterator[MyRow]:
            for i in range(a):
                yield MyRow(icol=i, scol=f'{prefix} {i}')

        mimic(iter_version_2)
        reload_and_validate_table()


evolving_iterator: func.GeneratingFunction | None = None
