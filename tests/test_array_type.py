from typing import Iterable

import numpy as np
import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable import type_system as ts
from pixeltable.type_system import ArrayType, ColumnType, IntType

from .utils import reload_catalog, validate_update_status


class TestArrayType:
    @pytest.mark.parametrize('do_reload_catalog', [False, True], ids=['no_reload_catalog', 'reload_catalog'])
    def test_array_dtypes(self, do_reload_catalog: bool, init_env: None, reset_db: None) -> None:
        test_cases: list = [
            (np.bool, []),
            (np.str_, []),
            (np.int8, [np.bool]),
            (np.int16, [np.bool, np.int8, np.uint8]),
            (np.int32, [np.bool, np.int8, np.int16, np.uint8, np.uint16]),
            (np.int64, [np.bool, np.int8, np.int16, np.int32, np.uint8, np.uint16, np.uint32]),
            (np.uint8, [np.bool]),
            (np.uint16, [np.bool, np.uint8]),
            (np.uint32, [np.bool, np.uint8, np.uint16]),
            (np.uint64, [np.bool, np.uint8, np.uint16, np.uint32]),
            (np.float16, [np.bool]),
            (np.float32, [np.bool, np.float16]),
            (np.float64, [np.bool, np.float16, np.float32]),
            (pxt.Int, [np.bool, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32]),
            (pxt.Float, [np.float16, np.float32]),
            (pxt.Bool, [np.bool]),
            (pxt.String, [np.str_]),
        ]
        for col_dtype, acceptable_dtypes in test_cases:
            try:
                self._test_array_dtype(col_dtype, acceptable_dtypes, do_reload_catalog)
            except Exception as e:
                raise type(e)(f'Failed for col_dtype={col_dtype}') from e

    def _test_array_dtype(
        self, col_dtype: type[np.generic] | ColumnType, acceptable_dtypes: list, do_reload_catalog: bool
    ) -> None:
        schema = {
            'array_col_req': pxt.Required[pxt.Array[col_dtype]],
            'array_col_opt': pxt.Array[col_dtype],  # type: ignore[misc]
        }
        pxt.create_table('test_numpy_dtypes', schema, if_exists='replace')
        reload_catalog(do_reload_catalog)
        t = pxt.get_table('test_numpy_dtypes')

        if col_dtype not in acceptable_dtypes:
            acceptable_dtypes.append(col_dtype)

        # Generate inserts for all dtypes that these columns should accept
        validate_update_status(
            t.insert(
                {
                    'array_col_req': self._make_array(acceptable_dtype),
                    'array_col_opt': self._make_array(acceptable_dtype),
                }
                for acceptable_dtype in acceptable_dtypes
            ),
            len(acceptable_dtypes),
        )
        assert t.count() == len(acceptable_dtypes)
        rows = t.select().collect()
        for row, dtype in zip(rows, acceptable_dtypes):
            for col in ['array_col_req', 'array_col_opt']:
                val = row[col]
                assert isinstance(val, np.ndarray)
                self._validate_dtype(val, dtype)

    def _make_array(self, dtype: type[np.generic] | ColumnType) -> np.ndarray:
        if isinstance(dtype, type) and issubclass(dtype, np.generic):
            return np.ones((2, 2), dtype=dtype)
        match dtype:
            case pxt.Int:
                return (1,)
            case pxt.Float:
                return (1.0,)
            case pxt.Bool:
                return (True,)
            case pxt.String:
                return ('abc', 'def')
            case _:
                raise ValueError(f'Unsupported dtype: {dtype}')

    def _validate_dtype(self, val: np.ndarray, literal_dtype: type[np.generic] | ColumnType) -> None:
        if literal_dtype in (pxt.String, np.str_):
            assert val.dtype.type == np.str_
            return
        if isinstance(literal_dtype, type) and issubclass(literal_dtype, np.generic):
            assert val.dtype == literal_dtype
            return
        match literal_dtype:
            case pxt.Int:
                assert val.dtype == np.int64
            case pxt.Bool:
                assert val.dtype == np.bool
            case pxt.Float:
                assert val.dtype == np.float32
            case _:
                raise ValueError(f'Unsupported dtype: {literal_dtype}')

    def test_non_parameterized_array_accepts_all_dtypes(self, init_env: None, reset_db: None) -> None:
        t = pxt.create_table('test_numpy_dtypes', {'array': pxt.Array})
        validate_update_status(t.insert(array=(1, 1)), 1)
        validate_update_status(t.insert(array=[1.0, 2.0]), 1)
        validate_update_status(t.insert(array=['abc', 'def']), 1)
        validate_update_status(
            t.insert({'array': np.ones((2, 2), dtype=dtype)} for dtype in [np.bool, np.int32, np.str_]), 3
        )

    @pytest.mark.parametrize('do_reload_catalog', [False, True], ids=['no_reload_catalog', 'reload_catalog'])
    def test_array_shape_validation(self, do_reload_catalog: bool, init_env: None, reset_db: None) -> None:
        schema = {
            'arr_1': pxt.Array[(1,), np.uint8],  # type: ignore[misc]
            'arr_2': pxt.Array[(2, 2), pxt.Float],  # type: ignore[misc]
            'arr_3': pxt.Array[(None,), pxt.String],  # type: ignore[misc]
            'arr_4': pxt.Array[(3, None, 2), np.int32],  # type: ignore[misc]
            'arr_5': pxt.Array,
        }
        pxt.create_table('test_numpy_dtypes', schema)
        reload_catalog(do_reload_catalog)
        t = pxt.get_table('test_numpy_dtypes')

        validate_update_status(t.insert(arr_1=np.ones((1,), dtype=np.uint8)), 1)
        with pytest.raises(excs.Error, match='arr_1'):
            t.insert(arr_1=np.ones((), dtype=np.uint8))
        with pytest.raises(excs.Error, match='arr_1'):
            t.insert(arr_1=np.ones((2,), dtype=np.uint8))
        with pytest.raises(excs.Error, match='arr_1'):
            t.insert(arr_1=np.ones((1, 3), dtype=np.uint8))

        validate_update_status(t.insert(arr_2=[[1.0, 0.0], [0.0, 5.0]]), 1)
        with pytest.raises(excs.Error, match='arr_2'):
            t.insert(arr_2=[[1.0, 0.0, 0.0], [0.0, 5.0, 0.0]])

        validate_update_status(t.insert(arr_3=[]), 1)
        validate_update_status(t.insert(arr_3=['']), 1)
        validate_update_status(t.insert(arr_3=['', 'a', 'b']), 1)
        with pytest.raises(excs.Error, match='arr_3'):
            t.insert(arr_3=[['a'], ['']])

        validate_update_status(t.insert(arr_4=np.ones((3, 1, 2), dtype=np.int32)), 1)
        validate_update_status(t.insert(arr_4=np.ones((3, 0, 2), dtype=np.int32)), 1)
        validate_update_status(t.insert(arr_4=np.ones((3, 6, 2), dtype=np.int32)), 1)
        with pytest.raises(excs.Error, match='arr_4'):
            t.insert(arr_4=np.ones((3, 2), dtype=np.int32))
        with pytest.raises(excs.Error, match='arr_4'):
            t.insert(arr_4=np.ones((3, 1, 2, 4), dtype=np.int32))

        validate_update_status(t.insert(arr_5=np.ones((2, 2), dtype=np.int32)), 1)
        validate_update_status(t.insert(arr_5=np.ones((4, 4, 4), dtype=np.str_)), 1)
        validate_update_status(t.insert(arr_5=np.zeros((0,), dtype=np.uint16)), 1)

        with pytest.raises(TypeError, match=r'Array type parameter must include a dtype.'):
            pxt.Array[1,]  # type: ignore[misc]

    def test_supertype(self) -> None:
        assert ArrayType(None, None).supertype(ArrayType(None, None)) == ArrayType(None, None)

        # dtype supertyping
        assert ArrayType(None, dtype=np.dtype('int32')).supertype(
            ArrayType(None, dtype=np.dtype('int32'))
        ) == ArrayType(None, dtype=np.dtype('int32'))
        assert ArrayType(None, dtype=np.dtype('uint8')).supertype(
            ArrayType(None, dtype=np.dtype('int32'))
        ) == ArrayType(None, dtype=np.dtype('int32'))
        assert ArrayType(None, dtype=np.dtype('bool')).supertype(ArrayType(None, dtype=np.dtype('str'))) == ArrayType(
            None, dtype=np.dtype('str')
        )
        # special case: the super dtype is neither of the two dtypes
        assert ArrayType(None, dtype=np.dtype('uint8')).supertype(ArrayType(None, dtype=np.dtype('int8'))) == ArrayType(
            None, dtype=np.dtype('int16')
        )
        assert ArrayType(None, None).supertype(ArrayType(None, dtype=np.dtype('int32'))) == ArrayType(None, None)

        # shape+dtype
        assert ArrayType((2, 2), dtype=np.dtype('int8')).supertype(
            ArrayType((2, 2), dtype=np.dtype('int8'))
        ) == ArrayType((2, 2), dtype=np.dtype('int8'))
        assert ArrayType(None, None).supertype(ArrayType((2, 2), dtype=np.dtype('int8'))) == ArrayType(None, None)
        assert ArrayType(None, dtype=np.dtype('int8')).supertype(
            ArrayType((2, 2), dtype=np.dtype('int8'))
        ) == ArrayType(None, dtype=np.dtype('int8'))
        assert ArrayType((2, 2), dtype=np.dtype('int8')).supertype(
            ArrayType((None, 2), dtype=np.dtype('int16'))
        ) == ArrayType((None, 2), dtype=np.dtype('int16'))
        assert ArrayType((2, None), dtype=np.dtype('int8')).supertype(
            ArrayType((None, 2), dtype=np.dtype('int16'))
        ) == ArrayType((None, None), dtype=np.dtype('int16'))
        assert ArrayType((2, 2), dtype=np.dtype('int8')).supertype(
            ArrayType((2, 2, 3), dtype=np.dtype('int8'))
        ) == ArrayType(None, dtype=np.dtype('int8'))
        assert ArrayType((1, 2, 3), dtype=np.dtype('bool')).supertype(
            ArrayType((3, 2, 1), dtype=np.dtype('bool'))
        ) == ArrayType((None, 2, None), dtype=np.dtype('bool'))

        # nullability
        assert ArrayType(None, None, nullable=False).supertype(ArrayType(None, None, nullable=True)) == ArrayType(
            None, None, nullable=True
        )
        assert ArrayType(None, None, nullable=True).supertype(ArrayType(None, None, nullable=True)) == ArrayType(
            None, None, nullable=True
        )
        assert ArrayType((2, 2), dtype=np.dtype('int8')).supertype(
            ArrayType((2, 2), dtype=np.dtype('int8'), nullable=True)
        ) == ArrayType((2, 2), dtype=np.dtype('int8'), nullable=True)

        # with incompatible types
        assert ArrayType(None, None).supertype(IntType()) == None

    def test_is_supertype_of(self) -> None:
        assert not ArrayType(None, None).is_supertype_of(IntType())
        assert not IntType().is_supertype_of(ArrayType(None, None))

        # Nullability
        assert ArrayType(None, None).is_supertype_of(ArrayType(None, None))
        assert ArrayType(None, None, nullable=True).is_supertype_of(
            ArrayType(None, None, nullable=False), ignore_nullable=False
        )
        assert not ArrayType(None, None, nullable=False).is_supertype_of(
            ArrayType(None, None, nullable=True), ignore_nullable=False
        )
        assert ArrayType(None, None, nullable=False).is_supertype_of(
            ArrayType(None, None, nullable=True), ignore_nullable=True
        )

        # dtypes only
        assert ArrayType(None, None).is_supertype_of(ArrayType(None, IntType()))
        assert not ArrayType(None, IntType()).is_supertype_of(ArrayType(None, None))
        assert ArrayType(None, IntType()).is_supertype_of(ArrayType(None, IntType()))
        assert ArrayType(None, np.dtype('int32')).is_supertype_of(ArrayType(None, np.dtype('int32')))
        assert ArrayType(None, np.dtype('int32')).is_supertype_of(ArrayType(None, np.dtype('int16')))
        assert ArrayType(None, np.dtype('int32')).is_supertype_of(ArrayType(None, np.dtype('bool')))
        assert ArrayType(None, IntType()).is_supertype_of(ArrayType(None, np.dtype('int16')))

        # shapes
        assert ArrayType(None, np.dtype('int32')).is_supertype_of(ArrayType((1, 2, 3), np.dtype('int32')))
        assert not ArrayType((1, 2, 3), np.dtype('int32')).is_supertype_of(ArrayType(None, np.dtype('int32')))
        assert not ArrayType((1, 2, 3), np.dtype('int32')).is_supertype_of(ArrayType((1, 2), np.dtype('int32')))
        assert not ArrayType((1, 2), np.dtype('int32')).is_supertype_of(ArrayType((1, 2, 3), np.dtype('int32')))

        assert ArrayType((1, 2, 3), np.dtype('int32')).is_supertype_of(ArrayType((1, 2, 3), np.dtype('int32')))
        assert ArrayType((1, None, 3), np.dtype('int32')).is_supertype_of(ArrayType((1, 2, 3), np.dtype('int32')))
        assert ArrayType((1, 2, None), np.dtype('int32')).is_supertype_of(ArrayType((1, 2, None), np.dtype('int32')))
        assert not ArrayType((1, 2, 3), np.dtype('int32')).is_supertype_of(ArrayType((4, 2, 3), np.dtype('int32')))
        assert not ArrayType((1, 2, 3), np.dtype('int32')).is_supertype_of(ArrayType((None, 2, 3), np.dtype('int32')))

    def test_matches(self) -> None:
        assert ArrayType(None, None).matches(ArrayType(None, None))
        assert ArrayType(None, np.dtype('uint8')).matches(ArrayType(None, np.dtype('uint8')))
        assert ArrayType(None, IntType()).matches(ArrayType(None, np.dtype('int64')))
        assert not ArrayType(None, np.dtype('uint8')).matches(ArrayType(None, np.dtype('uint16')))
        assert not ArrayType(None, np.dtype('uint16')).matches(ArrayType(None, np.dtype('uint8')))

        assert ArrayType((2, 2), np.dtype('uint8')).matches(ArrayType((2, 2), np.dtype('uint8')))
        assert not ArrayType((2, 1), np.dtype('uint8')).matches(ArrayType((2, 2), np.dtype('uint8')))
        assert not ArrayType((2, None), np.dtype('uint8')).matches(ArrayType((2, 2), np.dtype('uint8')))
        assert not ArrayType((2, 2), np.dtype('uint8')).matches(ArrayType(None, np.dtype('uint8')))

        assert ArrayType(None, np.dtype('uint8'), nullable=True).matches(
            ArrayType(None, np.dtype('uint8'), nullable=False)
        )

        assert not ArrayType(None, None).matches(IntType())

    def test_array_unsupported_dtypes(self) -> None:
        test_cases = [
            (np.generic, 'Unknown type'),
            (np.complex64, 'Unknown type'),
            (np.integer, 'Unknown type'),
            (np.number, 'Unknown type'),
            (np.floating, 'Unknown type'),
            (np.inexact, 'Unknown type'),
            (np.signedinteger, 'Unknown type'),
            (np.unsignedinteger, 'Unknown type'),
            (ColumnType, 'Unknown type'),
            (object, 'Unknown type'),
            (str, r'use.+pxt.String.+instead'),
            (int, r'use.+pxt.Int.+instead'),
        ]
        for type_, exc_regex in test_cases:
            with pytest.raises(excs.Error, match=exc_regex):
                pxt.Array[type_]  # type: ignore[misc]
            with pytest.raises(excs.Error, match=exc_regex):
                pxt.Array[(1,), type_]  # type: ignore[misc]

    def test_array_literal(self) -> None:
        test_cases: list[Iterable] = [
            ([1, 2, 3], (3,), np.dtype('int64')),
            ([[1, 2, 3.0]], (1, 3), np.dtype('float32')),
            ([np.ones((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)], (2, 1, 1), np.dtype('float32')),
            ([np.ones((1, 1), dtype=np.str_), np.zeros((1, 1), dtype=np.bool)], (2, 1, 1), np.dtype('str')),
        ]

        for i, (elements, expected_shape, expected_dtype) in enumerate(test_cases):
            try:
                arr = pxt.array(elements)
                assert isinstance(arr.col_type, ArrayType)
                assert arr.col_type.shape == expected_shape
                assert arr.col_type.dtype == expected_dtype
            except Exception as e:
                raise type(e)(f'Failed test case {i}') from e

    def test_bad_array_shape(self) -> None:
        test_cases: list[Iterable] = [
            [[1, 2], [3]],
            [[1, 2, 3], [1, 2, 3, 4]],
            [np.ones((3,), dtype=np.float32), np.zeros((4,), dtype=np.float32)],
            [np.ones((2, 2, 3), dtype=np.bool), np.zeros((2, 2, 4), dtype=np.bool)],
        ]

        for elements in test_cases:
            with pytest.raises(ValueError, match='inhomogeneous shape'):
                pxt.array(elements)

    def test_repr(self) -> None:
        test_cases = [
            (ArrayType(), 'Array'),
            (ArrayType(None, IntType()), 'Array[int64]'),
            (ArrayType((3,), ts.FloatType()), 'Array[(3,), float32]'),
            (ArrayType(None, np.dtype('uint8')), 'Array[uint8]'),
            (ArrayType((1, 2, 3), np.dtype('bool')), 'Array[(1, 2, 3), bool]'),
        ]
        for arr, expected_repr in test_cases:
            assert repr(arr) == expected_repr

    def test_json_schema(self) -> None:
        test_cases = [
            (ArrayType(), {'type': 'array'}),
            (ArrayType(None, IntType()), {'type': 'array', 'items': {'type': 'int64'}}),
            (ArrayType((3,), ts.FloatType()), {'type': 'array', 'items': {'type': 'float32'}}),
            (ArrayType(None, np.dtype('uint8')), {'type': 'array', 'items': {'type': 'uint8'}}),
            (ArrayType((1, 2, 3), np.dtype('bool')), {'type': 'array', 'items': {'type': 'bool'}}),
        ]
        for arr, expected_json_schema in test_cases:
            assert arr.to_json_schema() == expected_json_schema

    def test_to_dict(self) -> None:
        # Check that various dtypes are correctly serialized to dict
        # This is important because if done naively, a str_ can turn into something like '<U'
        as_dict = ArrayType((1, None, 3), np.dtype('uint8'), nullable=True)._as_dict()
        assert as_dict == {'shape': [1, None, 3], 'numpy_dtype': 'uint8', 'nullable': True}

        as_dict = ArrayType(shape=None, dtype=np.dtype('bool'))._as_dict()
        assert as_dict == {'shape': None, 'numpy_dtype': 'bool', 'nullable': False}

        as_dict = ArrayType(shape=None, dtype=np.dtype('str'))._as_dict()
        assert as_dict == {'shape': None, 'numpy_dtype': 'str', 'nullable': False}

        as_dict = ArrayType(shape=None, dtype=np.dtype('float64'))._as_dict()
        assert as_dict == {'shape': None, 'numpy_dtype': 'float64', 'nullable': False}

        as_dict = ArrayType(shape=(2, 2), dtype=np.ones((2, 2), dtype=np.float32).dtype)._as_dict()
        assert as_dict == {'shape': [2, 2], 'numpy_dtype': 'float32', 'nullable': False}

        as_dict = ArrayType(shape=(2, 2), dtype=np.ones((2, 2), dtype=np.str_).dtype)._as_dict()
        assert as_dict == {'shape': [2, 2], 'numpy_dtype': 'str', 'nullable': False}

    def test_to_from_dict(self) -> None:
        test_cases = [
            ArrayType(dtype=None, shape=None),
            ArrayType(dtype=np.dtype('int32')),
            ArrayType(dtype=np.dtype('float16')),
            ArrayType(shape=(2, 3), dtype=np.dtype('bool')),
            ArrayType(shape=(None, 4, 4), dtype=np.dtype('uint8')),
            ArrayType(dtype=np.dtype('float64'), nullable=True),
            ArrayType(shape=(42,), dtype=np.dtype('uint64'), nullable=True),
            ArrayType(shape=(2, 2), dtype=np.ones((2, 2), dtype=np.str_).dtype),
        ]
        for type in test_cases:
            as_dict = type._as_dict()
            from_dict = ArrayType._from_dict(as_dict)
            assert type == from_dict, type

    def test_numpy_dtypes_order(self) -> None:
        # ArrayType.supertype() relies on the property of ARRAY_SUPPORTED_NUMPY_DTYPES that all supertypes appear after
        # their subtypes
        for i, t_i in enumerate(ts.ARRAY_SUPPORTED_NUMPY_DTYPES):
            for j in range(i + 1, len(ts.ARRAY_SUPPORTED_NUMPY_DTYPES)):
                t_j = ts.ARRAY_SUPPORTED_NUMPY_DTYPES[j]
                can_cast = np.can_cast(t_j, t_i)
                assert not can_cast, f'Bad order of items of ARRAY_SUPPORTED_NUMPY_DTYPES: can cast from {t_j} to {t_i}'
