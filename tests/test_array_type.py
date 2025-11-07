import numpy as np
import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.type_system import ColumnType

from .utils import validate_update_status


class TestArrayType:
    # 2. dtypes correctly preserved on disk, i.e. loading a table restores dtype in columns correctly.
    # TODO fix warnings like DeprecationWarning: Converting `np.generic` to a dtype is deprecated. The current result
    # is `np.dtype(np.void)` which is not strictly correct
    @pytest.mark.parametrize(
        'col_dtype,acceptable_dtypes',
        [
            (np.bool, []),
            (np.int8, [np.bool]),
            (np.int16, [np.bool, np.int8, np.uint8]),
            (np.int32, [np.bool, np.int8, np.int16, np.uint8, np.uint16]),
            (np.int64, [np.bool, np.int8, np.int16, np.int32, np.uint8, np.uint16, np.uint32]),
            (np.signedinteger, [np.bool, np.int8, np.int16, np.int32, np.int64]),
            (np.uint8, [np.bool]),
            (np.uint16, [np.bool, np.uint8]),
            (np.uint32, [np.bool, np.uint8, np.uint16]),
            (np.uint64, [np.bool, np.uint8, np.uint16, np.uint32]),
            (np.float16, [np.bool]),
            (np.float32, [np.bool, np.float16]),
            (np.float64, [np.bool, np.float16, np.float32]),
            (np.floating, [np.bool, np.float16, np.float32, np.float64]),
            (np.number, [np.bool, np.int8, np.uint8, np.float64]),
            (pxt.Int, [np.bool, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32]),
            (pxt.Float, [np.float16, np.float32, np.float64]),
            (pxt.Bool, [np.bool]),
            (pxt.String, [np.str_]),
        ],
        ids=[
            'np_bool',
            'np_int8',
            'np_int16',
            'np_int32',
            'np_int64',
            'np_signedinteger',
            'np_uint8',
            'np_uint16',
            'np_uint32',
            'np_uint64',
            'np_float16',
            'np_float32',
            'np_float64',
            'np_floating',
            'np_number',
            'pxt_int',
            'pxt_float',
            'pxt_bool',
            'pxt_string',
        ],
    )
    def test_array_dtypes(
        self,
        col_dtype: type[np.generic] | ColumnType,
        acceptable_dtypes: list[type[np.generic] | ColumnType],
        init_env: None,
        reset_db: None,
    ) -> None:
        schema = {
            'array_col_req': pxt.Required[pxt.Array[col_dtype]],
            'array_col_opt': pxt.Array[col_dtype],  # type: ignore[misc]
        }
        t = pxt.create_table('test_numpy_dtypes', schema)

        if col_dtype not in acceptable_dtypes:
            acceptable_dtypes.append(col_dtype)

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
                assert val.dtype == np.float64
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

    def test_array_shape_validation(self, init_env: None, reset_db: None) -> None:
        schema = {
            'arr_1': pxt.Array[(1,), np.uint8],  # type: ignore[misc]
            'arr_2': pxt.Array[(2, 2), pxt.Float],  # type: ignore[misc]
            'arr_3': pxt.Array[(None,), pxt.String],  # type: ignore[misc]
            'arr_4': pxt.Array[(3, None, 2), np.int32],  # type: ignore[misc]
        }
        t = pxt.create_table('test_numpy_dtypes', schema)

        t.insert(arr_1=np.ones((1,), dtype=np.uint8))
        with pytest.raises(excs.Error, match='arr_1'):
            t.insert(arr_1=np.ones((), dtype=np.uint8))
        with pytest.raises(excs.Error, match='arr_1'):
            t.insert(arr_1=np.ones((2,), dtype=np.uint8))
        with pytest.raises(excs.Error, match='arr_1'):
            t.insert(arr_1=np.ones((1, 3), dtype=np.uint8))

        t.insert(arr_2=[[1.0, 0.0], [0.0, 5.0]])
        with pytest.raises(excs.Error, match='arr_2'):
            t.insert(arr_2=[[1.0, 0.0, 0.0], [0.0, 5.0, 0.0]])

        t.insert(arr_3=[])
        t.insert(arr_3=[''])
        t.insert(arr_3=['', 'a', 'b'])
        with pytest.raises(excs.Error, match='arr_3'):
            t.insert(arr_3=[['a'], ['']])

        t.insert(arr_4=np.ones((3, 1, 2), dtype=np.int32))
        t.insert(arr_4=np.ones((3, 0, 2), dtype=np.int32))
        t.insert(arr_4=np.ones((3, 6, 2), dtype=np.int32))
        with pytest.raises(excs.Error, match='arr_4'):
            t.insert(arr_4=np.ones((3, 2), dtype=np.int32))
        with pytest.raises(excs.Error, match='arr_4'):
            t.insert(arr_4=np.ones((3, 1, 2, 4), dtype=np.int32))

        with pytest.raises(TypeError, match=r'Array type parameter must include a dtype.'):
            pxt.Array[1,]  # type: ignore[misc]

    def test_array_unsupported_dtypes(self, init_env: None, reset_db: None) -> None:
        for type_ in [np.generic, np.complex64, ColumnType, object]:
            with pytest.raises(excs.Error, match='Unknown type'):
                pxt.Array[type_]  # type: ignore[misc]
            with pytest.raises(excs.Error, match='Unknown type'):
                pxt.Array[(1,), type_]  # type: ignore[misc]

        with pytest.raises(excs.Error, match=r'use.+pxt.String.+instead'):
            pxt.Array[str]  # type: ignore[misc]
        with pytest.raises(excs.Error, match=r'use.+pxt.Int.+instead'):
            pxt.Array[int]  # type: ignore[misc]
