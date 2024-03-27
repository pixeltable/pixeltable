from typing import Optional

import numpy as np
import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.func import Batch


class TestUdf:

    # Test that various invalid udf definitions generate
    # correct error messages.
    def test_invalid_udfs(self):
        with pytest.raises(excs.Error) as exc_info:
            @pxt.udf
            def udf1(name: Batch[str]) -> str:
                return ''
        assert ': Batched parameter in udf, but no `batch_size` given: `name`' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            @pxt.udf(batch_size=32)
            def udf2(name: Batch[str]) -> str:
                return ''
        assert ': batch_size is specified; Python return type must be a `Batch`' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            @pxt.udf
            def udf3(name: str) -> Optional[np.ndarray]:
                return None
        assert ': Cannot infer pixeltable result type. Specify `return_type` explicitly?' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            @pxt.udf
            def udf4(array: np.ndarray) -> str:
                return ''
        assert ': Cannot infer pixeltable type of parameter: `array`. Specify `param_types` explicitly?' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            @pxt.udf
            def udf5(name: str, untyped) -> str:
                return ''
        assert ': Cannot infer pixeltable types of parameters. Specify `param_types` explicitly?' in str(exc_info.value)
