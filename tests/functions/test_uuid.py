import uuid
from typing import Callable

import pixeltable as pxt
import pixeltable.functions as pxtf

from ..utils import validate_update_status


class TestUuid:
    def test_uuid_methods(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'id': pxt.UUID})

        # Create some test UUIDs
        test_uuids = [uuid.uuid4() for _ in range(3)]
        validate_update_status(t.insert({'id': u} for u in test_uuids), expected_rows=len(test_uuids))

        test_params: list[tuple[pxt.Function, Callable, list, dict]] = [
            # (pxt_fn, py_fn, args, **kwargs)
            (pxtf.uuid.to_string, str, [], {}),
            (pxtf.uuid.hex, uuid.UUID.hex.__get__, [], {}),
        ]

        for pxt_fn, py_fn, args, kwargs in test_params:
            print(f'Testing {pxt_fn.name} ...')
            actual = t.select(out=pxt_fn(t.id, *args, **kwargs)).collect()['out']
            expected = [py_fn(id, *args, **kwargs) for id in test_uuids]
            assert actual == expected
            # Run the same query, forcing the calculations to be done in Python (not SQL)
            # by interposing a non-SQLizable identity function
            actual_py = t.select(out=pxt_fn(t.id.apply(lambda x: x, col_type=pxt.UUID), *args, **kwargs)).collect()[
                'out'
            ]
            assert actual_py == expected
