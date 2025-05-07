import time
from datetime import date
from typing import Callable
from zoneinfo import ZoneInfo

import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs, exprs
from pixeltable.env import Env
from pixeltable.functions.date import (
    day,
    isocalendar,
    isoformat,
    isoweekday,
    make_date,
    month,
    strftime,
    toordinal,
    weekday,
    year,
)

from ..utils import validate_update_status


class TestDate:
    # All test datetimes are in America/Los_Angeles time zone
    TEST_DATES = ('2024-01-01', '2023-12-31', '2022-06-15', '2020-02-29', '2019-01-01')

    def make_test_table(self) -> tuple[list[date], pxt.Table]:
        # Create a test table with a date column
        t = pxt.create_table('test_tbl', {'dt': pxt.Date})
        # Insert test data
        test_dts = [date.fromisoformat(dt) for dt in self.TEST_DATES]
        validate_update_status(t.insert({'dt': dt} for dt in test_dts), expected_rows=len(test_dts))
        return test_dts, t

    def test_date_methods(self, reset_db: None) -> None:
        # Set a default time zone that's likely to be different from the system time zone of most test environments
        default_tz = ZoneInfo('America/Anchorage')
        Env.get().default_time_zone = default_tz

        test_dts, t = self.make_test_table()

        test_params: list[tuple[pxt.Function, Callable, list, dict]] = [
            # (pxt_fn, str_fn, args, **kwargs)
            # (date, lambda dt: datetime(dt.year, dt.month, dt.day), [], {}),
            # (time, lambda dt: datetime(1, 1, 1, dt.hour, dt.minute, dt.second, dt.microsecond), [], {}),
            (year, date.year.__get__, [], {}),
            (month, date.month.__get__, [], {}),
            (day, date.day.__get__, [], {}),
            (weekday, date.weekday, [], {}),
            (isoweekday, date.isoweekday, [], {}),
            (
                isocalendar,
                lambda dt: {'year': dt.isocalendar()[0], 'week': dt.isocalendar()[1], 'weekday': dt.isocalendar()[2]},
                [],
                {},
            ),
            (isoformat, date.isoformat, [], {}),
            (strftime, date.strftime, ['%Y-%m-%d'], {}),
            (toordinal, date.toordinal, [], {}),
        ]

        def debug_str() -> str:
            system_tz = time.tzname
            db_dts = [dt.isoformat() for dt in t.select(out=t.dt).collect()['out']]
            return f'system_tz={system_tz}, db_dts={db_dts}'

        for pxt_fn, dt_fn, args, kwargs in test_params:
            print(f'Testing {pxt_fn.name} ...')
            actual = t.select(out=pxt_fn(t.dt, *args, **kwargs)).collect()['out']
            expected = [dt_fn(dt, *args, **kwargs) for dt in test_dts]
            assert actual == expected, debug_str()
            # Run the same query, forcing the calculations to be done in Python (not SQL)
            # by interposing a non-SQLizable identity function
            actual_py = t.select(out=pxt_fn(t.dt.apply(lambda x: x, col_type=pxt.Date), *args, **kwargs)).collect()[
                'out'
            ]
            assert actual_py == expected, debug_str()

        # Check that they can all be called with method syntax too
        for pxt_fn, _, _, _ in test_params:
            mref = getattr(t.dt, pxt_fn.name)
            if isinstance(mref, exprs.MethodRef):
                # method
                assert mref.method_name == pxt_fn.name, pxt_fn
            elif isinstance(mref, exprs.FunctionCall):
                # property
                assert mref.fn.name == pxt_fn.name, pxt_fn
            else:
                raise AssertionError()

    def test_date_make(self, reset_db: None) -> None:
        Env.get().default_time_zone = ZoneInfo('America/Anchorage')
        test_dts, t = self.make_test_table()

        res = (
            t.select(out=make_date(year=t.dt.year, month=t.dt.month, day=t.dt.day))
            # .order_by(t.dt.day, asc=False)
            .collect()
        )

        assert res['out'] == test_dts

    def test_date_arith(self, reset_db: None) -> None:
        _, t = self.make_test_table()

        with pytest.raises(excs.Error, match='requires numeric types'):
            _ = t.select(t.dt, outp=t.dt + 1)
        with pytest.raises(excs.Error, match='requires numeric types'):
            _ = t.select(t.dt, outm=t.dt - 1)
