from datetime import datetime
from typing import Callable

import pixeltable as pxt

from ..utils import validate_update_status


class TestTimestamp:
    TEST_DATETIMES = [
        "2024-01-01T12:34:56",
        "2023-12-31T23:59:59",
        "2022-06-15T08:30:00",
        "2020-02-29T00:00:00",
        "2019-01-01T00:00:01"
    ]

    def test_methods(self, reset_db) -> None:
        t = pxt.create_table('test_tbl', {'dt': pxt.TimestampType()})
        test_dts = [datetime.fromisoformat(dt) for dt in self.TEST_DATETIMES]
        validate_update_status(t.insert({'dt': dt} for dt in test_dts), expected_rows=len(test_dts))

        from pixeltable.functions.timestamp import (day, hour, isocalendar,
                                                    isoformat, isoweekday,
                                                    microsecond, minute, month,
                                                    posix_timestamp, replace,
                                                    second, strftime,
                                                    toordinal, weekday, year)

        test_params: list[tuple[pxt.Function, Callable, list, dict]] = [
            # (pxt_fn, str_fn, args, kwargs)
            #(date, lambda dt: datetime(dt.year, dt.month, dt.day), [], {}),
            #(time, lambda dt: datetime(1, 1, 1, dt.hour, dt.minute, dt.second, dt.microsecond), [], {}),
            (year, datetime.year.__get__, [], {}),
            (month, datetime.month.__get__, [], {}),
            (day, datetime.day.__get__, [], {}),
            (hour, datetime.hour.__get__, [], {}),
            (minute, datetime.minute.__get__, [], {}),
            (second, datetime.second.__get__, [], {}),
            (microsecond, datetime.microsecond.__get__, [], {}),
            (weekday, datetime.weekday, [], {}),
            (isoweekday, datetime.isoweekday, [], {}),
            (isocalendar,
             lambda dt: {'year': dt.isocalendar()[0], 'week': dt.isocalendar()[1], 'weekday': dt.isocalendar()[2]},
             [],
             {}),
            (isoformat, datetime.isoformat, ['T'], {}),
            (strftime, datetime.strftime, ['%Y-%m-%d %H:%M:%S'], {}),
            (replace, datetime.replace, [2025, 1, 1], {}),
            (toordinal, datetime.toordinal, [], {}),
            (posix_timestamp, datetime.timestamp, [], {}),
        ]

        for pxt_fn, dt_fn, args, kwargs in test_params:
            assert (t.select(out=pxt_fn(t.dt, *args, **kwargs)).collect()['out']
                == [dt_fn(dt, *args, **kwargs) for dt in test_dts]), pxt_fn.name

        # Check that they can all be called with method syntax too
        for pxt_fn, _, _, _ in test_params:
            mref = t.dt.__getattr__(pxt_fn.name)
            if pxt_fn.is_property:
                assert isinstance(mref, pxt.exprs.FunctionCall)
            else:
                assert isinstance(mref, pxt.exprs.MethodRef)
                assert mref.method_name == pxt_fn.name, pxt_fn.name

    def test_make_ts(self, reset_db) -> None:
        t = pxt.create_table('test_tbl', {'dt': pxt.TimestampType()})
        test_dts = [datetime.fromisoformat(dt) for dt in self.TEST_DATETIMES]
        validate_update_status(t.insert({'dt': dt} for dt in test_dts), expected_rows=len(test_dts))
        from pixeltable.functions.timestamp import make_timestamp
        res = t.select(
            out=make_timestamp(
                year=t.dt.year, month=t.dt.month, day=t.dt.day, hour=t.dt.hour,
                # omit minute in order to force FunctionCall.sql_expr() to deal with kw args
                second=t.dt.second)
        ).collect()
        assert (
            res['out'] == [datetime(dt.year, dt.month, dt.day, dt.hour, minute=0, second=dt.second) for dt in test_dts]
        )
