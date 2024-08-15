from datetime import datetime, timedelta, tzinfo
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import pytest

import pixeltable as pxt
from pixeltable.env import Env

from ..utils import validate_update_status


class TestTimestamp:
    TEST_DATETIMES = [
        "2024-01-01T12:34:56-08:00",
        "2023-12-31T23:59:59-08:00",
        "2022-06-15T08:30:00-07:00",
        "2020-02-29T00:00:00-08:00",
        "2019-01-01T00:00:01-08:00"
    ]

    def test_all(self, reset_db) -> None:
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
                == [dt_fn(dt, *args, **kwargs) for dt in test_dts])

        # Check that they can all be called with method syntax too
        for pxt_fn, _, _, _ in test_params:
            mref = t.dt.__getattr__(pxt_fn.name)
            if isinstance(mref, pxt.exprs.MethodRef):
                # method
                assert mref.method_name == pxt_fn.name, pxt_fn
            elif isinstance(mref, pxt.exprs.FunctionCall):
                # property
                assert mref.fn.name == pxt_fn.name, pxt_fn
            else:
                assert False

    def test_time_zones(self, reset_db) -> None:
        timestamps = [
            datetime.fromisoformat('2024-07-01T22:45:12'),
            datetime.fromisoformat('2024-07-01T22:45:12-02:00'),
            datetime(2024, 7, 1, 22, 45, 12, tzinfo=ZoneInfo('Pacific/Auckland'))
        ]
        # Test various methods and properties that are timezone-sensitive
        props_to_test = ['year', 'month', 'day', 'hour', 'minute']
        methods_to_test = ['weekday', 'isoweekday']
        query_time_zone = ZoneInfo('Europe/London')
        # Test time zone conversions several ways: (i) with default time zone equal to
        # system local time, and (ii) with default time zone set to a couple different
        # values, to ensure that we test some values where it's different from
        # the system local time where the test is running.
        for tz_str in [None, 'America/Anchorage', 'America/New_York', 'Asia/Kolkata', 'Asia/Dubai']:
            print(f'Testing with default time zone equal to: {tz_str}')
            default_time_zone = None if tz_str is None else ZoneInfo(tz_str)
            Env.get().default_time_zone = default_time_zone

            pxt.drop_table('test_tbl', force=True)
            t = pxt.create_table('test_tbl', {'dt': pxt.TimestampType()})
            t.insert({'dt': dt} for dt in timestamps)
            selection = {'dt': t.dt, 'dt_tz': t.dt.to_tz(query_time_zone.key)}
            for prop in props_to_test:
                selection[prop] = getattr(t.dt, prop)
                selection[prop + '_tz'] = getattr(t.dt.to_tz(query_time_zone.key), prop)
            for method in methods_to_test:
                selection[method] = getattr(t.dt, method)()
                selection[method + '_tz'] = getattr(t.dt.to_tz(query_time_zone.key), method)()
            results = t.select(**selection).collect()

            assert len(results) == len(timestamps)
            for row in range(len(timestamps)):
                # `effective_dt` is the aware timestamp that is the Pixeltable interpretation of the input
                # timestamp: if the input timestamp is naive, it is interpreted as being in default_time_zone.
                effective_dt = (
                    timestamps[row] if timestamps[row].tzinfo is not None
                    else timestamps[row].replace(tzinfo=default_time_zone)
                )
                for col in 'dt', 'dt_tz':
                    # Ensure that timestamps are always returned from the database as aware datetimes
                    assert results[col][row].tzinfo is not None
                    # Ensure that they map to the same absolute time as the inputs
                    assert results[col][row].timestamp() == effective_dt.timestamp()
                # Ensure that the 'dt_tz' column is in the correct time zone
                # (the 'dt' column depends on the system time where the test is run!)
                assert results['dt_tz'][row].utcoffset() == timedelta(hours=1)
                # Ensure that the properties are computed correctly based on timezone
                for prop in props_to_test:
                    assert results[prop][row] == getattr(effective_dt.astimezone(default_time_zone), prop)
                    assert results[prop + '_tz'][row] == getattr(effective_dt.astimezone(query_time_zone), prop)
                for method in methods_to_test:
                    assert results[method][row] == getattr(effective_dt.astimezone(default_time_zone), method)()
                    assert results[method + '_tz'][row] == getattr(effective_dt.astimezone(query_time_zone), method)()

    def test_time_zone_in_literals(self, reset_db) -> None:
        Env.get().default_time_zone = ZoneInfo('America/Anchorage')
        t = pxt.create_table('test_tbl', {'n': pxt.IntType(), 'dt': pxt.TimestampType()})
        start = datetime.fromisoformat('2024-07-01T00:00:00+00:00')
        validate_update_status(
            t.insert({'n': n, 'dt': start + timedelta(minutes=n)} for n in range(1440)),
            expected_rows=1440
        )
        # Ensure literals are displayed correctly in the default TZ (with naive datetimes interpreted
        # as being in the default time zone)
        assert str(t.dt >= datetime.fromisoformat('2024-07-01T00:00:00')) == "dt >= '2024-07-01T00:00:00-08:00'"
        assert str(t.dt >= datetime.fromisoformat('2024-07-01T00:00:00-04:00')) == "dt >= '2024-06-30T20:00:00-08:00'"
        assert t.where(t.dt >= datetime.fromisoformat('2024-07-01T00:00:00')).count() == 960
        assert t.where(t.dt >= datetime.fromisoformat('2024-07-01T00:00:00-04:00')).count() == 1200
