import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pixeltable as pxt
from pixeltable import exprs
from pixeltable.env import Env

from ..utils import validate_update_status


class TestTimestamp:
    # All test datetimes are in America/Los_Angeles time zone
    TEST_DATETIMES = (
        '2024-01-01T12:34:56-08:00',
        '2023-12-31T23:59:59-08:00',
        '2022-06-15T08:30:00-07:00',
        '2020-02-29T00:00:00-08:00',
        '2019-01-01T00:00:01-08:00',
    )

    def make_test_table(self) -> tuple[list[datetime], pxt.Table]:
        # Create a test table with a date column
        t = pxt.create_table('test_tbl', {'dt': pxt.Timestamp})
        # Insert test data
        test_dts = [datetime.fromisoformat(dt) for dt in self.TEST_DATETIMES]
        validate_update_status(t.insert({'dt': dt} for dt in test_dts), expected_rows=len(test_dts))
        # Trivial validation that rows were inserted correctly
        r = t.collect()
        for irow, rrow in zip(test_dts, r['dt']):
            x = irow.astimezone(timezone.utc)
            y = rrow.astimezone(timezone.utc)
            assert x == y, (irow, rrow)
        return test_dts, t

    def test_timestamp_methods(self, uses_db: None) -> None:
        # Set a default time zone that's likely to be different from the system time zone of most test environments
        default_tz = ZoneInfo('America/Anchorage')
        Env.get().default_time_zone = default_tz
        assert default_tz == Env.get().default_time_zone
        test_dts, t = self.make_test_table()

        from pixeltable.functions.timestamp import (
            day,
            hour,
            isocalendar,
            isoformat,
            isoweekday,
            microsecond,
            minute,
            month,
            posix_timestamp,
            replace,
            second,
            strftime,
            toordinal,
            weekday,
            year,
        )

        test_params: list[tuple[pxt.Function, Callable, list, dict]] = [
            # (pxt_fn, str_fn, args, **kwargs)
            # (date, lambda dt: datetime(dt.year, dt.month, dt.day), [], {}),
            # (time, lambda dt: datetime(1, 1, 1, dt.hour, dt.minute, dt.second, dt.microsecond), [], {}),
            (year, datetime.year.__get__, [], {}),
            (month, datetime.month.__get__, [], {}),
            (day, datetime.day.__get__, [], {}),
            (hour, datetime.hour.__get__, [], {}),
            (minute, datetime.minute.__get__, [], {}),
            (second, datetime.second.__get__, [], {}),
            (microsecond, datetime.microsecond.__get__, [], {}),
            (weekday, datetime.weekday, [], {}),
            (isoweekday, datetime.isoweekday, [], {}),
            (
                isocalendar,
                lambda dt: {'year': dt.isocalendar()[0], 'week': dt.isocalendar()[1], 'weekday': dt.isocalendar()[2]},
                [],
                {},
            ),
            (isoformat, datetime.isoformat, ['T'], {}),
            (strftime, datetime.strftime, ['%Y-%m-%d %H:%M:%S'], {}),
            (replace, datetime.replace, [2025, 1, 1], {}),
            (toordinal, datetime.toordinal, [], {}),
            (posix_timestamp, datetime.timestamp, [], {}),
        ]

        def debug_str() -> str:
            system_tz = time.tzname
            db_dts = [dt.isoformat() for dt in t.select(out=t.dt).collect()['out']]
            return f'system_tz={system_tz}, db_dts={db_dts}'

        for pxt_fn, dt_fn, args, kwargs in test_params:
            print(f'Testing {pxt_fn.name} ...')
            actual = t.select(out=pxt_fn(t.dt, *args, **kwargs)).collect()['out']
            expected = [dt_fn(dt.astimezone(default_tz), *args, **kwargs) for dt in test_dts]
            assert actual == expected, debug_str()
            # Run the same query, forcing the calculations to be done in Python (not SQL)
            # by interposing a non-SQLizable identity function
            actual_py = t.select(
                out=pxt_fn(t.dt.apply(lambda x: x, col_type=pxt.Timestamp), *args, **kwargs)
            ).collect()['out']
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

    def test_timestamp_zones(self, uses_db: None) -> None:
        timestamps = [
            # Some random times in the summer months (to ensure varying DST treatment)
            datetime.fromisoformat('2024-07-01T22:45:12'),
            datetime.fromisoformat('2024-07-01T22:45:12-02:00'),
            datetime(2024, 7, 1, 22, 45, 12, tzinfo=ZoneInfo('Pacific/Auckland')),
        ]
        # Test various methods and properties that are timezone-sensitive
        props_to_test = ['year', 'month', 'day', 'hour', 'minute']
        methods_to_test = ['weekday', 'isoweekday']
        query_time_zone = ZoneInfo('Europe/Berlin')
        # Test time zone conversions several ways: (i) with default time zone equal to
        # system local time, and (ii) with default time zone set to a couple different
        # values, to ensure that we test some values where it's different from
        # the system local time where the test is running.
        for tz_str in [None, 'America/Anchorage', 'America/New_York', 'Asia/Kolkata', 'Asia/Dubai']:
            print(f'Requesting client in time zone: {tz_str}')
            default_time_zone = None if tz_str is None else ZoneInfo(tz_str)
            Env.get().default_time_zone = default_time_zone
            print(f'  (effective default time zone is: {default_time_zone})')

            pxt.drop_table('test_tbl', force=True)
            t = pxt.create_table('test_tbl', {'dt': pxt.Timestamp})
            t.insert({'dt': dt} for dt in timestamps)
            selection = {'dt': t.dt, 'dt_tz': t.dt.astimezone(query_time_zone.key)}
            for prop in props_to_test:
                selection[prop] = getattr(t.dt, prop)
                selection[prop + '_tz'] = getattr(t.dt.astimezone(query_time_zone.key), prop)
            for method in methods_to_test:
                selection[method] = getattr(t.dt, method)()
                selection[method + '_tz'] = getattr(t.dt.astimezone(query_time_zone.key), method)()
            results = t.select(**selection).collect()

            assert len(results) == len(timestamps)
            for row_idx in range(len(timestamps)):
                # `effective_dt` is the aware timestamp that is the Pixeltable interpretation of the input
                # timestamp: if the input timestamp is naive, it is interpreted as being in default_time_zone.
                effective_dt = (
                    timestamps[row_idx]
                    if timestamps[row_idx].tzinfo is not None
                    else timestamps[row_idx].replace(tzinfo=default_time_zone)
                )
                for col in 'dt', 'dt_tz':
                    # Ensure that timestamps are always returned from the database as aware datetimes
                    assert results[col][row_idx].tzinfo is not None
                    # Ensure that they map to the same absolute time as the inputs
                    assert results[col][row_idx].timestamp() == effective_dt.timestamp()
                # Ensure that the 'dt_tz' column is in the correct time zone
                # (the 'dt' column depends on the system time where the test is run!)
                assert results['dt_tz'][row_idx].utcoffset() == timedelta(hours=2)
                # Ensure that the properties are computed correctly based on timezone
                for prop in props_to_test:
                    assert results[prop][row_idx] == getattr(effective_dt.astimezone(default_time_zone), prop)
                    assert results[prop + '_tz'][row_idx] == getattr(effective_dt.astimezone(query_time_zone), prop)
                for method in methods_to_test:
                    assert results[method][row_idx] == getattr(effective_dt.astimezone(default_time_zone), method)()
                    assert (
                        results[method + '_tz'][row_idx] == getattr(effective_dt.astimezone(query_time_zone), method)()
                    )

    def test_timestamp_zone_in_literals(self, uses_db: None) -> None:
        Env.get().default_time_zone = ZoneInfo('America/Anchorage')
        t = pxt.create_table('test_tbl', {'n': pxt.Int, 'dt': pxt.Timestamp})
        start = datetime.fromisoformat('2024-07-01T00:00:00+00:00')
        validate_update_status(
            t.insert({'n': n, 'dt': start + timedelta(minutes=n)} for n in range(1440)), expected_rows=1440
        )
        # Ensure literals are displayed correctly in the default TZ (with naive datetimes interpreted
        # as being in the default time zone)
        assert str(t.dt >= datetime.fromisoformat('2024-07-01T00:00:00')) == "dt >= '2024-07-01T00:00:00-08:00'"
        assert str(t.dt >= datetime.fromisoformat('2024-07-01T00:00:00-04:00')) == "dt >= '2024-06-30T20:00:00-08:00'"
        assert t.where(t.dt >= datetime.fromisoformat('2024-07-01T00:00:00')).count() == 960
        assert t.where(t.dt >= datetime.fromisoformat('2024-07-01T00:00:00-04:00')).count() == 1200

    def test_isoformat_edge_cases(self, uses_db: None) -> None:
        """Test isoformat SQL/Python equivalence, focusing on the 'auto' timespec branch.

        The SQL implementation branches on whether sub-second microseconds are present.
        We need datetimes both with and without microseconds to exercise both branches.
        """
        from pixeltable.functions.timestamp import isoformat

        default_tz = ZoneInfo('America/Anchorage')
        Env.get().default_time_zone = default_tz

        # Include datetimes both with and without microseconds, and with boundary values
        raw_dts = [
            # No microseconds — SQL should produce 'HH24:MI:SS+TZ'
            datetime.fromisoformat('2024-01-01T12:34:56-08:00'),
            datetime.fromisoformat('2023-12-31T23:59:59-08:00'),
            datetime.fromisoformat('2020-02-29T00:00:00-08:00'),  # leap day, midnight
            # With microseconds — SQL should produce 'HH24:MI:SS.US+TZ'
            datetime(2024, 6, 15, 8, 30, 0, 123456, tzinfo=ZoneInfo('America/Los_Angeles')),
            datetime(2023, 7, 4, 0, 0, 0, 500000, tzinfo=timezone.utc),  # exactly 0.5 s
            datetime(2022, 1, 1, 12, 0, 0, 1, tzinfo=timezone.utc),  # single microsecond
            datetime(2022, 1, 1, 12, 0, 0, 999999, tzinfo=timezone.utc),  # max microseconds
        ]
        t = pxt.create_table('test_tbl', {'dt': pxt.Timestamp})
        validate_update_status(t.insert({'dt': dt} for dt in raw_dts), expected_rows=len(raw_dts))

        # Convert to the default timezone for expected-value computation (matches Pixeltable behaviour)
        test_dts = [dt.astimezone(default_tz) for dt in raw_dts]

        def check_sql_and_py(*args: Any, **kwargs: Any) -> tuple[list[Any], list[Any]]:
            res_sql = t.select(out=isoformat(t.dt, *args, **kwargs)).collect()['out']
            res_py = t.select(
                out=isoformat(t.dt.apply(lambda x: x, col_type=pxt.Timestamp), *args, **kwargs)
            ).collect()['out']
            return res_sql, res_py

        # Default sep='T', timespec='auto' — exercises both branches of the microseconds CASE
        res_sql, res_py = check_sql_and_py()
        expected = [dt.isoformat() for dt in test_dts]
        assert res_sql == expected, f'isoformat default SQL: {res_sql} != {expected}'
        assert res_py == expected, f'isoformat default Py: {res_py} != {expected}'

        # Microsecond branch: rows without microseconds must NOT have a decimal point
        no_us_idx = [i for i, dt in enumerate(test_dts) if dt.microsecond == 0]
        for i in no_us_idx:
            assert '.' not in res_sql[i].split('T')[1], (
                f'isoformat should omit microseconds for dt={test_dts[i]}, got {res_sql[i]!r}'
            )

        # Microsecond branch: rows with microseconds MUST have a 6-digit decimal part
        us_idx = [i for i, dt in enumerate(test_dts) if dt.microsecond != 0]
        for i in us_idx:
            time_part = res_sql[i].split('T')[1]
            assert '.' in time_part, f'isoformat missing microseconds for dt={test_dts[i]}, got {res_sql[i]!r}'
            decimal_digits = time_part.split('.')[1][:6]  # strip trailing tz offset
            assert len(decimal_digits) == 6, f'expected 6 microsecond digits, got {res_sql[i]!r}'

        # Custom separator
        res_sql, res_py = check_sql_and_py(sep=' ')
        expected = [dt.isoformat(sep=' ') for dt in test_dts]
        assert res_sql == expected, f'isoformat sep=" " SQL: {res_sql} != {expected}'
        assert res_py == expected

        # Explicit timespec falls back to Python — result must still be correct
        res_seconds = t.select(out=isoformat(t.dt, timespec='seconds')).collect()['out']
        assert res_seconds == [dt.isoformat(timespec='seconds') for dt in test_dts]

        res_ms = t.select(out=isoformat(t.dt, timespec='milliseconds')).collect()['out']
        assert res_ms == [dt.isoformat(timespec='milliseconds') for dt in test_dts]

    def test_strftime_formats(self, uses_db: None) -> None:
        """Test strftime SQL/Python equivalence across common format codes and edge cases.

        The SQL implementation translates literal Python strftime format strings to
        PostgreSQL to_char patterns at query-planning time.  Unsupported codes and
        non-literal format columns fall back to Python execution.
        """
        from pixeltable.functions.timestamp import strftime

        default_tz = ZoneInfo('America/Anchorage')
        Env.get().default_time_zone = default_tz

        raw_dts = [
            datetime.fromisoformat('2024-01-01T00:00:00+00:00'),  # midnight UTC / New Year
            datetime.fromisoformat('2024-07-04T12:30:45+00:00'),  # afternoon, mid-year
            datetime.fromisoformat('2020-02-29T08:15:00+00:00'),  # leap day
            datetime(2023, 11, 6, 23, 59, 59, 123456, tzinfo=timezone.utc),  # with microseconds
            datetime(2022, 12, 31, 11, 0, 0, tzinfo=timezone.utc),  # AM hour
        ]
        t = pxt.create_table('test_tbl', {'dt': pxt.Timestamp})
        validate_update_status(t.insert({'dt': dt} for dt in raw_dts), expected_rows=len(raw_dts))

        test_dts = [dt.astimezone(default_tz) for dt in raw_dts]

        def check_sql_and_py(fmt: str) -> tuple[list, list]:
            res_sql = t.select(out=strftime(t.dt, fmt)).collect()['out']
            res_py = t.select(out=strftime(t.dt.apply(lambda x: x, col_type=pxt.Timestamp), fmt)).collect()['out']
            return res_sql, res_py

        # Basic date/time components
        for fmt in [
            '%Y',  # 4-digit year
            '%y',  # 2-digit year
            '%m',  # zero-padded month
            '%d',  # zero-padded day
            '%H',  # 24h hour
            '%M',  # minute
            '%S',  # second
            '%Y-%m-%d',  # ISO date
            '%Y-%m-%d %H:%M:%S',  # full datetime
            '%H:%M',  # time only
            '%j',  # day of year
        ]:
            res_sql, res_py = check_sql_and_py(fmt)
            expected = [dt.strftime(fmt) for dt in test_dts]
            assert res_sql == expected, f'strftime SQL fmt={fmt!r}: {res_sql} != {expected}'
            assert res_py == expected, f'strftime Py fmt={fmt!r}: {res_py} != {expected}'

        # Microseconds
        res_sql, res_py = check_sql_and_py('%f')
        expected = [dt.strftime('%f') for dt in test_dts]
        assert res_sql == expected, f'strftime %f SQL: {res_sql} != {expected}'
        assert res_py == expected

        # AM/PM with 12-hour clock
        res_sql, res_py = check_sql_and_py('%I:%M %p')
        expected = [dt.strftime('%I:%M %p') for dt in test_dts]
        assert res_sql == expected, f'strftime %I:%M %p SQL: {res_sql} != {expected}'
        assert res_py == expected

        # Day and month names — both full and abbreviated
        res_sql, res_py = check_sql_and_py('%A')
        expected = [dt.strftime('%A') for dt in test_dts]
        assert res_sql == expected, f'strftime %A SQL: {res_sql} != {expected}'
        assert res_py == expected

        res_sql, res_py = check_sql_and_py('%a')
        expected = [dt.strftime('%a') for dt in test_dts]
        assert res_sql == expected, f'strftime %a SQL: {res_sql} != {expected}'
        assert res_py == expected

        res_sql, res_py = check_sql_and_py('%B')
        expected = [dt.strftime('%B') for dt in test_dts]
        assert res_sql == expected, f'strftime %B SQL: {res_sql} != {expected}'
        assert res_py == expected

        res_sql, res_py = check_sql_and_py('%b')
        expected = [dt.strftime('%b') for dt in test_dts]
        assert res_sql == expected, f'strftime %b SQL: {res_sql} != {expected}'
        assert res_py == expected

        # Literal text mixed with format codes — alphabetic literals must be double-quoted in PG
        res_sql, res_py = check_sql_and_py('Year: %Y, Month: %m')
        expected = [dt.strftime('Year: %Y, Month: %m') for dt in test_dts]
        assert res_sql == expected, 'strftime literal text SQL'
        assert res_py == expected

        # Double-percent produces a literal '%'
        res_sql, res_py = check_sql_and_py('100%%')
        expected = [dt.strftime('100%%') for dt in test_dts]
        assert res_sql == expected, f'strftime 100%% SQL: {res_sql} != {expected}'
        assert res_py == expected

        # Unsupported format code (%z) falls back to Python — must still be correct
        res_fallback = t.select(out=strftime(t.dt, '%z')).collect()['out']
        assert res_fallback == [dt.strftime('%z') for dt in test_dts], 'strftime %z fallback'

        # Non-literal format column also falls back to Python — must still be correct
        t2 = pxt.create_table('test_tbl2', {'dt': pxt.Timestamp, 'fmt': pxt.String})
        validate_update_status(t2.insert({'dt': dt, 'fmt': '%Y-%m-%d'} for dt in raw_dts), expected_rows=len(raw_dts))
        res_col = t2.select(out=strftime(t2.dt, t2.fmt)).collect()['out']
        assert res_col == [dt.strftime('%Y-%m-%d') for dt in test_dts], 'strftime column format'

    def test_timestamp_make(self, uses_db: None) -> None:
        Env.get().default_time_zone = ZoneInfo('America/Anchorage')
        test_dts, t = self.make_test_table()
        from pixeltable.functions.timestamp import make_timestamp

        res = (
            t.select(
                out=make_timestamp(
                    year=t.dt.year,
                    month=t.dt.month,
                    day=t.dt.day,
                    hour=t.dt.hour,
                    # omit minute in order to force FunctionCall.sql_expr() to deal with kw args
                    second=t.dt.second,
                )
            )
            # .order_by(t.dt.day, asc=False)
            .collect()
        )
        assert res['out'] == [dt.replace(minute=0).astimezone(Env.get().default_time_zone) for dt in test_dts]
