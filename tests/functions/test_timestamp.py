import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable import exprs
from pixeltable.env import Env

from ..utils import pxt_raises, reload_catalog, validate_update_status

pytestmark = pytest.mark.db_roots('local', reason='UDF/integration test')


class TestTimestamp:
    # All test datetimes are in America/Los_Angeles time zone
    TEST_DATETIMES = (
        '2024-01-01T12:34:56-08:00',
        '2023-12-31T23:59:59-08:00',
        '2022-06-15T08:30:00-07:00',
        '2020-02-29T00:00:00-08:00',
        '2019-01-01T00:00:01-08:00',
    )
    LA = ZoneInfo('America/Los_Angeles')
    # daylight saving time starts in America/Los_Angeles and America/Anchorage on 2026-03-08
    BEFORE_DST = datetime(2026, 3, 7, 12, 0, tzinfo=LA)

    def make_test_table(self) -> tuple[list[datetime], pxt.Table]:
        # Create a test table with a date column
        t = pxt.create_table('test_tbl', {'dt': pxt.Timestamp | None})
        # Insert test data
        test_dts = sorted([datetime.fromisoformat(dt) for dt in self.TEST_DATETIMES])
        validate_update_status(t.insert({'dt': dt} for dt in test_dts), expected_rows=len(test_dts))
        # Trivial validation that rows were inserted correctly
        r = t.order_by(t.dt).collect()
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
            date,
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
            (date, datetime.date, [], {}),
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
            actual = t.select(out=pxt_fn(t.dt, *args, **kwargs)).order_by(t.dt).collect()['out']
            expected = [dt_fn(dt.astimezone(default_tz), *args, **kwargs) for dt in test_dts]
            assert actual == expected, debug_str()
            # Run the same query, forcing the calculations to be done in Python (not SQL)
            # by interposing a non-SQLizable identity function
            actual_py = (
                t.select(out=pxt_fn(t.dt.apply(lambda x: x, col_type=pxt.Timestamp), *args, **kwargs))
                .order_by(t.dt)
                .collect()['out']
            )
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

    def test_microsecond(self, uses_db: None) -> None:
        # Regression test for the `microsecond` SQL translation with nonzero sub-second values.
        # Includes fractional seconds >= 0.5, where rounding the `second` field to an integer would
        # previously produce negative results.
        from pixeltable.functions.timestamp import microsecond

        test_dts = [
            datetime.fromisoformat(f'2024-01-01T12:34:56.{frac}-08:00')
            for frac in ('000000', '123456', '500000', '789456', '999999')
        ]
        t = pxt.create_table('test_tbl', {'dt': pxt.Timestamp | None})
        validate_update_status(t.insert({'dt': dt} for dt in test_dts), expected_rows=len(test_dts))

        expected = [dt.microsecond for dt in test_dts]
        # SQL path
        actual = t.order_by(t.dt).select(out=microsecond(t.dt)).collect()['out']
        assert actual == expected, actual
        # Python path (interpose a non-SQLizable identity function to force in-Python evaluation)
        actual_py = (
            t.order_by(t.dt).select(out=microsecond(t.dt.apply(lambda x: x, col_type=pxt.Timestamp))).collect()['out']
        )
        assert actual_py == expected, actual_py

    def make_arithmetic_table(self) -> tuple[list[dict[str, Any]], pxt.Table]:
        Env.get().default_time_zone = ZoneInfo('America/Anchorage')
        t = pxt.create_table(
            'test_tbl',
            {
                'id': pxt.Int,
                'dt': pxt.Timestamp | None,
                'end': pxt.Timestamp | None,
                'i': pxt.Int | None,
                'f': pxt.Float | None,
                'j': pxt.Json | None,
            },
        )
        la = self.LA
        rows: list[dict[str, Any]] = [
            {'dt': self.BEFORE_DST, 'end': datetime(2026, 3, 8, 12, 0, tzinfo=la), 'i': 86400, 'f': 86400.0},
            {'dt': datetime(2024, 1, 1, 0, 0, 0, 123457, tzinfo=la), 'end': self.BEFORE_DST, 'i': -1, 'f': 1.5e-6},
            {'dt': None, 'end': self.BEFORE_DST, 'i': 1, 'f': 1.0},
            {'dt': self.BEFORE_DST, 'end': None, 'i': None, 'f': None},
        ]
        # the last four offsets end in half a microsecond
        fs = [-0.25, 86400.0000005, -86400.0000005, 86400.0000015, 12345.6789015]
        rows.extend(
            {
                'dt': datetime.fromisoformat(dt),
                'end': datetime(2025, 5, 17, 3, 4, 5, 678901, tzinfo=la),
                'i': 90,
                'f': f,
            }
            for dt, f in zip(self.TEST_DATETIMES, fs)
        )
        for row in rows:
            row['j'] = {'ts': '2025-11-02T01:30:00-07:00'}
        validate_update_status(t.insert({'id': i, **row} for i, row in enumerate(rows)), expected_rows=len(rows))
        return rows, t

    def shifted(self, dt: datetime | None, seconds: float | None) -> datetime | None:
        if dt is None or seconds is None:
            return None
        return dt.astimezone(timezone.utc) + timedelta(seconds=seconds)

    def test_timestamp_arithmetic(self, uses_db: None) -> None:
        rows, t = self.make_arithmetic_table()
        sql_dt = t.dt
        # a non-SQLizable identity function forces evaluation in Python
        py_dt = t.dt.apply(lambda x: x, col_type=pxt.Timestamp)
        timedeltas = [
            timedelta(hours=1),
            timedelta(days=1, microseconds=1),
            timedelta(days=-400, seconds=5),
            timedelta(0),
        ]
        for dt_expr in (sql_dt, py_dt):
            for td in timedeltas:
                res = t.select(plus=dt_expr + td, rplus=td + dt_expr, minus=dt_expr - td).order_by(t.id).collect()
                expected_plus = [self.shifted(row['dt'], td.total_seconds()) for row in rows]
                assert res['plus'] == expected_plus, td
                assert res['rplus'] == expected_plus, td
                assert res['minus'] == [self.shifted(row['dt'], -td.total_seconds()) for row in rows], td

            # row 0: one day after noon on the day before the DST change is 13:00 wall-clock time
            res = (
                t.where(t.id == 0).select(out=dt_expr + timedelta(days=1), since=t.end.seconds_since(dt_expr)).collect()
            )
            assert res['out'] == [datetime(2026, 3, 8, 13, 0, tzinfo=self.LA)]
            assert res['since'] == [82800.0]

            res = (
                t.select(
                    i=dt_expr.add_seconds(t.i),
                    f=dt_expr.add_seconds(t.f),
                    since=t.end.seconds_since(dt_expr),
                    rsince=dt_expr.seconds_since(t.end),
                    posix=dt_expr.posix_timestamp(),
                )
                .order_by(t.id)
                .collect()
            )
            assert res['i'] == [self.shifted(row['dt'], row['i']) for row in rows]
            assert res['f'] == [self.shifted(row['dt'], row['f']) for row in rows]
            expected_since = [
                None
                if row['dt'] is None or row['end'] is None
                else (row['end'].astimezone(timezone.utc) - row['dt'].astimezone(timezone.utc)).total_seconds()
                for row in rows
            ]
            assert res['since'] == expected_since
            assert res['rsince'] == [None if s is None else -s for s in expected_since]
            assert res['posix'] == [None if row['dt'] is None else row['dt'].timestamp() for row in rows]

    def test_timestamp_arithmetic_exprs(self, uses_db: None) -> None:
        rows, t = self.make_arithmetic_table()

        # strip_timezone() returns the wall-clock time in the default time zone
        naive_dt = t.dt.strip_timezone()
        res = (
            t.select(
                plus=naive_dt + timedelta(days=1),
                since=naive_dt.seconds_since(t.dt),
                rsince=t.dt.seconds_since(naive_dt),
            )
            .order_by(t.id)
            .collect()
        )
        default_tz = Env.get().default_time_zone
        assert res['plus'] == [
            None if row['dt'] is None else self.shifted(row['dt'], 86400).astimezone(default_tz).replace(tzinfo=None)
            for row in rows
        ]
        assert res['since'] == [None if row['dt'] is None else 0.0 for row in rows]
        assert res['rsince'] == res['since']

        res = t.select(out=t.j.ts.astype(pxt.Timestamp) + timedelta(hours=1)).order_by(t.id).collect()
        assert res['out'] == [self.shifted(datetime.fromisoformat(row['j']['ts']), 3600) for row in rows]

        # TODO: PXT-1825: run these filters on a Timestamp expression evaluated in Python as well
        week_ago = pxtf.timestamp.now() - timedelta(days=7)
        assert t.where(t.dt < week_ago).count() == sum(row['dt'] is not None for row in rows)
        assert t.where(t.dt > week_ago).count() == 0
        cutoff = datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp()
        assert t.where(t.dt.posix_timestamp() < cutoff).count() == sum(
            row['dt'] is not None and row['dt'].timestamp() < cutoff for row in rows
        )

        assert str(t.dt - timedelta(hours=1)) == 'add_seconds(dt, -3600.0)'

        t.add_computed_column(later=t.dt + timedelta(hours=1))
        reload_catalog()
        t = pxt.get_table('test_tbl')
        validate_update_status(t.insert([{'id': len(rows), 'dt': self.BEFORE_DST}]), expected_rows=1)
        res = t.select(t.later).order_by(t.id).collect()
        assert res['later'] == [self.shifted(row['dt'], 3600) for row in rows] + [self.shifted(self.BEFORE_DST, 3600)]

    def test_timestamp_arithmetic_errors(self, uses_db: None) -> None:
        _, t = self.make_arithmetic_table()
        seconds_since_hint = r'- is not defined for two Timestamps; `end.seconds_since\(start\)`'
        td = timedelta(hours=1)
        errors: list[tuple[Callable[[], exprs.Expr], str]] = [
            (lambda: t.end - t.dt, seconds_since_hint),
            (lambda: t.dt - self.BEFORE_DST, seconds_since_hint),
            (lambda: self.BEFORE_DST - t.dt, seconds_since_hint),
            (lambda: td - t.dt, 'timedelta - dt: a timedelta has to be the right operand of -'),
            (lambda: t.i + td, r'i: a timedelta can only be added .* has type `Int \| None`'),
            (lambda: td + t.f, r'f: a timedelta can only be added .* has type `Float \| None`'),
            (lambda: t.j.ts - td, r'j.ts: a timedelta can only be .* `astype\(pxt.Timestamp\)`'),
            (
                lambda: t.end.seconds_since(t.dt) < td,
                r'< is not defined for a timedelta; `timedelta.total_seconds\(\)`',
            ),
        ]
        for make_expr, match in errors:
            with pxt_raises(pxt.ErrorCode.TYPE_MISMATCH, match=match):
                _ = make_expr()

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
            t = pxt.create_table('test_tbl', {'dt': pxt.Timestamp | None})
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
        t = pxt.create_table('test_tbl', {'n': pxt.Int | None, 'dt': pxt.Timestamp | None})
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
            .order_by(t.dt, asc=True)
            .collect()
        )
        assert res['out'] == [dt.replace(minute=0).astimezone(Env.get().default_time_zone) for dt in test_dts]

    def test_now(self, uses_db: None) -> None:
        default_tz = ZoneInfo('America/Anchorage')
        Env.get().default_time_zone = default_tz
        from pixeltable.functions.timestamp import now

        t = pxt.create_table('test_tbl', {'n': pxt.Int})
        t.add_computed_column(inserted_at=now())
        t.add_computed_column(queried_at=now(), stored=False)
        before_insert = datetime.now(timezone.utc)
        validate_update_status(t.insert({'n': n} for n in range(3)), expected_rows=3)
        after_insert = datetime.now(timezone.utc)
        res = t.select(t.inserted_at, t.queried_at, selected_at=now()).collect()
        after_query = datetime.now(timezone.utc)
        assert all(before_insert <= ts <= after_insert for ts in res['inserted_at'])
        assert all(after_insert <= ts <= after_query for ts in res['queried_at'] + res['selected_at'])

        evaluated = pxt.eval_udf(now)
        assert after_query <= evaluated <= datetime.now(timezone.utc)
        assert evaluated.tzinfo == default_tz

        before_recompute = datetime.now(timezone.utc)
        t.recompute_columns('inserted_at')
        assert all(before_recompute <= ts for ts in t.select(t.inserted_at).collect()['inserted_at'])

    def test_parse(self, uses_db: None) -> None:
        default_tz = ZoneInfo('America/Anchorage')
        Env.get().default_time_zone = default_tz
        from pixeltable.functions.timestamp import strptime

        t = pxt.create_table('test_tbl', {'n': pxt.Int, 'local': pxt.String | None, 'iso': pxt.String | None})
        rows = [
            {'n': 0, 'local': '01/10/2026 10:30', 'iso': '2026-10-01T10:30:00'},
            {'n': 1, 'local': 'not a date', 'iso': '2026-10-01T10:30:00+02:00'},
            {'n': 2, 'local': None, 'iso': None},
        ]
        validate_update_status(t.insert(rows), expected_rows=3)
        status = t.add_computed_column(parsed=strptime(t.local, '%d/%m/%Y %H:%M'), on_error='ignore')
        assert status.num_excs == 1
        res = t.select(t.parsed, err=t.parsed.errormsg, cast=t.iso.astype(pxt.Timestamp | None)).order_by(t.n).collect()
        assert res['parsed'] == [datetime(2026, 10, 1, 10, 30, tzinfo=default_tz), None, None]
        assert "does not match format '%d/%m/%Y %H:%M'" in res['err'][1]
        assert res['cast'] == [
            datetime(2026, 10, 1, 10, 30, tzinfo=default_tz),
            datetime(2026, 10, 1, 8, 30, tzinfo=timezone.utc),
            None,
        ]

        # a format with a time zone
        assert pxt.eval_udf(strptime, '2026-10-01 10:30 +0200', '%Y-%m-%d %H:%M %z') == datetime(
            2026, 10, 1, 8, 30, tzinfo=timezone.utc
        )
