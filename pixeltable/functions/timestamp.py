"""
Pixeltable UDFs for `TimestampType`.

Usage example:
```python
import pixeltable as pxt

t = pxt.get_table(...)
t.select(t.timestamp_col.year, t.timestamp_col.weekday()).collect()
```
"""

import re as _re
from datetime import datetime

import sqlalchemy as sql
from sqlalchemy.sql.elements import BindParameter as _BindParameter

import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.utils.code import local_public_names

_SQL_ZERO = sql.literal(0)


@pxt.udf(is_property=True)
def year(self: datetime) -> int:
    """
    Between [`MINYEAR`](https://docs.python.org/3/library/datetime.html#datetime.MINYEAR) and
    [`MAXYEAR`](https://docs.python.org/3/library/datetime.html#datetime.MAXYEAR) inclusive.

    Equivalent to [`datetime.year`](https://docs.python.org/3/library/datetime.html#datetime.datetime.year).
    """
    return self.year


@year.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.extract('year', self)


@pxt.udf(is_property=True)
def month(self: datetime) -> int:
    """
    Between 1 and 12 inclusive.

    Equivalent to [`datetime.month`](https://docs.python.org/3/library/datetime.html#datetime.datetime.month).
    """
    return self.month


@month.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.extract('month', self)


@pxt.udf(is_property=True)
def day(self: datetime) -> int:
    """
    Between 1 and the number of days in the given month of the given year.

    Equivalent to [`datetime.day`](https://docs.python.org/3/library/datetime.html#datetime.datetime.day).
    """
    return self.day


@day.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.extract('day', self)


@pxt.udf(is_property=True)
def hour(self: datetime) -> int:
    """
    Between 0 and 23 inclusive.

    Equivalent to [`datetime.hour`](https://docs.python.org/3/library/datetime.html#datetime.datetime.hour).
    """
    return self.hour


@hour.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.extract('hour', self)


@pxt.udf(is_property=True)
def minute(self: datetime) -> int:
    """
    Between 0 and 59 inclusive.

    Equivalent to [`datetime.minute`](https://docs.python.org/3/library/datetime.html#datetime.datetime.minute).
    """
    return self.minute


@minute.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.extract('minute', self)


@pxt.udf(is_property=True)
def second(self: datetime) -> int:
    """
    Between 0 and 59 inclusive.

    Equivalent to [`datetime.second`](https://docs.python.org/3/library/datetime.html#datetime.datetime.second).
    """
    return self.second


@second.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.extract('second', self)


@pxt.udf(is_property=True)
def microsecond(self: datetime) -> int:
    """
    Between 0 and 999999 inclusive.

    Equivalent to [`datetime.microsecond`](https://docs.python.org/3/library/datetime.html#datetime.datetime.microsecond).
    """
    return self.microsecond


@microsecond.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.extract('microseconds', self) - sql.extract('second', self) * 1000000


@pxt.udf(is_method=True)
def astimezone(self: datetime, tz: str) -> datetime:
    """
    Convert the datetime to the given time zone.

    Args:
        tz: The time zone to convert to. Must be a valid time zone name from the
            [IANA Time Zone Database](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones).
    """
    from zoneinfo import ZoneInfo

    tzinfo = ZoneInfo(tz)
    return self.astimezone(tzinfo)


# Note: astimezone cannot be implemented in SQL because PostgreSQL's timestamptz
# stores UTC instants, and the target timezone info cannot be preserved in the result.
# psycopg interprets timestamptz using session timezone, not the query's target timezone.
# Python's astimezone() returns datetime with target tzinfo, which SQL cannot replicate.


@pxt.udf(is_method=True)
def weekday(self: datetime) -> int:
    """
    Between 0 (Monday) and 6 (Sunday) inclusive.

    Equivalent to [`datetime.weekday()`](https://docs.python.org/3/library/datetime.html#datetime.datetime.weekday).
    """
    return self.weekday()


@weekday.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.extract('isodow', self) - 1


@pxt.udf(is_method=True)
def isoweekday(self: datetime) -> int:
    """
    Return the day of the week as an integer, where Monday is 1 and Sunday is 7.

    Equivalent to [`datetime.isoweekday()`](https://docs.python.org/3/library/datetime.html#datetime.datetime.isoweekday).
    """
    return self.isoweekday()


@isoweekday.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.extract('isodow', self)


@pxt.udf(is_method=True)
def isocalendar(self: datetime) -> dict:
    """
    Return a dictionary with three entries: `'year'`, `'week'`, and `'weekday'`.

    Equivalent to
    [`datetime.isocalendar()`](https://docs.python.org/3/library/datetime.html#datetime.datetime.isocalendar).
    """
    iso_year, iso_week, iso_weekday = self.isocalendar()
    return {'year': iso_year, 'week': iso_week, 'weekday': iso_weekday}


@isocalendar.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    # Build JSON object with ISO calendar components
    return sql.func.jsonb_build_object(
        'year',
        sql.extract('isoyear', self).cast(sql.Integer),
        'week',
        sql.extract('week', self).cast(sql.Integer),
        'weekday',
        sql.extract('isodow', self).cast(sql.Integer),
    )


@pxt.udf(is_method=True)
def isoformat(self: datetime, sep: str = 'T', timespec: str = 'auto') -> str:
    """
    Return a string representing the date and time in ISO 8601 format.

    Equivalent to [`datetime.isoformat()`](https://docs.python.org/3/library/datetime.html#datetime.datetime.isoformat).

    Args:
        sep: Separator between date and time.
        timespec: The number of additional terms in the output. See the
            [`datetime.isoformat()`](https://docs.python.org/3/library/datetime.html#datetime.datetime.isoformat)
            documentation for more details.
    """
    return self.isoformat(sep=sep, timespec=timespec)


@isoformat.to_sql
def _(
    self: sql.ColumnElement, sep: sql.ColumnElement | None = None, timespec: sql.ColumnElement | None = None
) -> sql.ColumnElement:
    if timespec is not None:
        return None  # Can't dynamically select format for arbitrary timespec in SQL
    # Default timespec='auto': include microseconds only when non-zero
    separator = sep if sep is not None else sql.literal('T')
    date_part = sql.func.to_char(self, 'YYYY-MM-DD')
    # extract('microseconds') returns total seconds*1e6; modulo 1e6 gives sub-second microseconds
    sub_us = sql.extract('microseconds', self).cast(sql.Integer) % 1000000
    time_part = sql.case(
        (sub_us == 0, sql.func.to_char(self, 'HH24:MI:SS')), else_=sql.func.to_char(self, 'HH24:MI:SS.US')
    )
    return sql.func.concat(date_part, separator, time_part, sql.func.to_char(self, 'TZH:TZM'))


# Mapping from Python strftime codes to PostgreSQL to_char patterns.
_PY_TO_PG: dict[str, str] = {
    '%Y': 'YYYY',  # 4-digit year
    '%y': 'YY',  # 2-digit year
    '%m': 'MM',  # month (01-12)
    '%d': 'DD',  # day of month (01-31)
    '%H': 'HH24',  # hour 24h (00-23)
    '%I': 'HH12',  # hour 12h (01-12)
    '%M': 'MI',  # minute (00-59)
    '%S': 'SS',  # second (00-59)
    '%f': 'US',  # microseconds (000000-999999)
    '%p': 'AM',  # AM/PM indicator
    '%j': 'DDD',  # day of year (001-366)
    '%A': 'TMDay',  # full weekday name, unpadded (e.g. Monday)
    '%a': 'Dy',  # abbreviated weekday name (Mon, Tue, ...)
    '%B': 'TMMonth',  # full month name, unpadded (e.g. January)
    '%b': 'Mon',  # abbreviated month name (Jan, Feb, ...)
    '%h': 'Mon',  # same as %b
    '%%': '%',  # literal percent sign
}

# Python strftime codes with no reliable SQL equivalent; fall back to Python for these.
_UNSUPPORTED_PY_CODES = frozenset({'%c', '%x', '%X', '%G', '%V', '%u', '%w', '%W', '%U', '%z', '%Z'})

_PY_FORMAT_RE = _re.compile(r'%.')


def _translate_strftime_format(fmt: str) -> str | None:
    """Translate a Python strftime format string to a PostgreSQL to_char format string.

    Returns None if the format contains codes that can't be reliably translated.
    Literal text segments are wrapped in double-quotes to prevent PostgreSQL from
    misinterpreting alphabetic characters as format patterns.
    """
    result: list[str] = []
    last = 0
    for m in _PY_FORMAT_RE.finditer(fmt):
        lit = fmt[last : m.start()]
        if lit:
            result.append('"' + lit.replace('"', '\\"') + '"')
        code = m.group()
        if code in _UNSUPPORTED_PY_CODES:
            return None
        pg = _PY_TO_PG.get(code)
        if pg is None:
            return None
        result.append(pg)
        last = m.end()
    lit = fmt[last:]
    if lit:
        result.append('"' + lit.replace('"', '\\"') + '"')
    return ''.join(result)


@pxt.udf(is_method=True)
def strftime(self: datetime, format: str) -> str:
    """
    Return a string representing the date and time, controlled by an explicit format string.

    Equivalent to [`datetime.strftime()`](https://docs.python.org/3/library/datetime.html#datetime.datetime.strftime).

    Args:
        format: The format string to control the output. For a complete list of formatting directives, see
            [`strftime()` and `strptime()` Behavior](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior).
    """
    return self.strftime(format)


@strftime.to_sql
def _(self: sql.ColumnElement, format: sql.ColumnElement) -> sql.ColumnElement:
    if not isinstance(format, _BindParameter):
        return None  # Can only translate literal format strings at query-planning time
    pg_fmt = _translate_strftime_format(format.value)
    if pg_fmt is None:
        return None
    return sql.func.to_char(self, pg_fmt)


@pxt.udf(is_method=True)
def make_timestamp(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0, microsecond: int = 0
) -> datetime:
    """
    Create a timestamp.

    Equivalent to [`datetime()`](https://docs.python.org/3/library/datetime.html#datetime.datetime).
    """
    return datetime(year, month, day, hour, minute, second, microsecond, tzinfo=Env.get().default_time_zone)


@make_timestamp.to_sql
def _(
    year: sql.ColumnElement,
    month: sql.ColumnElement,
    day: sql.ColumnElement,
    hour: sql.ColumnElement = _SQL_ZERO,
    minute: sql.ColumnElement = _SQL_ZERO,
    second: sql.ColumnElement = _SQL_ZERO,
    microsecond: sql.ColumnElement = _SQL_ZERO,
) -> sql.ColumnElement:
    return sql.func.make_timestamptz(
        year.cast(sql.Integer),
        month.cast(sql.Integer),
        day.cast(sql.Integer),
        hour.cast(sql.Integer),
        minute.cast(sql.Integer),
        (second + microsecond / 1000000.0).cast(sql.Float),
    )


# @pxt.udf
# def date(self: datetime) -> datetime:
#     """
#     Return the date part of the datetime.
#
#     Equivalent to [`datetime.date()`](https://docs.python.org/3/library/datetime.html#datetime.datetime.date).
#     """
#     d = self.date()
#     return datetime(d.year, d.month, d.day)
#
#
# @pxt.udf
# def time(self: datetime) -> datetime:
#     """
#     Return the time part of the datetime, with microseconds set to 0.
#
#     Equivalent to [`datetime.time()`](https://docs.python.org/3/library/datetime.html#datetime.datetime.time).
#     """
#     t = self.time()
#     return datetime(1, 1, 1, t.hour, t.minute, t.second, t.microsecond)


@pxt.udf(is_method=True)
def replace(
    self: datetime,
    year: int | None = None,
    month: int | None = None,
    day: int | None = None,
    hour: int | None = None,
    minute: int | None = None,
    second: int | None = None,
    microsecond: int | None = None,
) -> datetime:
    """
    Return a datetime with the same attributes, except for those attributes given new values by whichever keyword
    arguments are specified.

    Equivalent to [`datetime.replace()`](https://docs.python.org/3/library/datetime.html#datetime.datetime.replace).
    """
    kwargs = {k: v for k, v in locals().items() if k != 'self' and v is not None}
    return self.replace(**kwargs)


@replace.to_sql
def _(
    self: sql.ColumnElement,
    year: sql.ColumnElement | None = None,
    month: sql.ColumnElement | None = None,
    day: sql.ColumnElement | None = None,
    hour: sql.ColumnElement | None = None,
    minute: sql.ColumnElement | None = None,
    second: sql.ColumnElement | None = None,
    microsecond: sql.ColumnElement | None = None,
) -> sql.ColumnElement:
    # Use coalesce to use original value when replacement is not specified
    new_year = year.cast(sql.Integer) if year is not None else sql.extract('year', self).cast(sql.Integer)
    new_month = month.cast(sql.Integer) if month is not None else sql.extract('month', self).cast(sql.Integer)
    new_day = day.cast(sql.Integer) if day is not None else sql.extract('day', self).cast(sql.Integer)
    new_hour = hour.cast(sql.Integer) if hour is not None else sql.extract('hour', self).cast(sql.Integer)
    new_minute = minute.cast(sql.Integer) if minute is not None else sql.extract('minute', self).cast(sql.Integer)
    # For seconds, we need to combine second and microsecond
    orig_second = sql.extract('second', self)
    orig_microsecond = sql.extract('microseconds', self) - sql.extract('second', self) * 1000000
    new_second = second.cast(sql.Integer) if second is not None else sql.func.floor(orig_second).cast(sql.Integer)
    new_microsecond = microsecond.cast(sql.Integer) if microsecond is not None else orig_microsecond.cast(sql.Integer)
    # Combine second and microsecond for make_timestamptz
    combined_seconds = (new_second + new_microsecond / 1000000.0).cast(sql.Float)
    return sql.func.make_timestamptz(new_year, new_month, new_day, new_hour, new_minute, combined_seconds)


@pxt.udf(is_method=True)
def toordinal(self: datetime) -> int:
    """
    Return the proleptic Gregorian ordinal of the date, where January 1 of year 1 has ordinal 1.

    Equivalent to [`datetime.toordinal()`](https://docs.python.org/3/library/datetime.html#datetime.datetime.toordinal).
    """
    return self.toordinal()


@toordinal.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    # Ordinal is days since Jan 1, year 1 (which has ordinal 1)
    # Cast timestamp to date first, then calculate ordinal
    epoch = sql.cast(sql.literal('0001-01-01'), sql.Date)
    return (self.cast(sql.Date) - epoch + 1).cast(sql.Integer)


@pxt.udf(is_method=True)
def posix_timestamp(self: datetime) -> float:
    """
    Return POSIX timestamp corresponding to the datetime instance.

    Equivalent to [`datetime.timestamp()`](https://docs.python.org/3/library/datetime.html#datetime.datetime.timestamp).
    """
    return self.timestamp()


@posix_timestamp.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.extract('epoch', self)


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
