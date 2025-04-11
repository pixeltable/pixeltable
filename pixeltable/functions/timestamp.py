"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs) for `TimestampType`.

Usage example:
```python
import pixeltable as pxt

t = pxt.get_table(...)
t.select(t.timestamp_col.year, t.timestamp_col.weekday()).collect()
```
"""

from datetime import datetime
from typing import Optional

import sqlalchemy as sql

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
        tz: The time zone to convert to. Must be a valid time zone name from the IANA Time Zone Database.
    """
    from zoneinfo import ZoneInfo

    tzinfo = ZoneInfo(tz)
    return self.astimezone(tzinfo)


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
        sql.cast(year, sql.Integer),
        sql.cast(month, sql.Integer),
        sql.cast(day, sql.Integer),
        sql.cast(hour, sql.Integer),
        sql.cast(minute, sql.Integer),
        sql.cast(second + microsecond / 1000000.0, sql.Float),
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
    year: Optional[int] = None,
    month: Optional[int] = None,
    day: Optional[int] = None,
    hour: Optional[int] = None,
    minute: Optional[int] = None,
    second: Optional[int] = None,
    microsecond: Optional[int] = None,
) -> datetime:
    """
    Return a datetime with the same attributes, except for those attributes given new values by whichever keyword
    arguments are specified.

    Equivalent to [`datetime.replace()`](https://docs.python.org/3/library/datetime.html#datetime.datetime.replace).
    """
    kwargs = {k: v for k, v in locals().items() if k != 'self' and v is not None}
    return self.replace(**kwargs)


@pxt.udf(is_method=True)
def toordinal(self: datetime) -> int:
    """
    Return the proleptic Gregorian ordinal of the date, where January 1 of year 1 has ordinal 1.

    Equivalent to [`datetime.toordinal()`](https://docs.python.org/3/library/datetime.html#datetime.datetime.toordinal).
    """
    return self.toordinal()


@pxt.udf(is_method=True)
def posix_timestamp(self: datetime) -> float:
    """
    Return POSIX timestamp corresponding to the datetime instance.

    Equivalent to [`datetime.timestamp()`](https://docs.python.org/3/library/datetime.html#datetime.datetime.timestamp).
    """
    return self.timestamp()


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
