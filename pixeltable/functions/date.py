"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs) for `DateType`.

Usage example:
```python
import pixeltable as pxt

t = pxt.get_table(...)
t.select(t.date_col.year, t.date_col.weekday()).collect()
```
"""

from datetime import date, timedelta

import sqlalchemy as sql

import pixeltable as pxt
from pixeltable.utils.code import local_public_names

_SQL_ZERO = sql.literal(0)

# NOT YET SUPPORTED date +/- integer
# NOT YET SUPPORTED date1 - date2 -> integer
# NOT YET SUPPORTED timestamp(date)
# NOT YET SUPPORTED date(timestamp)


@pxt.udf(is_property=True)
def year(self: date) -> int:
    """
    Between [`MINYEAR`](https://docs.python.org/3/library/datetime.html#datetime.MINYEAR) and
    [`MAXYEAR`](https://docs.python.org/3/library/datetime.html#datetime.MAXYEAR) inclusive.

    Equivalent to [`date.year`](https://docs.python.org/3/library/datetime.html#datetime.date.year).
    """
    return self.year


@year.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.extract('year', self)


@pxt.udf(is_property=True)
def month(self: date) -> int:
    """
    Between 1 and 12 inclusive.

    Equivalent to [`date.month`](https://docs.python.org/3/library/datetime.html#datetime.date.month).
    """
    return self.month


@month.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.extract('month', self)


@pxt.udf(is_property=True)
def day(self: date) -> int:
    """
    Between 1 and the number of days in the given month of the given year.

    Equivalent to [`date.day`](https://docs.python.org/3/library/datetime.html#datetime.date.day).
    """
    return self.day


@day.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.extract('day', self)


@pxt.udf(is_method=True)
def make_date(year: int, month: int, day: int) -> date:
    """
    Create a date.

    Equivalent to [`datetime()`](https://docs.python.org/3/library/datetime.html#datetime.date).
    """
    return date(year, month, day)


@make_date.to_sql
def _(year: sql.ColumnElement, month: sql.ColumnElement, day: sql.ColumnElement) -> sql.ColumnElement:
    return sql.func.make_date(sql.cast(year, sql.Integer), sql.cast(month, sql.Integer), sql.cast(day, sql.Integer))


@pxt.udf(is_method=True)
def weekday(self: date) -> int:
    """
    Between 0 (Monday) and 6 (Sunday) inclusive.

    Equivalent to [`date.weekday()`](https://docs.python.org/3/library/datetime.html#datetime.date.weekday).
    """
    return self.weekday()


@weekday.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.extract('isodow', self) - 1


@pxt.udf(is_method=True)
def isoweekday(self: date) -> int:
    """
    Return the day of the week as an integer, where Monday is 1 and Sunday is 7.

    Equivalent to [`date.isoweekday()`](https://docs.python.org/3/library/datetime.html#datetime.date.isoweekday).
    """
    return self.isoweekday()


@isoweekday.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.extract('isodow', self)


@pxt.udf(is_method=True)
def isocalendar(self: date) -> dict:
    """
    Return a dictionary with three entries: `'year'`, `'week'`, and `'weekday'`.

    Equivalent to
    [`date.isocalendar()`](https://docs.python.org/3/library/datetime.html#datetime.date.isocalendar).
    """
    iso_year, iso_week, iso_weekday = self.isocalendar()
    return {'year': iso_year, 'week': iso_week, 'weekday': iso_weekday}


@pxt.udf(is_method=True)
def isoformat(self: date, sep: str = 'T', timespec: str = 'auto') -> str:
    """
    Return a string representing the date and time in ISO 8601 format.

    Equivalent to [`date.isoformat()`](https://docs.python.org/3/library/datetime.html#datetime.date.isoformat).

    Args:
        sep: Separator between date and time.
        timespec: The number of additional terms in the output. See the
            [`date.isoformat()`](https://docs.python.org/3/library/datetime.html#datetime.date.isoformat)
            documentation for more details.
    """
    return self.isoformat()


@pxt.udf(is_method=True)
def toordinal(self: date) -> int:
    """
    Return the proleptic Gregorian ordinal of the date, where January 1 of year 1 has ordinal 1.

    Equivalent to [`date.toordinal()`](https://docs.python.org/3/library/datetime.html#datetime.date.toordinal).
    """
    return self.toordinal()


@pxt.udf(is_method=True)
def strftime(self: date, format: str) -> str:
    """
    Return a string representing the date and time, controlled by an explicit format string.

    Equivalent to [`date.strftime()`](https://docs.python.org/3/library/datetime.html#datetime.date.strftime).

    Args:
        format: The format string to control the output. For a complete list of formatting directives, see
            [`strftime()` and `strptime()` Behavior](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior).
    """
    return self.strftime(format)


@pxt.udf(is_method=True)
def add_days(self: date, n: int) -> date:
    """
    Add `n` days to the date.

    Equivalent to [`date + timedelta(days=n)`](https://docs.python.org/3/library/datetime.html#datetime.timedelta).
    """
    return self + timedelta(days=n)


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
