# Expressions

Operators, casts, null handling and time functions on column values, and how to find and try the built-in functions.

## Finding functions

- `pxt functions` lists the functions, aggregates and iterators, one per line, each with its call syntax and the
  first paragraph of its docstring.
- `pxt functions` runs without a daemon or catalog.
- `pxt functions video` lists the functions in `pixeltable.functions.video`.
- `pxt functions app` lists the UDFs of `app.py`.
- `--on Timestamp` lists the methods and properties of a type.
- `--search crop` matches names and docstrings.
- `--kind` restricts the list to one kind: `udf`, `aggregate`, `iterator` or `query`.
- `--json` prints one JSON record per function, with the full docstring.
- `pxt.list_functions()` takes the same filters (`module`, `on=`, `search=`, `kind=`) and returns a `FunctionList` of
  dicts with `path`, `name`, `kind`, `call`, `signatures` and `summary`.
- A property, such as `Timestamp.year` or `Image.width`, takes no parentheses: `t.departs_at.year`.

## Operators

```python
import pixeltable as pxt
import pixeltable.functions as pxtf

pxt.create_dir('exprs', if_exists='ignore')
trips = pxt.create_table(
    'exprs.trips',
    {'id': pxt.Int, 'driver': pxt.String | None, 'delay': pxt.Float, 'departs': pxt.String, 'note': pxt.String | None},
)
trips.insert(
    [
        {'id': 1, 'driver': 'Ana', 'delay': 7.5, 'departs': '03/04/2026 10:30'},
        {'id': 2, 'driver': None, 'delay': 1.0, 'departs': '04/04/2026 08:00', 'note': 'rerouted'},
    ]
)
late = trips.where((trips.delay > 5.0) & (trips.driver != None)).select(trips.id, label='Driver ' + trips.driver)
num_late = trips.select(pxtf.sum((trips.delay > 5.0).astype(pxt.Int))).collect()
parsed = trips.select(
    departs_at=pxtf.timestamp.strptime(trips.departs, '%d/%m/%Y %H:%M'),
    driver=pxtf.coalesce(trips.driver, trips.note, 'unknown'),
    checked_at=pxtf.timestamp.now(),
).collect()
```

- Numeric expressions support arithmetic, and String expressions support `+`.
- Conditions combine with `&`, `|` and `~`.
- `x == None` and `x != None` test for null.
- `x.isin(['a', 'b'])` tests membership.
- A UUID compares with the string form of a UUID, as in `t.id == str(doc_id)`; a string that is not a UUID raises
  `not a valid UUID`.
- Python `and`, `or` and `not` on a column raise `TypeError`.
- `x is None` is a Python `bool`, not a condition.
- An f-string or `str.format()` on a column raises `An f-string or str.format() on a Pixeltable expression produces
  one constant string, not a value per row`.
- `str(col)` and `'%s' % col` return one constant string for every row, without an error.
- `pxt schema check` warns when a computed column in a model references no column.
- `'Driver ' + driver` and `pxtf.string.format('Driver {0}', driver)` build a string per row.

## Casts

`astype()` sets an expression's type. These casts also convert the value:

| From | `astype()` to | Result |
|---|---|---|
| Bool | `pxt.Int`, `pxt.Float` | 0 for `False`, 1 for `True`, so `pxtf.sum(cond.astype(pxt.Int))` counts matching rows |
| Int | `pxt.Float` | the number as a float |
| ISO 8601 String | `pxt.Timestamp`, `pxt.Date` | the parsed value; a timestamp without a time zone is in the default time zone |
| path or URL String | `pxt.Image` | the image file |

- `pxtf.timestamp.strptime(s, '%d/%m/%Y %H:%M')` and `pxtf.date.strptime(s, '%d/%m/%Y')` parse other formats.
- `strptime()` fails the row of a string that does not match the format.
- `astype(pxt.Int)` on a Float raises `Cannot convert ... from Float to Int with astype()`.
- `to_int()` converts a Float to an Int and truncates toward zero; `round().to_int()`, `floor().to_int()` and
  `ceil().to_int()` round first.
- `astype(pxt.String)` on a UUID fails each row with `TypeError`.
- `pxtf.uuid.to_string(col)` converts a UUID to a String.
- A cast of a Json value checks the value when the expression runs (`pxt guide json`).

## Nulls

- `pxtf.coalesce(a, b, ...)` returns its first non-null argument.
- The result type of `pxtf.coalesce()` is the common supertype of the arguments' types (`Float` for `Int` and
  `Float`, `Json` for `String` and `Int`).
- The result of `pxtf.coalesce()` is nullable only if every argument is nullable.
- To branch on a value, write a UDF (`pxt guide udfs`); Pixeltable has no conditional expression.
- `pxt guide udfs` also covers how a UDF handles a null argument.

## Time

- The default time zone is `time_zone` in the `[pixeltable]` section of the configuration, or
  `PIXELTABLE_TIME_ZONE`; without either, it is the system's time zone.
- `pxtf.timestamp.now()` returns the current time in the default time zone.
- `pxtf.date.today()` returns the current date in the default time zone, or in the time zone of its argument, as in
  `pxtf.date.today('Asia/Tokyo')`.
- A query, or a computed column with `stored=False`, evaluates `pxtf.timestamp.now()` and `pxtf.date.today()` each
  time it runs.
- A stored computed column records the time of the insert, and a recompute replaces it with the time of the
  recompute.
- `ts + timedelta(hours=1)` and `ts - timedelta(days=7)` shift a Timestamp, as in
  `t.where(t.created_at > pxtf.timestamp.now() - timedelta(days=7))`.
- A timedelta added to or subtracted from a Date must be a whole number of days.
- `date.add_days(n)` adds `n` days to a Date.
- `ts.add_seconds(n)` adds `n` seconds to a Timestamp; `n` can be a column.
- `end.seconds_since(start)` returns the seconds from `start` to `end`; `end - start` raises `- is not defined for
  two Timestamps`.
- A timedelta, `add_seconds()` and `seconds_since()` measure elapsed time: across a daylight saving time change,
  `ts + timedelta(days=1)` moves the wall-clock time by 23 or 25 hours.

## Media values

- `col.fileurl` is the URL of a media cell's file.
- `col.localpath` is a local path to a media cell's file.
- `col.errortype` and `col.errormsg` record the error of a failed cell in a stored computed column or a media column
  (`pxt guide tables`).

## String methods

- `rpartition()` returns two empty strings followed by the original string when the separator is absent.

## Trying an expression

- `select()` evaluates an expression on existing rows and stores nothing.
- `limit()` keeps the number of model calls small.
- `pxt.eval_udf(pxtf.string.upper, 'text')` calls a UDF, a built-in function or a `@pxt.query` function on Python
  values, without a table.
- A media argument of `pxt.eval_udf()` can be a file path, a URL or a PIL image.
- `eval_udf()` refuses an aggregate (`sum() is an aggregate function, which needs rows; use it in a query`) and a
  column as an argument (`the arguments must be Python values, not columns`).

```bash
pxt functions --on Timestamp
pxt functions --search coalesce
pxt functions video --kind iterator
pxt functions app --kind udf --json
```
