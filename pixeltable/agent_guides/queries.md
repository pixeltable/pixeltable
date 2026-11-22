# Queries

Building and running queries, reading their results, and defining `@pxt.query` functions.

## Queries in an application

In `app.py`, a query is a `@pxt.query` function over the model classes. It can supply the value of a computed column in
another model, or serve an HTTP route added with `FastAPIRouter.add_query_route()`.

```python
import pixeltable as pxt
import pixeltable.functions as pxtf

TableModel = pxt.model_base()


class Orders(TableModel, name='orders'):
    id = pxt.Column(type=pxt.Int, primary_key=True)
    customer: pxt.String
    amount: pxt.Float


@pxt.query
def customer_total(customer: str) -> pxt.Query:
    return Orders.where(Orders.customer == customer).select(pxtf.sum(Orders.amount))


class Customers(TableModel, name='customers'):
    name = pxt.Column(type=pxt.String, primary_key=True)
    total = customer_total(name)
```

- `join()` in a `@pxt.query` over models raises `join(): a query over model ... cannot be joined`.

## Building a query

```python
pxt.create_dir('shop', if_exists='ignore')
orders = pxt.create_table(
    'shop.orders', {'id': pxt.Int, 'region': pxt.String, 'amount': pxt.Float, 'note': pxt.String | None}
)
orders.insert([{'id': 1, 'region': 'east', 'amount': 10.0}, {'id': 2, 'region': 'west', 'amount': 2.5, 'note': 'rush'}])
rows = (
    orders.where((orders.amount > 5) & (orders.note == None))
    .order_by(orders.amount, asc=False)
    .select(orders.id, doubled=orders.amount * 2)
    .limit(10)
    .collect()
)
by_region = orders.group_by(orders.region).select(orders.region, total=pxtf.sum(orders.amount)).collect()
sample = orders.select(orders.id, orders.region).sample(n=2, seed=0).collect()

regions = pxt.create_table('shop.regions', {'region': pxt.String})
regions.insert([{'region': 'east'}, {'region': 'north'}])


@pxt.query
def region_total(region: str) -> pxt.Query:
    return orders.where(orders.region == region).select(pxtf.sum(orders.amount))


totals = regions.select(regions.region, total=region_total(regions.region)).collect()
```

- Each method returns a new query; nothing runs before `collect()`, `count()`, `head()`, `tail()`, `show()` or
  `cursor()`.
- A query takes one `where()`; a second `where()` raises `where() clause already specified`.
- Combine conditions with `&`, `|` and `~`.
- A query also takes one `select()` and one `group_by()`.
- `col is None` is a Python `bool`, which `where()` rejects; write `col == None`.
- `pxt guide expressions` covers the operators.
- `order_by()` calls accumulate: later keys break ties.
- Nulls sort last in ascending order and first in descending order.
- `order_by()` takes only expressions that run in SQL. A UDF call, a Json path, or a computed column with
  `stored=False` over either raises `Invalid order_by() expression, needs to be expressible in SQL`; order by a
  stored computed column instead.
- `limit(n, offset=k)` skips the first `k` rows.
- Without `select()`, a query returns every column.
- Result column names: a keyword argument sets the name, a column keeps its name, a function call gets the
  function's name, and any other expression gets `col_0`, `col_1`, ... by its position.
- A repeated result column name gets the suffix `_1`.
- Result column names are lowercased.
- `sample()` comes after `where()`; `select()` can come before or after it.
- Calling `where()`, `order_by()`, `limit()`, `group_by()` or `join()` after `sample()` raises.
- The same `seed` returns the same rows.
- `stratify_by=` samples each group of rows with equal `stratify_by` values separately.

## Results

- `collect()` returns a `ResultSet`.
- `rows[0]` is a dict, `rows['doubled']` a list of the column's values, and `rows[0, 'doubled']` one value.
- Iterating over `rows` yields dicts.
- Values come back as Python objects: an `Image` as a `PIL.Image.Image`, a `Video`, `Audio` or `Document` as a local
  file path, a `Timestamp` as an aware `datetime` in the default time zone, `Json` as dicts and lists, an `Array` as
  a `numpy.ndarray`.
- `rows.to_pandas()` returns a DataFrame; call it on the `ResultSet`, not on the query.
- `rows.to_pydantic(Model)` yields model instances.
- `cursor()` yields rows one at a time instead of loading them all.
- `count()` counts rows, or groups after `group_by()`.
- `count()` after `limit()` raises `count() cannot be used with limit() or offset()`.
- `head(n)` and `tail(n)` follow insertion order.
- `head()` and `tail()` refuse a query with `order_by()`, `group_by()`, `sample()` or a join.

## Aggregates and joins

- Aggregates include `pxtf.count()`, `pxtf.sum()`, `pxtf.mean()`, `pxtf.min()`, `pxtf.max()` and
  `pxtf.json.make_list()`, which collects values into a list.
- Without `group_by()`, an aggregate query returns one row.
- Selecting a column that is neither in `group_by()` nor aggregated raises `Invalid non-aggregate expression in
  aggregate query`.
- In a `group_by()` query that runs partly in Python, such as one with a UDF call in `where()` or `select()`,
  `order_by()` takes only the grouping columns: `order_by() is limited to the grouping expressions`. Sort the
  collected rows instead.
- An aggregate with `group_by=` and `order_by=` arguments is a window function, with one value per row:
  `pxtf.sum(t.amount, group_by=t.region, order_by=t.id)` is a running total per region.
- Known issue (PXT-1801): an aggregate with `group_by=` and no `order_by=` returns a running value instead of the
  group's value. `pxtf.min(t.at, group_by=t.key)` gives each row the minimum over that row and the rows before it, in
  the order in which Pixeltable reads them. For the group's value, run a `group_by()` query and look up its rows by
  key.
- Known issue (PXT-1802): a window function in `where()` selects the wrong rows. Select the window function instead,
  and filter the results in Python.
- `distinct()` removes duplicate rows.
- Defining a computed column with an aggregate raises `aggregate functions are not supported in computed columns`.
- `t.join(other, on=t.region == other.region, how='left')` joins two tables.
- `on=` also takes a column, matched by name in both tables.
- `how='cross'` takes no `on=`; every other join type requires it.
- A join without `select()` returns the columns of both tables, with a suffix on a repeated name (`region_1`).
- `join()` takes a table, a view, a snapshot or a bound model, not a query: the result of a `group_by()` query
  cannot be joined to its table. Collect the `group_by()` result and look up its rows by key in Python.

## `@pxt.query` functions

- `@pxt.query` calls the function once, at decoration, with a placeholder expression for each parameter; the
  function returns the query.
- Python control flow on a parameter, such as `if region == 'east':`, raises.
- Each parameter needs a type annotation or a type in `@pxt.query(param_types=[...])`; otherwise `@pxt.query`
  raises `Cannot infer pixeltable type for parameter ...`.
- A parameter can appear in `where()`, in `limit(n)` and in `similarity()` (`pxt guide indexes`).
- Calling the function returns an expression, not rows.
- A call in `where()`, `select()` or a computed column runs a separate query for each row.
- To compute one value per group, such as the first signup per email address, run one `group_by()` query with an
  aggregate and match its rows to the table's rows by key.
- `pxt.eval_udf(region_total, 'east')` runs the query on Python values and returns its result.
- A call returns a value, a dict, a list of values or a list of dicts, depending on the query:

| The query | Selects one unnamed expression, as in `select(t.id)` | Any other select list |
|---|---|---|
| Aggregates without `group_by()`, or ends in a literal `limit(1)` | the value, or `None` | the row as a dict, or `None` |
| Any other query | a list of values | a list of dicts |

- A call to a query whose `limit()` takes a parameter returns a list, even when the argument is 1.
- A call to a query with `select(id=t.id)` returns dicts.
- A computed column that calls a `@pxt.query` runs it when its row is inserted or when an update changes the
  arguments of the call.
- The computed column keeps its value when the queried table changes later.
- A `select()` that calls a `@pxt.query` sees the table's current rows.
- A computed column whose query reads its own table sees the rows of earlier inserts, not the other rows of the same
  insert.
- Known issue (PXT-1736): in a script, a `@pxt.query` declared over models fails in `select()` and `where()`, with
  `Table was dropped` or `declared over a model rather than a table`. It works in a model's computed column, in a
  route and in `pxt.eval_udf()`. In a script, declare the query over the `Table` objects from `pxt.get_table()`.
- `add_computed_column()` refuses a `@pxt.query` declared over models: `can only be used in a TableModel or a query
  route`.
- A computed column stores the query itself, so it can call a `@pxt.query` defined in a notebook.
- After a change to the function's body, `pxt schema diff` reports the column's expression as changed, and existing
  rows keep their values.

```bash
pxt rows my_app/orders -n 5 --cols id,amount
pxt count my_app/orders
```
