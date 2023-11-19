# API Overview

Import conventions:
```python
import pixeltable as pxt
import pixeltable.functions as pxtf
```

## Creating a client
```python
cl = pxt.Client()
```

## Client operations summary

### Tables

| Task                   | Code                                                                                                  |
|--------------------------|-------------------------------------------------------------------------------------------------------|
| Create a table           | t = cl.[create_table](pixeltable.Client.create_table)('table_name', [pxt.Column('name', <type>), ...]) |
| Use an existing table    | t = cl.[get_table](pixeltable.Client.get_table)('video_data')                                         |
| Rename a table           | cl.[move](pixeltable.Client.move)('video_data', 'vd')                                 |
| List tables              | cl.[list_tables](pixeltable.Client.list_tables)()                                                     |
| Delete a table           | cl.[drop_table](pixeltable.Client.drop_table)('video_data')                                           |

**Create a table**
```python
t = cl.create_table('table_name', [pt.Column('name', <type>), ...]) 
```
See also: [create_table](pixeltable.Client.create_table)

**Use an existing table**
```python
t = cl.get_table('video_data')                                         |
```
See also [get_table](pixeltable.Client.get_table)

### Views
| Task                   | Code                                                                                                  |
|--------------------------|-------------------------------------------------------------------------------------------------------|
| Create a view           | t = cl.[create_view](pixeltable.Client.create_view)('view_name', base_tbl, schema=[pt.Column('name', <type>), ...], filter=base_tbl.col > 10) |
| Use an existing view    | t = cl.[get_table](pixeltable.Client.get_table)('view_name')                                         |
| Rename a view           | cl.[move](pixeltable.Client.move)('view_name', 'v1')                                 |

### Directories
| Task                   | Code                                                                                                  |
|--------------------------|-------------------------------------------------------------------------------------------------------|
| Create a table           | t = cl.[create_table](pixeltable.Client.create_table)('table_name', [pt.Column('name', <type>), ...]) |
| Use an existing table    | t = cl.[get_table](pixeltable.Client.get_table)('video_data')                                         |
| Rename a table           | cl.[rename_table](pixeltable.Client.rename_table)('video_data', 'vd')                                 |
| List tables              | cl.[list_tables](pixeltable.Client.list_tables)()                                                     |
| Delete a table           | cl.[drop_table](pixeltable.Client.drop_table)('video_data')                                           |
| Create a stored function | cl.[create_function](pixeltable.Client.create_function)('func_name', ...)                             |
| Load a stored function   | f = cl.[get_function](pixeltable.Client.get_function)('func_name')                                    |
| Rename a stored function | cl.[rename_function](pixeltable.Client.rename_function)('func_name', 'better_name')                   |
| Update a stored function | cl.[update_function](pixeltable.Client.update_function)('func_name', ...)                             |
| Delete a stored function | cl.[drop_function](pixeltable.Client.drop_function)('func_name')                                      |

### Functions
| Task                   | Code                                                                                                  |
|--------------------------|-------------------------------------------------------------------------------------------------------|
| Create a stored function | cl.[create_function](pixeltable.Client.create_function)('func_name', ...)                             |
| Load a stored function   | f = cl.[get_function](pixeltable.Client.get_function)('func_name')                                    |
| Rename a stored function | cl.[rename_function](pixeltable.Client.rename_function)('func_name', 'better_name')                   |
| Update a stored function | cl.[update_function](pixeltable.Client.update_function)('func_name', ...)                             |
| Delete a stored function | cl.[drop_function](pixeltable.Client.drop_function)('func_name')                                      |

## Frame extraction for video data
Creating a table with video data and automatic frame extraction:
```python
c1 = pt.Column('video', pt.VideoType())
c2 = pt.Column('frame_idx', pt.IntType())
c3 = pt.Column('frame', pt.ImageType())
t = db.create_table(
    'video_table', [c1, c2, c3],
    extract_frames_from='video',
    extracted_frame_col='frame',
    extracted_frame_idx_col='frame_idx',
    extracted_fps=1)
```

`extracted_fps=0` extracts frames at the original frame rate.

## Pixeltable types
|Pixeltable type|Python type|
|----|----|
| `pt.StringType()`| `str` |
| `pt.IntType()`| `int` |
| `pt.FloatType()`| `float` |
| `pt.BoolType()`| `bool` |
| `pt.TimestampType()`| `datetime.datetime` |
| `pt.JsonType()`| lists and dicts that can be converted to JSON|
| `pt.ArrayType()`| `numpy.ndarray`|
| `pt.ImageType()`| `PIL.Image.Image`|
| `pt.VideoType()`| `str` (the file path)|


## Table operations summary
|Action|Code|
|----|----|
| Print table schema | `t.describe()`|
| Query a table | `t[<column selections>][<filter expression>].show()` |
| Insert rows into a table| `t.insert_rows([<list of column values for row 1>, <row 2>, ...], columns=<list of column names>)`|
| Add a column| `t.add_column(pt.Column(...))`|
| Rename a column| `t.rename_column('col_name', 'new_col_name')`|
| Drop a column| `t.drop_column('col_name')`|
| Undo the last update operation (add/rename/drop column or insert)| `t.revert()`|

## Querying a table

|Action|Code|
|----|----|
| Look at 10 rows | `t.show(10)` |
| Look at all rows | `t.show(0)` |
| Look at row for frame 15 | `t[t.frame_idx == 15].show()` |
| Look at rows before index 15 | `t[t.frame_idx < 15].show(0)` |
| Look at rows before index 15 with RGB frames | `t[(t.frame_idx < 15) & (t.frame.mode == 'RGB')].show(0)` |

Pixeltable supports the standard comparison operators (`>=`, `>`, `==`, `<=`, `<`).
`== None` is the equivalent of `isna()/isnull()` in Pandas.

Boolean operators are the same as in Pandas: `&` for `and`, `|` for `or`, `~` for `not`.
They also require parentheses, for example: `(t.frame_idx < 15) & (t.frame.mode == 'RGB')`
or `~(t.frame.mode == 'RGB')`.

## Selecting and transforming columns

|Action|Code|
|----|----|
| Only retrieve the frame index and frame | `t[t.frame_idx, t.frame].show()` |
| Look at frames rotated 90 degrees | `t[t.frame.rotate(90)].show()` |
| Overlay frame with itself rotated 90 degrees | `t[pt.functions.blend(t.frame, t.frame.rotate(90))].show()` |

## Computed columns

The values in a computed column are automatically filled when data is added:
```python
t.add_column(pt.Column('c_added', computed_with=(t.frame_idx + 1)))
```

Computed columns have attributes `errortype` and `errormsg`, which contain the exception type and string
in rows where the `computed_with` expression results in an exception (the column value itself will be `None`).

Example:
```python
t[t.c_added.errortype != None][t.c_added.errortype, t.c_added.errormsg].show()
```
returns the exception type and message for rows with an exception.

## Inserting data into a table
```python
t.insert_rows([['/path/to/video1.mp4'], ['/path/to/video2.mp4']], columns=['video'])
```
Each row is a list of column values (do not provide values for computed columns). The
`columns` argument contains the names of columns for which values are being provided.

## Attributes and methods on image data

Images are currently represented as `PIL.Image.Image` instances in memory and support a lot of the
attributes and methods documented
[here](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image).

Available attributes are: `mode`, `height`, `width`.

Available methods are: `convert`, `crop`, `effect_spread`, `entropy`, `filter`, `getbands`, `getbbox`,
`getchannel`, `getcolors`, `getextrema`, `getpalette`, `getpixel`, `getprojection`, `histogram`,
`point`, `quantize`, `reduce`, `remap_palette`, `resize`, `rotate`, `transform`, `transpose`.

Methods can be chained, for example: `t.frame.resize((224, 224)).rotate(90).convert('L')`

## Functions

Functions can be used to transform data, both during querying as well as when data is added to a table.

```python
@pt.udf(return_type=pt.IntType(), param_types=[pt.IntType()])
def add1(x):
    return x + 1
```

For querying: `t[t.frame_idx, add1(t.frame_idx)].show()`

As a computed column: `t.add_column(pt.Column('c', computed_with=add1(t.frame_idx)))`

## Image similarity search

In order to enable similarity on specific image columns, create those columns with `indexed=True`.
This will compute an embedding for every image and store it in a vector index.
```python
c3 = pt.Column('frame', pt.ImageType(), indexed=True)
```

Assuming `img = PIL.Image.open(...)`

Similarity search: `t[t.frame.nearest(img)].show(10)`

Keyword search: `t[t.frame.matches('car')].show(10)`

