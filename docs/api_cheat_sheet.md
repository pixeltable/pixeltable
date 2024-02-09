# API Cheat Sheet

Import conventions:
```python
import pixeltable as pxt
import pixeltable.functions as pxtf
```

Creating a client
```python
cl = pxt.Client()
```

## Client operations summary

| Task                   | Code                                                                                                                                        |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| Create a (mutable) table           | t = cl.[create_table][pixeltable.Client.create_table]('table_name', {'col_1': pxt.StringType(), 'col_2': pxt.IntType(), ...})               |
| Create a view           | t = cl.[create_view][pixeltable.Client.create_view]('view_name', base_tbl, schema={'col_1': pxt.StringType, ...}, filter=base_tbl.col > 10) |
| Create a snapshot           | t = cl.[create_view][pixeltable.Client.create_view]('snapshot_name', t, is_snapshot=True)                                                   |

The following functions apply to tables, views, and snapshots.

| Task                   | Code                                                                                                  |
|--------------------------|-------------------------------------------------------------------------------------------------------|
| Use an existing table | t = cl.[get_table][pixeltable.Client.get_table]('video_data')                                         |
| Rename a table | cl.[move][pixeltable.Client.move]('video_data', 'vd')                                 |
| Move a table | cl.[move][pixeltable.Client.move]('video_data', 'experiments.video_data')                                 |
| List tables              | cl.[list_tables][pixeltable.Client.list_tables]()                                                     |
| Delete a table           | cl.[drop_table][pixeltable.Client.drop_table]('video_data')                                           |


### Directories
| Task                   | Code                                                                                                  |
|--------------------------|-------------------------------------------------------------------------------------------------------|
| Create a directory           | cl.[create_dir][pixeltable.Client.create_dir]('experiments')                                           |
| Rename or move a directory | cl.[move][pixeltable.Client.move]('experiments', 'project_x.experiments')                             |
| Delete a directory | f = cl.[rm_dir][pixeltable.Client.rm_dir]('experiments')                                    |
| List directories | cl.[list_dirs][pixeltable.Client.list_dirs]('project_x')                   |

### Functions
| Task                   | Code                                                                                                  |
|--------------------------|-------------------------------------------------------------------------------------------------------|
| Create a stored function | cl.[create_function][pixeltable.Client.create_function]('func_name', ...)                             |
| Load a stored function   | f = cl.[get_function][pixeltable.Client.get_function]('func_name')                                    |
| Rename a stored function | cl.[move][pixeltable.Client.move]('func_name', 'better_name')                   |
| Move a stored function | cl.[move][pixeltable.Client.move]('func_name', 'experiments.func_name')                   |
| Update a stored function | cl.[update_function][pixeltable.Client.update_function]('func_name', ...)                             |
| Delete a stored function | cl.[drop_function][pixeltable.Client.drop_function]('func_name')                                      |

## Frame extraction for video data
Create a table with video data and view for the frames:
```python
v = cl.create_table('tbl_name', [pxt.Column('video', pxt.VideoType())])
from pixeltable.iterators import FrameIterator
args = {'video': v.video, 'fps': 0}
f = cl.create_view('frame_view_name', v, iterator_class=FrameIterator, iterator_args=args)
```

`fps: 0` extracts frames at the original frame rate.

## Pixeltable types
|Pixeltable type|Python type|
|----|----|
| `pxt.StringType()`| `str` |
| `pxt.IntType()`| `int` |
| `pxt.FloatType()`| `float` |
| `pxt.BoolType()`| `bool` |
| `pxt.TimestampType()`| `datetime.datetime` |
| `pxt.JsonType()`| lists and dicts that can be converted to JSON|
| `pxt.ArrayType()`| `numpy.ndarray`|
| `pxt.ImageType()`| `PIL.Image.Image`|
| `pxt.VideoType()`| `str` (the file path)|
| `pxt.AudioType()`| `str` (the file path)|


## Table operations summary
|Action| Code                                                                                             |
|----|--------------------------------------------------------------------------------------------------|
| Print table schema | t.[describe][pixeltable.Table.describe]()                                                                                   |
| Query a table | t.[select][pixeltable.Table.select](t.col2, t.col3 + 5).where(t.col1 == 'green').show()                                |
| Insert rows into a table| t.[insert][pixeltable.InsertableTable.insert]([{'col1': 'green', ...}, {'col1': 'red', ...}, ...]) |
| Add a column| t.[add_column][pixeltable.Table.add_column](new_col_name=pxt.IntType())                                                       |
| Rename a column| t.[rename_column][pixeltable.Table.rename_column]('col_name', 'new_col_name')                                                    |
| Drop a column| t.[drop_column][pixeltable.Table.drop_column]('col_name')                                                                      |
| Undo the last update operation (add/rename/drop column or insert)| t.[revert][pixeltable.Table.revert]()                                                                                     |

## Querying a table

| Action                                       | Code                                                      |
|----------------------------------------------|-----------------------------------------------------------|
| Look at 10 rows                              | t.[show][pixeltable.Table.show](10)                                              |
| Look at the oldest 10 rows                   | t.[head][pixeltable.Table.head](n=10)                                            |
| Look at the most recently added 10 rows      | t.[tail][pixeltable.Table.tail](n=10)                                            |
| Look at all rows                             | t.collect()                                             |
| Iterate over all rows as dictionaries        | for row in t.collect(): ...                             |
| Look at row for frame 15                     | t.[where][pixeltable.Table.where}(t.pos  == 15).show()                            |
| Look at rows before index 15                 | t.[where][pixeltable.Table.where](t.pos < 15).show(0)                             |
| Look at rows before index 15 with RGB frames | t.[where][pixeltable.Table.where]((t.pos < 15) & (t.frame.mode == 'RGB')).collect() |

Pixeltable supports the standard comparison operators (`>=`, `>`, `==`, `<=`, `<`).
`== None` is the equivalent of `isna()/isnull()` in Pandas.

Boolean operators are the same as in Pandas: `&` for `and`, `|` for `or`, `~` for `not`.
They also require parentheses, for example: `(t.pos < 15) & (t.frame.mode == 'RGB')`
or `~(t.frame.mode == 'RGB')`.

## Selecting and transforming columns

|Action|Code|
|----|----|
| Only retrieve the frame index and frame | t.[select][pixeltable.Table.select](t.frame_idx, t.frame).collect() |
| Look at frames rotated 90 degrees | t.[select][pixeltable.Table.select](t.frame.rotate(90)).collect() |
| Overlay frame with itself rotated 90 degrees | t.[select][pixeltable.Table.select](pxt.functions.pil.image.blend(t.frame, t.frame.rotate(90))).collect() |

## Computed columns

The values in a computed column are automatically filled when data is added:
```python
t.add_column(c_added=t.frame.rotate(30))
```
Alternatively:
```python
t['c_added'] = t.frame.rotate(30)
```

Computed columns and media columns (video, image, audio) have attributes `errortype` and `errormsg`,
which contain the exception type and string
in rows where the computation expression or media type validation results in an exception
(the column value itself will be `None`).

Example:
```python
t.where(t.c_added.errortype != None).select(t.c_added.errortype, t.c_added.errormsg).show()
```
returns the exception type and message for rows with an exception.

## Inserting data into a table
```python
t.insert([{'video': '/path/to/video1.mp4'}, {'video': '/path/to/video2.mp4'}])
```
Each row is a dictionary mapping column names to column values (do not provide values for computed columns).

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
@pxt.udf(return_type=pxt.IntType(), param_types=[pxt.IntType()])
def add1(x):
    return x + 1
```

For querying: `t.select(t.frame_idx, add1(t.frame_idx)).show()`

As a computed column: `t.add_column(c=add1(t.frame_idx))`

<!---
## Image similarity search

In order to enable similarity on specific image columns, create those columns with `indexed=True`.
This will compute an embedding for every image and store it in a vector index.
```python
c3 = pxt.Column('frame', pxt.ImageType(), indexed=True)
```

Assuming `img = PIL.Image.open(...)`

Similarity search: `t.where(t.frame.nearest(img)).show(10)`

Keyword search: `t.where(t.frame.matches('car')).show(10)`
-->
