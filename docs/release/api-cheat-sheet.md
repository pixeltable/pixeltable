# API Cheat Sheet

Import conventions:
```python
import pixeltable as pxt
import pixeltable.functions as pxtf
```

## Operations summary

| Task                        | Code                                                                                                                             |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| Create a (mutable) table    | t = [`pxt.create_table`][pixeltable.create_table]('table_name', {'col_1': pxt.StringType(), 'col_2': pxt.IntType(), ...})        |
| Create a view               | t = [`pxt.create_view`][pixeltable.create_view]('view_name', base_tbl, filter=base_tbl.col > 10)                                 |
| Create a view with iterator | t = [`pxt.create_view`][pixeltable.create_view]('view_name', base_tbl, iterator=FrameIterator.create(video=base_tbl.col, fps=0)) |
| Create a snapshot           | t = [`pxt.create_view`][pixeltable.create_view]('snapshot_name', t, is_snapshot=True)                                            |

The following functions apply to tables, views, and snapshots.

| Task                  | Code                                                                  |
|-----------------------|-----------------------------------------------------------------------|
| Use an existing table | t = [`pxt.get_table`][pixeltable.get_table]('video_data')             |
| Rename a table        | [`pxt.move`][pixeltable.move]('video_data', 'vd')                     |
| Move a table          | [`pxt.move`][pixeltable.move]('video_data', 'experiments.video_data') |
| List tables           | [`pxt.list_tables`][pixeltable.list_tables]()                         |
| Delete a table        | [`pxt.drop_table`][pixeltable.drop_table]('video_data')               |


### Directories
| Task                       | Code                                                                  |
|----------------------------|-----------------------------------------------------------------------|
| Create a directory         | [`pxt.create_dir`][pixeltable.create_dir]('experiments')              |
| Rename or move a directory | [`pxt.move`][pixeltable.move]('experiments', 'project_x.experiments') |
| Delete a directory         | f = [`pxt.rm_dir`][pixeltable.rm_dir]('experiments')                  |
| List directories           | [`pxt.list_dirs`][pixeltable.list_dirs]('project_x')                  |

## Frame extraction for video data
Create a table with video data and view for the frames:
```python
import pixeltable as pxt
from pixeltable.iterators import FrameIterator
t = pxt.create_table('tbl_name', {'video': pxt.VideoType()})
f = pxt.create_view('frame_view_name', t, iterator=FrameIterator.create(video=t, fps=0))
```

`fps=0` extracts frames at the original frame rate.

## Pixeltable types
| Pixeltable type       | Python type                  |
|-----------------------|------------------------------|
| `pxt.StringType()`    | `str`                        |
| `pxt.IntType()`       | `int`                        |
| `pxt.FloatType()`     | `float`                      |
| `pxt.BoolType()`      | `bool`                       |
| `pxt.TimestampType()` | `datetime.datetime`          |
| `pxt.JsonType()`      | `list` or `dict`             |
| `pxt.ArrayType()`     | `numpy.ndarray`              |
| `pxt.ImageType()`     | `PIL.Image.Image`            |
| `pxt.VideoType()`     | `str` (the file path or URL) |
| `pxt.AudioType()`     | `str` (the file path or URL) |


## Table operations summary

| Action                                                            | Code                                                                                                 |
|-------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| Print table schema                                                | t.[`describe`][pixeltable.Table.describe]()                                                          |
| Query a table                                                     | t.[`select`][pixeltable.Table.select](t.col2, t.col3 + 5).where(t.col1 == 'green').show()            |
| Insert a single row into a table                                  | t.[`insert`][pixeltable.InsertableTable.insert](col1='green', ...)                                   |
| Insert multiple rows into a table                                 | t.[`insert`][pixeltable.InsertableTable.insert]([{'col1': 'green', ...}, {'col1': 'red', ...}, ...]) |
| Add a column                                                      | t.[`add_column`][pixeltable.Table.add_column](new_col_name=pxt.IntType())                            |
| Add a column (alternate form)                                     | t[new_col_name] = pxt.IntType()                                                                      |
| Rename a column                                                   | t.[`rename_column`][pixeltable.Table.rename_column]('col_name', 'new_col_name')                      |
| Drop a column                                                     | t.[`drop_column`][pixeltable.Table.drop_column]('col_name')                                          |
| Undo the last update operation (add/rename/drop column or insert) | t.[`revert`][pixeltable.Table.revert]()                                                              |

## Querying a table

| Action                                       | Code                                                                                  |
|----------------------------------------------|---------------------------------------------------------------------------------------|
| Look at 10 rows                              | t.[`show`][pixeltable.Table.show](10)                                                 |
| Look at the oldest 10 rows                   | t.[`head`][pixeltable.Table.head](10)                                                 |
| Look at the most recently added 10 rows      | t.[`tail`][pixeltable.Table.tail](10)                                                 |
| Look at all rows                             | t.[`collect`][pixeltable.Table.collect]()                                             |
| Iterate over all rows as dictionaries        | for row in t.[`collect`][pixeltable.Table.collect](): ...                             |
| Look at row for frame 15                     | t.[`where`][pixeltable.Table.where](t.pos  == 15).show()                              |
| Look at rows before index 15                 | t.[`where`][pixeltable.Table.where](t.pos < 15).show()                                |
| Look at rows before index 15 with RGB frames | t.[`where`][pixeltable.Table.where]((t.pos < 15) & (t.frame.mode == 'RGB')).collect() |

Pixeltable supports the standard comparison operators (`>=`, `>`, `==`, `<=`, `<`).
`== None` is the equivalent of `isna()/isnull()` in Pandas.

Boolean operators are the same as in Pandas: `&` for `and`, `|` for `or`, `~` for `not`.
They also require parentheses, for example: `(t.pos < 15) & (t.frame.mode == 'RGB')`
or `~(t.frame.mode == 'RGB')`.

## Selecting and transforming columns

| Action                                       | Code                                                                                                        |
|----------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| Only retrieve the frame index and frame      | t.[`select`][pixeltable.Table.select](t.frame_idx, t.frame).collect()                                       |
| Look at frames rotated 90 degrees            | t.[`select`][pixeltable.Table.select](t.frame.rotate(90)).collect()                                         |
| Overlay frame with itself rotated 90 degrees | t.[`select`][pixeltable.Table.select](pxt.functions.pil.image.blend(t.frame, t.frame.rotate(90))).collect() |

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

Images are represented as `PIL.Image.Image` instances in memory and support a lot of the
attributes and methods documented
[here](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image).

Available attributes are: `mode`, `height`, `width`.

Available methods are: `convert`, `crop`, `effect_spread`, `entropy`, `filter`, `getbands`, `getbbox`,
`getchannel`, `getcolors`, `getextrema`, `getpalette`, `getpixel`, `getprojection`, `histogram`,
`point`, `quantize`, `reduce`, `remap_palette`, `resize`, `rotate`, `transform`, `transpose`.

Methods can be chained, for example: `t.frame.resize((224, 224)).rotate(90).convert('L')`
