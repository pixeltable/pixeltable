# DataFrameResultSet

`DataFrameResultSet` represents the result of executing a DataFrame query. It's returned by methods like 
[`collect`][pixeltable.DataFrame.collect], [`show`][pixeltable.DataFrame.show], 
[`head`][pixeltable.DataFrame.head], and [`tail`][pixeltable.DataFrame.tail].

The result set provides methods to access the query results and convert them to different formats for further processing.

## Overview

| Data Access                          |                                  |
|--------------------------------------|----------------------------------|
| `__len__`                            | Returns the number of rows       |
| `__getitem__`                        | Access rows, columns or cells by index or name |
| `__iter__`                           | Iterate over rows                |

| Data Export                                                      |                                     |
|------------------------------------------------------------------|-------------------------------------|
| [`to_pandas`][pixeltable.DataFrameResultSet.to_pandas]          | Convert to Pandas DataFrame         |
| [`to_polars`][pixeltable.DataFrameResultSet.to_polars]          | Convert to Polars DataFrame         |
| [`to_pydantic`][pixeltable.DataFrameResultSet.to_pydantic]      | Convert to Pydantic model instances |

## ::: pixeltable.DataFrameResultSet

    options:
      members:
      - to_pandas
      - to_polars 
      - to_pydantic 