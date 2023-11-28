# Differences with Pandas

| | Pandas | Pixeltable |
|:-|:---|:---|
|reading data | Read from file system with `pd.read_*` methods: eg, `.csv`, `.json`, `.parquet`, etc. | In `pixeltable`, data is stored in tables. `cl.list_tables`, `tab = cl.get_table('mytable')` |
| saving data (fist time) | Save to file system, format of choice | `table.insert` | 
| updating data | to update data persistently, use `pd.write_*()`  to over-write or save new versions of the dataset | `table.update` statements on tables allow for fine-grained persistent updates only on columns with specific values | 
| selecting rows | `df[ df.col > 1 ]` | `tab.where(tab.col > 1)` | 
| selecting rows (predicates) | `df[(df.a > 0) & (df.b > 0)]` | `df.where((df.a > 0) & (df.b > 0))` both will error if `and` or `or` is used.
| selecting columns (aka projection) | `df[['col']]` | `tab.select(tab.col)` | 
| new column with computed value | `df.assign(new_col= fun(df.input_col1, df.input_col2,..))` or `df['new_col'] = fun(df.input_col1, df.input_col2,..))` (the latter acts in-place, modifying the df object) | `tab.select(old_colA, old_colB,  new_col=fun(tab.input_col1, tab.input_col2,...))`  |
| computing new values row by row | `df['new_col'] = df.apply(fun, axis=1)` | `df.select(old_colA, old_colB, ..., new_col=pxt.function(fun)(tab.input_col1, tab.input_col2,...)` |
