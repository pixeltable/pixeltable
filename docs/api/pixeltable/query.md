# Query

`Query` represents a query against a specific table. Unlike containers such as Pandas DataFrames,
Pixeltable Queries do not hold data or allow you to update data
(use [insert][pixeltable.Table.insert]/[update][pixeltable.Table.update]/[delete][pixeltable.Table.delete]
for that purpose).

## Overview

| Query Construction                          |                                                       |
|---------------------------------------------|-------------------------------------------------------|
| [`select`][pixeltable.Query.select]     | Select output expressions                             |
| [`join`][pixeltable.Query.join]         | Join to another table                                 |
| [`where`][pixeltable.Query.where]       | Filter table rows                                     |
| [`group_by`][pixeltable.Query.group_by] | Group table rows in order to apply aggregate functions |
| [`order_by`][pixeltable.Query.order_by] | Order output rows                                     |
| [`limit`][pixeltable.Query.limit]       | Limit the number of output rows                       |
| [`distinct`][pixeltable.Query.distinct] | Remove duplicate rows                                 |
| [`sample`][pixeltable.Query.sample]     | Select shuffled sample of rows                        |

| Query Execution                           |                                     |
|-------------------------------------------|-------------------------------------|
| [`collect`][pixeltable.Query.collect] | Return all output rows              |
| [`show`][pixeltable.Query.show]       | Return a number of output rows      |
| [`head`][pixeltable.Query.head]       | Return the oldest rows              |
| [`tail`][pixeltable.Query.tail]       | Return the most recently added rows |

| Data Export                                                     |                                                                                                                                      |
|-----------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| [`to_pytorch_dataset`][pixeltable.Query.to_pytorch_dataset] | Return the query result as a pytorch [`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) |
| [`to_coco_dataset`][pixeltable.Query.to_coco_dataset]       | Return the query result as a COCO dataset                                                                                            |

## ::: pixeltable.Query

    options:
      members:
      - collect
      - group_by
      - head
      - limit
      - order_by
      - select
      - join
      - show
      - tail
      - to_pytorch_dataset
      - to_coco_dataset
      - where
      - distinct
      - sample
