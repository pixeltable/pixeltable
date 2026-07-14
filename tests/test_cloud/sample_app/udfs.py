import pixeltable as pxt


@pxt.query
def find_by_id(item_id: int) -> pxt.Query:
    items = pxt.get_table('e2e_items')
    return items.where(items.id == item_id)
