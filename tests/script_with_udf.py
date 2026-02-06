import pixeltable as pxt


@pxt.udf
def inline_udf(x: int) -> int:
    return x + 1


def main() -> None:
    tbl = pxt.create_table('inline_udf_test', {'x': pxt.Int}, if_exists='replace')
    tbl.add_computed_column(y=inline_udf(tbl.x))


if __name__ == '__main__':
    main()
