"""
Auto-generates type stubs for pixeltable.catalog.model based on FORWARDED_TABLE_METHODS.

This script first uses `stubgen` to generate stubs for both `model.py` and `table.py`, then merges the relevant
method definitions from the `Table` class in `table.pyi` into the `TableModelMeta` definition in `model.pyi`.
This is necessary because the model metaclass forwards certain method calls to an underlying table instance, and
we want those methods to be properly typed on the model class.
"""

import subprocess

from pixeltable.catalog.model import FORWARDED_TABLE_METHODS


def generate_stubs() -> None:
    subprocess.run(('stubgen', '-m', 'pixeltable.catalog.model', '-o', 'target'), check=True)
    subprocess.run(('stubgen', '-m', 'pixeltable.catalog.table', '-o', 'target'), check=True)


def merge_stubs() -> None:
    with open('target/pixeltable/catalog/model.pyi', 'r', encoding='utf-8') as f:
        model_stub = f.readlines()
    with open('target/pixeltable/catalog/table.pyi', 'r', encoding='utf-8') as f:
        table_stub = f.readlines()

    defns_to_merge = ['    # BEGIN Forwarded table methods merged from table.pyi\n']
    for fn in sorted(FORWARDED_TABLE_METHODS):
        fn_stub_lines = [
            line.replace('pixeltable.', 'pxt.')
            for i, line in enumerate(table_stub)
            if f'def {fn}' in line or ('@overload' in line and f'def {fn}' in table_stub[i + 1])
        ]
        assert len(fn_stub_lines) > 0, (
            f'Method {fn!r} not found in table.pyi; check that `_FORWARDED_TABLE_METHODS` is up to date.'
        )
        defns_to_merge.extend(fn_stub_lines)
    defns_to_merge.append('    # END Forwarded table methods merged from table.pyi\n')

    # Inject the merged definitions into the model stub, immediately after the class declaration.
    class_defn_idx = next((i for i, line in enumerate(model_stub) if line.startswith('class TableModelMeta')), None)
    assert class_defn_idx is not None, '`TableModelMeta` class definition not found in model.pyi'
    generated_stub = [
        # mypy fundamentally does not understand metaclasses; the disable-error-code="override" directive is one of the
        # hacks we need to get it to cooperate.
        '# mypy: disable-error-code="override"\n\n',
        # Inject additional imports not picked up by stubgen.
        'from typing import Iterable, overload\n',
        'import pandas as pd\n',
        'import pixeltable as pxt\n',
        'from pixeltable import Query, ResultCursor, ResultSet, TableMetadata, UpdateStatus, VersionMetadata\n',
        'from pixeltable import type_system as ts\n',
        'from pixeltable.exprs import ColumnRef\n',
        'from pixeltable.globals import TableDataSource\n',
        'from pixeltable.query_clauses import JoinType\n',
        *model_stub[: class_defn_idx + 1],
        *defns_to_merge,
        *model_stub[class_defn_idx + 1 :],
    ]
    with open('pixeltable/catalog/model.pyi', 'w', encoding='utf-8') as f:
        f.writelines(generated_stub)


def main() -> None:
    print('Generating type stubs for pixeltable.catalog.model ...')
    generate_stubs()
    print('Merging forwarded method definitions from table.pyi into model.pyi ...')
    merge_stubs()
    print('Done.')


if __name__ == '__main__':
    main()
