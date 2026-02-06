from uuid import UUID

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=45)
def _(engine: sql.engine.Engine) -> None:
    """
    Materializes the `stores_cellmd` property on ColumnMd based on the existing logic around column and index types.
    """
    convert_table_md(engine, table_md_updater=_update_table_md)


_MEDIA_TYPE_CLASSES = {'ImageType', 'VideoType', 'AudioType', 'DocumentType'}
_TYPE_CLASSES_WITH_FILE_OFFLOADING = {'ArrayType', 'JsonType', 'BinaryType'}


def _update_table_md(table_md: dict, table_id: UUID) -> None:
    idx_val_cols = {}
    idx_undo_cols = set()
    for index_md in table_md['index_md'].values():
        idx_val_cols[index_md['index_val_col_id']] = index_md
        idx_undo_cols.add(index_md['index_val_undo_col_id'])

    for column_md in table_md['column_md'].values():
        if column_md['id'] in idx_val_cols:
            idx = idx_val_cols[column_md['id']]
            match idx['class_fqn']:
                case 'pixeltable.index.btree.BtreeIndex':
                    stores_cellmd = False
                case 'pixeltable.index.embedding_index.EmbeddingIndex':
                    stores_cellmd = True
                case _:
                    raise AssertionError(f'Unexpected index class {idx["class_fqn"]}')
        elif column_md['id'] in idx_undo_cols:
            stores_cellmd = False
        else:
            stored = column_md['stored']
            computed = column_md.get('value_expr', None) is not None
            col_type_cls = column_md['col_type']['_classname']
            is_media_type = col_type_cls in _MEDIA_TYPE_CLASSES
            supports_file_offloading = col_type_cls in _TYPE_CLASSES_WITH_FILE_OFFLOADING
            stores_cellmd = stored and (computed or is_media_type or supports_file_offloading)
        column_md['stores_cellmd'] = stores_cellmd
