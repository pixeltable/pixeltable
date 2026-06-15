from uuid import UUID

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=17)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, table_md_updater=__update_table_md)


def __update_table_md(table_md: dict, table_id: UUID) -> None:
    # key changes in IndexMd.init_args: img_embed -> image_embed, txt_embed -> string_embed
    if len(table_md['index_md']) == 0:
        return
    for idx_md in table_md['index_md'].values():
        if not idx_md['class_fqn'].endswith('.EmbeddingIndex'):
            continue
        init_dict = idx_md['init_args']
        init_dict['image_embed'] = init_dict['img_embed']
        del init_dict['img_embed']
        init_dict['string_embed'] = init_dict['txt_embed']
        del init_dict['txt_embed']
