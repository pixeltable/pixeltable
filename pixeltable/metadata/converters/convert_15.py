import uuid

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


def convert_15(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, column_md_updater=update_column_md, remote_md_updater=update_remote_md)


def update_column_md(column_md: dict) -> None:
    column_md['proxy_base'] = None


def update_remote_md(remote_md: dict) -> None:
    if remote_md['class'] == 'MockRemote':
        remote_md['remote_md']['name'] = f'remote_{uuid.uuid4()}'
    if remote_md['class'] == 'LabelStudioProject':
        # 'post' is the media_import_method for legacy remotes
        remote_md['remote_md']['media_import_method'] = 'post'


register_converter(15, convert_15)
