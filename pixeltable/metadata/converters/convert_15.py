import uuid

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=15)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, column_md_updater=update_column_md, remote_md_updater=update_remote_md)


def update_column_md(column_md: dict) -> None:
    column_md['proxy_base'] = None


def update_remote_md(remote_md: dict) -> None:
    # New schema uses a single `class` string to be consistent with other pxt metadata
    remote_md['class'] = f'{remote_md["module"]}.{remote_md["class"]}'
    del remote_md['module']

    if remote_md['class'] == 'pixeltable.datatransfer.remote.MockRemote':
        remote_md['class'] = 'pixeltable.io.external_store.MockExternalStore'
        remote_md['remote_md']['name'] = f'remote_{uuid.uuid4()}'

    elif remote_md['class'] == 'pixeltable.datatransfer.label_studio.LabelStudioProject':
        remote_md['class'] = 'pixeltable.io.label_studio.LabelStudioProject'
        # 'post' is the media_import_method for legacy LabelStudioProject remotes
        remote_md['remote_md']['media_import_method'] = 'post'

    else:
        assert False, remote_md['class']
