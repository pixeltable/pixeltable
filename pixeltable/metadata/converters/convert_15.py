import uuid

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=15)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(
        engine,
        column_md_updater=__update_column_md,
        remote_md_updater=__update_remote_md,
        table_md_updater=__update_table_md
    )


def __update_table_md(table_md: dict) -> None:
    pass


def __update_column_md(column_md: dict) -> None:
    column_md['proxy_base'] = None


def __update_remote_md(remote_md: dict) -> None:
    # New schema uses a single `class` string to be consistent with other pxt metadata
    remote_md['class'] = f'{remote_md["module"]}.{remote_md["class"]}'
    del remote_md['module']

    # col_mapping moved inside class-specific md

    project_md = remote_md['remote_md']
    project_md['col_mapping'] = remote_md['col_mapping']
    del remote_md['col_mapping']

    if remote_md['class'] == 'pixeltable.datatransfer.remote.MockRemote':
        remote_md['class'] = 'pixeltable.io.external_store.MockProject'
        project_md['name'] = f'remote_{uuid.uuid4()}'
        project_md['export_cols'] = project_md['push_cols']
        project_md['import_cols'] = project_md['pull_cols']
        del project_md['push_cols']
        del project_md['pull_cols']

    elif remote_md['class'] == 'pixeltable.datatransfer.label_studio.LabelStudioProject':
        remote_md['class'] = 'pixeltable.io.label_studio.LabelStudioProject'
        # 'post' is the media_import_method for legacy LabelStudioProjects
        remote_md['remote_md']['media_import_method'] = 'post'
        # version 15 disallowed more than one ExternalStore per table, so we can safely
        # use a static default name
        remote_md['remote_md']['name'] = 'ls_project_0'

    else:
        assert False, remote_md['class']
