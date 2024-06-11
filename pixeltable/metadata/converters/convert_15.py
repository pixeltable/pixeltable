import uuid

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=15)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(
        engine,
        table_md_updater=__update_table_md,
        column_md_updater=__update_column_md,
        external_store_md_updater=__update_external_store_md
    )


def __update_table_md(table_md: dict) -> None:
    table_md['external_stores'] = table_md['remotes']
    del table_md['remotes']


def __update_column_md(column_md: dict) -> None:
    column_md['proxy_base'] = None


def __update_external_store_md(store_md: dict) -> None:
    # New schema uses a single `class` string to be consistent with other pxt metadata
    store_md['class'] = f'{store_md["module"]}.{store_md["class"]}'
    del store_md['module']
    # We're no longer calling them `remote`s
    store_md['md'] = store_md['remote_md']
    del store_md['remote_md']

    # col_mapping moved inside class-specific md
    project_md = store_md['md']
    project_md['col_mapping'] = store_md['col_mapping']
    del store_md['col_mapping']

    if store_md['class'] == 'pixeltable.datatransfer.remote.MockRemote':
        store_md['class'] = 'pixeltable.io.external_store.MockProject'
        project_md['name'] = f'project_{uuid.uuid4()}'
        project_md['export_cols'] = project_md['push_cols']
        project_md['import_cols'] = project_md['pull_cols']
        del project_md['push_cols']
        del project_md['pull_cols']

    elif store_md['class'] == 'pixeltable.datatransfer.label_studio.LabelStudioProject':
        store_md['class'] = 'pixeltable.io.label_studio.LabelStudioProject'
        # 'post' is the media_import_method for legacy LabelStudioProjects
        store_md['md']['media_import_method'] = 'post'
        # version 15 disallowed more than one ExternalStore per table, so we can safely
        # use a static default name
        store_md['md']['name'] = 'ls_project_0'

    else:
        assert False, store_md['class']
