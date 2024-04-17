from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator
from xml.etree import ElementTree

import more_itertools

import pixeltable.env as env
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable import Table
from pixeltable.datatransfer.remote import Remote

_logger = logging.getLogger('pixeltable')


class LabelStudioProject(Remote):

    def __init__(self, project_id: int):
        self.project_id = project_id
        self.ls_client = env.Env.get().label_studio_client
        self.project = self.ls_client.get_project(self.project_id)
        self.project_params = self.project.get_params()
        self.project_title = self.project_params['title']

    def get_push_columns(self) -> dict[str, ts.ColumnType]:
        config: str = self.project.get_params()['label_config']
        return self._parse_project_config(config)

    def get_pull_columns(self) -> dict[str, ts.ColumnType]:
        return {'annotations': ts.StringType()}

    def sync(self, t: Table, col_mapping: dict[str, str], push: bool, pull: bool) -> None:
        _logger.info(f'Syncing Label Studio project "{self.project_title}" with table `{t.get_name()}`'
                     f' (push: {push}, pull: {pull}).')
        tasks = list(self._get_all_tasks())
        if pull:
            self._update_table_from_tasks(t, col_mapping, tasks)
        if push:
            self._create_tasks_from_table(t, col_mapping, tasks)

    def _get_all_tasks(self) -> Iterator[dict]:
        page = 1
        unknown_task_count = 0
        while True:
            result = self.project.get_paginated_tasks(page=page, page_size=_PAGE_SIZE)
            if result.get('end_pagination'):
                break
            for task in result['tasks']:
                row_id = task['meta'].get('row_id')
                if row_id is None:
                    unknown_task_count += 1
                else:
                    yield task
            page += 1
        if unknown_task_count > 0:
            _logger.warning(
                f'Skipped {unknown_task_count} unrecognized task(s) when syncing Label Studio project "{self.project_title}".'
            )

    @classmethod
    def _update_table_from_tasks(cls, t: Table, col_mapping: dict[str, str], tasks: list[dict]) -> None:
        # `col_mapping` is guaranteed to be a one-to-one dict whose values are a superset
        # of `get_pull_columns`
        annotations_column = next(k for k, v in col_mapping.items() if v == 'annotations')
        updates = [
            {'row_id': task['meta']['row_id'], annotations_column: task['annotations']}
            for task in tasks
        ]
        _logger.info(
            f'Updating table `{t.get_name()}`, column `{annotations_column}` with {len(updates)} total annotations.'
        )
        # TODO: Apply updates

    def _create_tasks_from_table(self, t: Table, col_mapping: dict[str, str], existing_tasks: list[dict]) -> None:
        row_ids_in_ls = {task['meta']['row_id'] for task in existing_tasks}
        t_col_types = t.column_types()
        media_cols = [col_name for col_name, _ in col_mapping.items() if t_col_types[col_name].is_media_type()]
        if len(media_cols) > 1:
            raise excs.Error(
                f'`LabelStudioProject` supports at most 1 media column; found {len(media_cols)}: {media_cols}'
            )
        if media_cols:
            # This project has a media column. We will iterate through the rows of the Table, one
            # at a time. For each row, we upload the media file, then fill in the remaining fields.
            media_col = media_cols[0]
            col_names = [col_name for col_name in col_mapping.keys() if col_name != media_col]
            columns = [t[col_name] for col_name in col_names]
            columns.append(t[media_col].localpath)
            rows = t.select(*columns)
            for row in rows._exec():
                if row.pk not in row_ids_in_ls:
                    file = Path(row.vals[-1])  # Media column is in last position
                    # Upload the media file to Label Studio
                    task_id: int = self.project.import_tasks(file)[0]
                    # Assemble the remaining columns into `data`
                    data = {
                        col_mapping[col_name]: value
                        for col_name, value in zip(col_names, row.vals)
                    }
                    meta = {'row_id': row.pk}
                    self.project.update_task(task_id, meta=meta)
            print(f'Created {t.count()} task(s) in {self}.')
        else:
            # No media column, just structured data; we upload the rows in pages.
            rows = t.select(*col_mapping.keys())
            new_rows = filter(lambda row: row.pk not in row_ids_in_ls, rows._exec())
            for page in more_itertools.batched(new_rows, n=_PAGE_SIZE):
                tasks = [
                    {'data': zip(col_mapping.values(), row.vals),
                     'meta': {'row_id': row.pk}}
                    for row in page
                ]
                self.project.import_tasks(tasks)

    # def pull_task(self, t: Table, task: dict) -> None:
    #     if not task['annotations']:
    #         return
    #     pk_hack = task['meta']['pk_hack']
    #     annotations = task['annotations']
    #     # Total hack for the demo
    #     t.update({'annotations': annotations}, where=(t['file'] == pk_hack))

    #
    # def pull(self, t: Table, col_mapping: dict[str, str]) -> None:
    #     # rev_mapping = {v: k for k, v in col_mapping.items()}
    #     page = 1
    #     if 'annotations' not in t.column_names():
    #         t.add_column(annotations=pxt.JsonType(nullable=True))
    #     while True:
    #         result = self.project.get_paginated_tasks(page=page, page_size=_PAGE_SIZE)
    #         if result.get('end_pagination'):
    #             break
    #         for task in result['tasks']:
    #             self.pull_task(t, task)
    #         page += 1
    #     print(f'Updated annotations from {self}.')

    def to_dict(self) -> dict[str, Any]:
        return {'project_id': self.project_id}

    @classmethod
    def from_dict(cls, md: dict[str, Any]) -> LabelStudioProject:
        return LabelStudioProject(md['project_id'])

    def __repr__(self) -> str:
        name = self.project.get_params()['title']
        return f'LabelStudioProject `{name}`'

    @classmethod
    def _parse_project_config(cls, config: str) -> dict[str, ts.ColumnType]:
        """
        Parses a Label Studio XML config, extracting the names and Pixeltable types of
        all input variables.
        """
        root: ElementTree.Element = ElementTree.fromstring(config)
        if root.tag.lower() != 'view':
            raise excs.Error('Root of Label Studio config must be a `View`')
        assert root.tag == 'View'
        return dict(cls._extract_vars_from_project_config(root))

    @classmethod
    def _extract_vars_from_project_config(cls, root: ElementTree.Element) -> Iterator[(str, ts.ColumnType)]:
        for element in root:
            if 'value' in element.attrib and element.attrib['value'][0] == '$':
                element_type = _LS_TAG_MAP.get(element.tag.lower())
                if element_type is None:
                    raise excs.Error(f'Unsupported Label Studio data type: `{element.tag}`')
                yield element.attrib['value'][1:], element_type


_PAGE_SIZE = 100  # This is the default used in the LS SDK
_LS_TAG_MAP = {
    'text': ts.StringType(),
    'image': ts.ImageType(),
    'video': ts.VideoType(),
    'audio': ts.AudioType()
}
