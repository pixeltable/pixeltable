from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

import label_studio_sdk
import more_itertools

import pixeltable.env as env
import pixeltable.exceptions as excs
from pixeltable import Table, View
from pixeltable.datatransfer.remote import Remote


class LabelStudioProject(Remote):

    def __init__(self, project_id: int):
        self.project_id = project_id
        self._project: Optional[label_studio_sdk.Project] = None

    @property
    def project(self) -> label_studio_sdk.Project:
        if self._project is None:
            ls_client = env.Env.get().label_studio_client
            self._project = ls_client.get_project(self.project_id)
        return self._project

    def push(self, t: Table, col_mapping: dict[str, str]) -> None:
        t_col_types = t.column_types()
        media_cols = [col_name for col_name, _ in col_mapping.items() if t_col_types[col_name].is_media_type()]
        if len(media_cols) > 1:
            raise excs.Error(
                f'`LabelStudioProject` supports at most 1 media column; found {len(media_cols)}: {media_cols}'
            )
        print('PUSH')
        print(col_mapping)
        print(media_cols)
        if media_cols:
            # This project has a media column. We will iterate through the rows of the Table, one
            # at a time. For each row, we upload the media file, then fill in the remaining fields.
            media_col = media_cols[0]
            col_names = [col_name for col_name in col_mapping.keys() if col_name != media_col]
            columns = [t[col_name] for col_name in col_names]
            columns.append(t[media_col].localpath)
            rows = t.select(*columns)
            for row in rows._exec():
                file = Path(row.vals[-1])  # Media column is in last position
                # Upload the media file to Label Studio
                task_id: int = self.project.import_tasks(file)[0]
                # Assemble the remaining columns into `data`
                data = {
                    col_mapping[col_name]: value
                    for col_name, value in zip(col_names, row.vals)
                }
                meta = {'row_id': row.pk, 'pk_hack': row.vals[-1]}
                self.project.update_task(task_id, meta=meta)
        else:
            # No media column, just structured data; we upload the rows in pages.
            rows = t.select(*col_mapping.keys())
            for page in more_itertools.batched(rows._exec(), n=_PAGE_SIZE):
                tasks = [
                    {'data': zip(col_mapping.values(), row.vals),
                     'meta': {'row_id': row.pk}}
                    for row in page
                ]
                print(tasks)
                self.project.import_tasks(tasks)

    def pull(self, t: Table, col_mapping: dict[str, str]) -> None:
        rev_mapping = {v: k for k, v in col_mapping.items()}
        page = 1
        while True:
            result = self.project.get_paginated_tasks(page=page, page_size=_PAGE_SIZE)
            if result.get('end_pagination'):
                break
            for task in result['tasks']:
                self.pull_task(t, rev_mapping, task)
            page += 1

    def pull_task(self, t: Table, rev_mapping: dict[str, str], task: dict) -> None:
        pk_hack = task['meta']['pk_hack']

    def to_dict(self) -> dict[str, Any]:
        return {'project_id': self.project_id}

    @classmethod
    def from_dict(cls, md: dict[str, Any]) -> LabelStudioProject:
        return LabelStudioProject(md['project_id'])

    def __repr__(self) -> str:
        name = self.project.get_params()['title']
        return f'LabelStudioProject `{name}`'


_PAGE_SIZE = 100  # This is the default used in the LS SDK
