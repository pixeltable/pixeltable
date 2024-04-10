from pathlib import Path

import more_itertools

import pixeltable.env as env
import pixeltable.exceptions as excs
from pixeltable import Table
from pixeltable.datatransfer.remote import Remote


class LabelStudioProject(Remote):

    def __init__(
            self,
            project_id: int
    ):
        self.ls = env.Env.get().label_studio_client
        self.project_id = project_id
        self.project = self.ls.get_project(project_id)

    def push(self, t: Table, col_mapping: dict[str, str]) -> None:
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
                file = Path(row.vals[-1])  # Media column is in last position
                # Upload the media file to Label Studio
                task_id: int = self.project.import_tasks(file)[0]
                # Assemble the remaining columns into `data`
                data = {
                    col_mapping[col_name]: value
                    for col_name, value in zip(col_names, row.vals)
                }
                meta = {'row_id': row.pk}
                self.project.update_task(task_id, data=data, meta=meta)
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
        row_id = task['meta']


_PAGE_SIZE = 100  # This is the default used in the LS SDK
