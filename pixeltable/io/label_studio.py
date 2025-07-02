import copy
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Optional
from xml.etree import ElementTree as ET

import label_studio_sdk
import PIL.Image
from requests.exceptions import HTTPError

import pixeltable.type_system as ts
from pixeltable import Column, Table, env, exceptions as excs
from pixeltable.catalog import ColumnHandle
from pixeltable.catalog.update_status import RowCountStats, UpdateStatus
from pixeltable.config import Config
from pixeltable.exprs import ColumnRef, DataRow, Expr
from pixeltable.io.external_store import Project
from pixeltable.utils import coco

# label_studio_sdk>=1 and label_studio_sdk<1 are not compatible, so we need to try
# the import two different ways to insure intercompatibility
try:
    # label_studio_sdk<1 compatibility
    import label_studio_sdk.project as ls_project  # type: ignore
except ImportError:
    # label_studio_sdk>=1 compatibility
    import label_studio_sdk._legacy.project as ls_project

_logger = logging.getLogger('pixeltable')


@env.register_client('label_studio')
def _(api_key: str, url: str) -> label_studio_sdk.Client:
    return label_studio_sdk.Client(api_key=api_key, url=url)


def _label_studio_client() -> label_studio_sdk.Client:
    return env.Env.get().get_client('label_studio')


class LabelStudioProject(Project):
    """
    An [`ExternalStore`][pixeltable.io.ExternalStore] that represents a Label Studio project, providing functionality
    for synchronizing between a Pixeltable table and a Label Studio project.
    """

    project_id: int  # Label Studio project ID
    media_import_method: Literal['post', 'file', 'url']
    _project: Optional[ls_project.Project]

    def __init__(
        self,
        name: str,
        project_id: int,
        media_import_method: Literal['post', 'file', 'url'],
        col_mapping: dict[ColumnHandle, str],
        stored_proxies: Optional[dict[ColumnHandle, ColumnHandle]] = None,
    ):
        """
        The constructor will NOT create a new Label Studio project; it is also used when loading
        metadata for existing projects.
        """
        self.project_id = project_id
        self.media_import_method = media_import_method
        self._project = None
        super().__init__(name, col_mapping, stored_proxies)

    @property
    def project(self) -> ls_project.Project:
        """The `Project` object corresponding to this Label Studio project."""
        if self._project is None:
            try:
                self._project = _label_studio_client().get_project(self.project_id)
            except HTTPError as exc:
                raise excs.Error(
                    f'Could not locate Label Studio project: {self.project_id} '
                    '(cannot connect to server or project no longer exists)'
                ) from exc
        return self._project

    @property
    def project_params(self) -> dict[str, Any]:
        """The parameters of this Label Studio project."""
        return self.project.get_params()

    @property
    def project_title(self) -> str:
        """The title of this Label Studio project."""
        return self.project_params['title']

    @property
    def __project_config(self) -> '_LabelStudioConfig':
        return self.__parse_project_config(self.project_params['label_config'])

    def get_export_columns(self) -> dict[str, ts.ColumnType]:
        """
        The data keys and preannotation fields specified in this Label Studio project.
        """
        return self.__project_config.export_columns

    def get_import_columns(self) -> dict[str, ts.ColumnType]:
        """
        Always contains a single entry:

        ```
        {"annotations": ts.JsonType(nullable=True)}
        ```
        """
        return {ANNOTATIONS_COLUMN: ts.JsonType(nullable=True)}

    def sync(self, t: Table, export_data: bool, import_data: bool) -> UpdateStatus:
        _logger.info(
            f'Syncing Label Studio project "{self.project_title}" with table `{t._name}`'
            f' (export: {export_data}, import: {import_data}).'
        )
        # Collect all existing tasks into a dict with entries `rowid: task`
        tasks = {tuple(task['meta']['rowid']): task for task in self.__fetch_all_tasks()}
        sync_status = UpdateStatus()
        if export_data:
            export_sync_status = self.__update_tasks(t, tasks)
            sync_status += export_sync_status
        if import_data:
            import_sync_status = self.__update_table_from_tasks(t, tasks)
            sync_status += import_sync_status
        return sync_status

    def __fetch_all_tasks(self) -> Iterator[dict[str, Any]]:
        """Retrieves all tasks and task metadata in this Label Studio project."""
        page = 1
        unknown_task_count = 0
        while True:
            result = self.project.get_paginated_tasks(page=page, page_size=_PAGE_SIZE)
            if result.get('end_pagination'):
                break
            for task in result['tasks']:
                rowid = task['meta'].get('rowid')
                if rowid is None:
                    unknown_task_count += 1
                else:
                    yield task
            page += 1
        if unknown_task_count > 0:
            _logger.warning(
                f'Skipped {unknown_task_count} unrecognized task(s) when syncing '
                f'Label Studio project {self.project_title!r}.'
            )

    def __update_tasks(self, t: Table, existing_tasks: dict[tuple, dict]) -> UpdateStatus:
        """
        Updates all tasks in this Label Studio project based on the Pixeltable data:
        - Creates new tasks for rows that don't map to any existing task;
        - Updates existing tasks for rows whose data has changed;
        - Deletes any tasks whose rows no longer exist in the Pixeltable table.
        """
        config = self.__project_config

        # Columns in `t` that map to Label Studio data keys
        t_data_cols = [t_col for t_col, ext_col_name in self.col_mapping.items() if ext_col_name in config.data_keys]

        if len(t_data_cols) == 0:
            return UpdateStatus()

        # Columns in `t` that map to `rectanglelabels` preannotations
        t_rl_cols = [
            t_col for t_col, ext_col_name in self.col_mapping.items() if ext_col_name in config.rectangle_labels
        ]

        # Destinations for `rectanglelabels` preannotations
        rl_info = list(config.rectangle_labels.values())

        _logger.debug('`t_data_cols`: %s', t_data_cols)
        _logger.debug('`t_rl_cols`: %s', t_rl_cols)
        _logger.debug('`rl_info`: %s', rl_info)

        if self.media_import_method == 'post':
            # Send media to Label Studio by HTTP post.
            assert len(t_data_cols) == 1  # This was verified when the project was set up
            return self.__update_tasks_by_post(t, existing_tasks, t_data_cols[0], t_rl_cols, rl_info)
        elif self.media_import_method in ('file', 'url'):
            # Send media to Label Studio by file reference (local file or URL).
            return self.__update_tasks_by_files(t, existing_tasks, t_data_cols, t_rl_cols, rl_info)
        else:
            raise AssertionError()

    def __update_tasks_by_post(
        self,
        t: Table,
        existing_tasks: dict[tuple, dict],
        media_col: ColumnHandle,
        t_rl_cols: list[ColumnHandle],
        rl_info: list['_RectangleLabel'],
    ) -> UpdateStatus:
        is_stored = media_col.get().is_stored
        # If it's a stored column, we can use `localpath`
        localpath_col_opt = [t[media_col.get().name].localpath] if is_stored else []
        # Select the media column, rectanglelabels columns, and localpath (if appropriate)
        rows = t.select(t[media_col.get().name], *[t[col.get().name] for col in t_rl_cols], *localpath_col_opt)
        tasks_created = 0
        row_ids_in_pxt: set[tuple] = set()

        for row in rows._exec():
            media_col_idx = rows._select_list_exprs[0].slot_idx
            rl_col_idxs = [expr.slot_idx for expr in rows._select_list_exprs[1 : 1 + len(t_rl_cols)]]
            row_ids_in_pxt.add(row.rowid)
            if row.rowid not in existing_tasks:
                # Upload the media file to Label Studio
                if is_stored:
                    # There is an existing localpath; use it!
                    localpath_col_idx = rows._select_list_exprs[-1].slot_idx
                    file = Path(row[localpath_col_idx])
                    task_id: int = self.project.import_tasks(file)[0]
                else:
                    # No localpath; create a temp file and upload it
                    assert isinstance(row[media_col_idx], PIL.Image.Image)
                    file = env.Env.get().create_tmp_path(extension='.png')
                    row[media_col_idx].save(file, format='png')
                    task_id = self.project.import_tasks(file)[0]
                    os.remove(file)

                # Update the task with `rowid` metadata
                self.project.update_task(task_id, meta={'rowid': row.rowid})

                # Convert coco annotations to predictions
                coco_annotations = [row[i] for i in rl_col_idxs]
                _logger.debug('`coco_annotations`: %s', coco_annotations)
                predictions = [
                    self.__coco_to_predictions(
                        coco_annotations[i], self.col_mapping[t_rl_cols[i]], rl_info[i], task_id=task_id
                    )
                    for i in range(len(coco_annotations))
                ]
                _logger.debug('`predictions`: {%s}', predictions)
                self.project.create_predictions(predictions)
                tasks_created += 1

        env.Env.get().console_logger.info(f'Created {tasks_created} new task(s) in {self}.')

        sync_status = UpdateStatus(ext_row_count_stats=RowCountStats(ins_rows=tasks_created))

        deletion_sync_status = self.__delete_stale_tasks(existing_tasks, row_ids_in_pxt, tasks_created)
        sync_status += deletion_sync_status
        return sync_status

    def __update_tasks_by_files(
        self,
        t: Table,
        existing_tasks: dict[tuple, dict],
        t_data_cols: list[ColumnHandle],
        t_rl_cols: list[ColumnHandle],
        rl_info: list['_RectangleLabel'],
    ) -> UpdateStatus:
        ext_data_cols = [self.col_mapping[col] for col in t_data_cols]
        expr_refs: dict[str, Expr] = {}  # kwargs for the select statement
        for col in t_data_cols:
            col_name = col.get().name
            if self.media_import_method == 'url':
                expr_refs[col_name] = t[col_name].fileurl
            else:
                assert self.media_import_method == 'file'
                if not col.get().col_type.is_media_type():
                    # Not a media column; query the data directly
                    expr_refs[col_name] = t[col_name]
                elif col in self.stored_proxies:
                    # Media column that has a stored proxy; use it. We have to give it a name,
                    # since it's an anonymous column
                    stored_proxy_col = self.stored_proxies[col].get()
                    expr_refs[f'{col_name}_proxy'] = ColumnRef(stored_proxy_col).localpath
                else:
                    # Media column without a stored proxy; this means it's a stored computed column,
                    # and we can just use the localpath
                    expr_refs[col_name] = t[col_name].localpath

        df = t.select(*[t[col.get().name] for col in t_rl_cols], **expr_refs)
        # The following buffers will hold `DataRow` indices that correspond to each of the selected
        # columns. `rl_col_idxs` holds the indices for the columns that map to RectangleLabels
        # preannotations; `data_col_idxs` holds the indices for the columns that map to data fields.
        # We have to wait until we begin iterating to populate them, so they're initially `None`.
        rl_col_idxs: Optional[list[int]] = None
        data_col_idxs: Optional[list[int]] = None

        row_ids_in_pxt: set[tuple] = set()
        tasks_created = 0
        tasks_updated = 0
        page: list[dict[str, Any]] = []  # buffer to hold tasks for paginated API calls

        # Function that turns a `DataRow` into a `dict` for creating or updating a task in the
        # Label Studio SDK.
        def create_task_info(row: DataRow) -> dict[str, Any]:
            data_vals = [row[idx] for idx in data_col_idxs]
            coco_annotations = [row[idx] for idx in rl_col_idxs]
            for i in range(len(t_data_cols)):
                if t_data_cols[i].get().col_type.is_media_type():
                    # Special handling for media columns
                    assert isinstance(data_vals[i], str)
                    if self.media_import_method == 'url':
                        data_vals[i] = self.__validate_fileurl(t_data_cols[i].get(), data_vals[i])
                    else:
                        assert self.media_import_method == 'file'
                        data_vals[i] = self.__localpath_to_lspath(data_vals[i])
            predictions = [
                self.__coco_to_predictions(coco_annotations[i], self.col_mapping[t_rl_cols[i]], rl_info[i])
                for i in range(len(coco_annotations))
            ]
            return {
                'data': dict(zip(ext_data_cols, data_vals)),
                'meta': {'rowid': row.rowid},
                'predictions': predictions,
            }

        for row in df._exec():
            if rl_col_idxs is None:
                rl_col_idxs = [expr.slot_idx for expr in df._select_list_exprs[: len(t_rl_cols)]]
                data_col_idxs = [expr.slot_idx for expr in df._select_list_exprs[len(t_rl_cols) :]]
            row_ids_in_pxt.add(row.rowid)
            task_info = create_task_info(row)
            # TODO(aaron-siegel): Implement more efficient update logic (currently involves a full table scan)
            if row.rowid in existing_tasks:
                # A task for this row already exists; see if it needs an update.
                existing_task = existing_tasks[row.rowid]
                if (
                    task_info['data'] != existing_task['data']
                    or task_info['predictions'] != existing_task['predictions']
                ):
                    _logger.debug(f'Updating task for rowid {row.rowid}.')
                    self.project.update_task(existing_tasks[row.rowid]['id'], **task_info)
                    tasks_updated += 1
            else:
                # No task exists for this row; we need to create one.
                page.append(task_info)
                tasks_created += 1
                if len(page) == _PAGE_SIZE:
                    self.project.import_tasks(page)
                    page.clear()

        if len(page) > 0:
            self.project.import_tasks(page)

        env.Env.get().console_logger.info(
            f'Created {tasks_created} new task(s) and updated {tasks_updated} existing task(s) in {self}.'
        )

        sync_status = UpdateStatus(ext_row_count_stats=RowCountStats(ins_rows=tasks_created, upd_rows=tasks_updated))

        deletion_sync_status = self.__delete_stale_tasks(existing_tasks, row_ids_in_pxt, tasks_created)
        sync_status += deletion_sync_status
        return sync_status

    @classmethod
    def __validate_fileurl(cls, col: Column, url: str) -> Optional[str]:
        # Check that the URL is one that will be visible to Label Studio. If it isn't, log an info message
        # to help users debug the issue.
        if not (url.startswith('http://') or url.startswith('https://')):
            _logger.info(
                f'URL found in media column `{col.name}` will not render correctly in Label Studio, since '
                f'it is not an HTTP URL: {url}'
            )
        return url

    @classmethod
    def __localpath_to_lspath(cls, localpath: str) -> str:
        # Transform the local path into Label Studio's bespoke path format.
        relpath = Path(localpath).relative_to(Config.get().home)
        return f'/data/local-files/?d={relpath}'

    def __delete_stale_tasks(
        self, existing_tasks: dict[tuple, dict], row_ids_in_pxt: set[tuple], tasks_created: int
    ) -> UpdateStatus:
        deleted_rowids = set(existing_tasks.keys()) - row_ids_in_pxt
        # Sanity check the math
        assert len(deleted_rowids) == len(existing_tasks) + tasks_created - len(row_ids_in_pxt)
        tasks_to_delete = [existing_tasks[rowid]['id'] for rowid in deleted_rowids]

        if len(tasks_to_delete) > 0:
            self.project.delete_tasks(tasks_to_delete)
            env.Env.get().console_logger.info(
                f'Deleted {len(tasks_to_delete)} tasks(s) in {self} that are no longer present in Pixeltable.'
            )

        # Remove them from the `existing_tasks` dict so that future updates are applied correctly
        for rowid in deleted_rowids:
            del existing_tasks[rowid]

        return UpdateStatus(ext_row_count_stats=RowCountStats(del_rows=len(deleted_rowids)))

    def __update_table_from_tasks(self, t: Table, tasks: dict[tuple, dict]) -> UpdateStatus:
        if ANNOTATIONS_COLUMN not in self.col_mapping.values():
            return UpdateStatus()

        annotations = {
            # Replace [] by None to indicate no annotations. We do want to sync rows with no annotations,
            # in order to properly handle the scenario where existing annotations have been deleted in
            # Label Studio.
            tuple(task['meta']['rowid']): task[ANNOTATIONS_COLUMN] if len(task[ANNOTATIONS_COLUMN]) > 0 else None
            for task in tasks.values()
        }

        local_annotations_col = next(k for k, v in self.col_mapping.items() if v == ANNOTATIONS_COLUMN).get()

        # Prune the annotations down to just the ones that have actually changed.
        rows = t.select(t[local_annotations_col.name])
        for row in rows._exec():
            assert len(row.vals) == 1
            if row.rowid in annotations and annotations[row.rowid] == row[0]:
                del annotations[row.rowid]

        # Apply updates
        updates = [{'_rowid': rowid, local_annotations_col.name: ann} for rowid, ann in annotations.items()]
        if len(updates) > 0:
            _logger.info(
                f'Updating table {t._name!r}, column {local_annotations_col.name!r} '
                f'with {len(updates)} total annotations.'
            )
            # batch_update currently doesn't propagate from views to base tables. As a workaround, we call
            # batch_update on the actual ancestor table that holds the annotations column.
            # TODO(aaron-siegel): Simplify this once propagation is properly implemented in batch_update
            ancestor = t
            while local_annotations_col not in ancestor._tbl_version.get().cols:
                assert ancestor._get_base_table is not None
                ancestor = ancestor._get_base_table()
            update_status = ancestor.batch_update(updates)
            env.Env.get().console_logger.info(f'Updated annotation(s) from {len(updates)} task(s) in {self}.')
            return update_status
        else:
            return UpdateStatus()

    def as_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'project_id': self.project_id,
            'media_import_method': self.media_import_method,
            'col_mapping': [[k.as_dict(), v] for k, v in self.col_mapping.items()],
            'stored_proxies': [[k.as_dict(), v.as_dict()] for k, v in self.stored_proxies.items()],
        }

    @classmethod
    def from_dict(cls, md: dict[str, Any]) -> 'LabelStudioProject':
        return LabelStudioProject(
            md['name'],
            md['project_id'],
            md['media_import_method'],
            {ColumnHandle.from_dict(entry[0]): entry[1] for entry in md['col_mapping']},
            {ColumnHandle.from_dict(entry[0]): ColumnHandle.from_dict(entry[1]) for entry in md['stored_proxies']},
        )

    def __repr__(self) -> str:
        name = self.project.get_params()['title']
        return f'LabelStudioProject `{name}`'

    @classmethod
    def __parse_project_config(cls, xml_config: str) -> '_LabelStudioConfig':
        """
        Parses a Label Studio XML config, extracting the names and Pixeltable types of
        all input variables.
        """
        root: ET.Element = ET.fromstring(xml_config)
        if root.tag.lower() != 'view':
            raise excs.Error('Root of Label Studio config must be a `View`')
        config = _LabelStudioConfig(
            data_keys=cls.__parse_data_keys_config(root), rectangle_labels=cls.__parse_rectangle_labels_config(root)
        )
        config.validate()
        return config

    @classmethod
    def __parse_data_keys_config(cls, root: ET.Element) -> dict[str, '_DataKey']:
        """Parses the data keys from a Label Studio XML config."""
        config: dict[str, '_DataKey'] = {}
        for element in root:
            if 'value' in element.attrib and element.attrib['value'][0] == '$':
                external_col_name = element.attrib['value'][1:]
                name = element.attrib.get('name')
                column_type = _LS_TAG_MAP.get(element.tag.lower())
                if column_type is None:
                    raise excs.Error(
                        f'Unsupported Label Studio data type: `{element.tag}` (in data key `{external_col_name}`)'
                    )
                config[external_col_name] = _DataKey(name=name, column_type=column_type)
        return config

    @classmethod
    def __parse_rectangle_labels_config(cls, root: ET.Element) -> dict[str, '_RectangleLabel']:
        """Parses the RectangleLabels from a Label Studio XML config."""
        config: dict[str, '_RectangleLabel'] = {}
        for element in root:
            if element.tag.lower() == 'rectanglelabels':
                name = element.attrib['name']
                to_name = element.attrib['toName']
                labels = [child.attrib['value'] for child in element if child.tag.lower() == 'label']
                for label in labels:
                    if label not in coco.COCO_2017_CATEGORIES.values():
                        raise excs.Error(f'Label in `rectanglelabels` config is not a valid COCO object name: {label}')
                config[name] = _RectangleLabel(to_name=to_name, labels=labels)
        return config

    @classmethod
    def __coco_to_predictions(
        cls, coco_annotations: dict[str, Any], from_name: str, rl_info: '_RectangleLabel', task_id: Optional[int] = None
    ) -> dict[str, Any]:
        width = coco_annotations['image']['width']
        height = coco_annotations['image']['height']
        result = [
            {
                'id': f'result_{i}',
                'type': 'rectanglelabels',
                'from_name': from_name,
                'to_name': rl_info.to_name,
                'image_rotation': 0,
                'original_width': width,
                'original_height': height,
                'value': {
                    'rotation': 0,
                    # Label Studio expects image coordinates as % of image dimensions
                    'x': entry['bbox'][0] * 100.0 / width,
                    'y': entry['bbox'][1] * 100.0 / height,
                    'width': entry['bbox'][2] * 100.0 / width,
                    'height': entry['bbox'][3] * 100.0 / height,
                    'rectanglelabels': [coco.COCO_2017_CATEGORIES[entry['category']]],
                },
            }
            for i, entry in enumerate(coco_annotations['annotations'])
            # include only the COCO labels that match a rectanglelabel name
            if coco.COCO_2017_CATEGORIES[entry['category']] in rl_info.labels
        ]
        if task_id is not None:
            return {'task': task_id, 'result': result}
        else:
            return {'result': result}

    def delete(self) -> None:
        """
        Deletes this Label Studio project. This will remove all data and annotations
        associated with this project in Label Studio.
        """
        title = self.project_title
        _label_studio_client().delete_project(self.project_id)
        env.Env.get().console_logger.info(f'Deleted Label Studio project: {title}')

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LabelStudioProject) and self.project_id == other.project_id

    def __hash__(self) -> int:
        return hash(self.project_id)

    @classmethod
    def create(
        cls,
        t: Table,
        label_config: str,
        name: Optional[str],
        title: Optional[str],
        media_import_method: Literal['post', 'file', 'url'],
        col_mapping: Optional[dict[str, str]],
        s3_configuration: Optional[dict[str, Any]],
        **kwargs: Any,
    ) -> 'LabelStudioProject':
        """
        Creates a new Label Studio project, using the Label Studio client configured in Pixeltable.
        """
        # Check that the config is valid before creating the project
        config = cls.__parse_project_config(label_config)

        if name is None:
            # Create a default name that's unique to the table
            all_stores = t.external_stores()
            n = 0
            while f'ls_project_{n}' in all_stores:
                n += 1
            name = f'ls_project_{n}'

        if title is None:
            # `title` defaults to table name
            title = t._name

        # Create a column to hold the annotations, if one does not yet exist
        if col_mapping is None or ANNOTATIONS_COLUMN in col_mapping.values():
            if col_mapping is None:
                local_annotations_column = ANNOTATIONS_COLUMN
            else:
                local_annotations_column = next(k for k, v in col_mapping.items() if v == ANNOTATIONS_COLUMN)
            if local_annotations_column not in t._get_schema():
                t.add_columns({local_annotations_column: ts.Json})

        resolved_col_mapping = cls.validate_columns(
            t, config.export_columns, {ANNOTATIONS_COLUMN: ts.JsonType(nullable=True)}, col_mapping
        )

        # Perform some additional validation
        if media_import_method == 'post' and len(config.data_keys) > 1:
            raise excs.Error('`media_import_method` cannot be `post` if there is more than one data key')

        if s3_configuration is not None:
            if media_import_method != 'url':
                raise excs.Error("`s3_configuration` is only valid when `media_import_method == 'url'`")
            s3_configuration = copy.copy(s3_configuration)
            if 'bucket' not in s3_configuration:
                raise excs.Error('`s3_configuration` must contain a `bucket` field')
            if 'title' not in s3_configuration:
                s3_configuration['title'] = 'Pixeltable-S3-Import-Storage'
            if (
                'aws_access_key_id' not in s3_configuration
                and 'aws_secret_access_key' not in s3_configuration
                and 'aws_session_token' not in s3_configuration
            ):
                # Attempt to fill any missing credentials from the environment
                try:
                    import boto3

                    s3_credentials = boto3.Session().get_credentials().get_frozen_credentials()
                    _logger.info(f'Using AWS credentials from the environment for Label Studio project: {title}')
                    s3_configuration['aws_access_key_id'] = s3_credentials.access_key
                    s3_configuration['aws_secret_access_key'] = s3_credentials.secret_key
                    s3_configuration['aws_session_token'] = s3_credentials.token
                except Exception as exc:
                    # This is not necessarily a problem, but we should log that it happened
                    _logger.debug(f'Unable to retrieve AWS credentials from the environment: {exc}')
                    pass

        _logger.info(f'Creating Label Studio project: {title}')
        project = _label_studio_client().start_project(title=title, label_config=label_config, **kwargs)

        if media_import_method == 'file':
            # We need to set up a local storage connection to receive media files
            os.environ['LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT'] = str(Config.get().home)
            try:
                project.connect_local_import_storage(local_store_path=str(env.Env.get().media_dir))
            except HTTPError as exc:
                if exc.errno == 400:
                    response: dict = json.loads(exc.response.text)
                    if (
                        'validation_errors' in response
                        and 'non_field_errors' in response['validation_errors']
                        and 'LOCAL_FILES_SERVING_ENABLED' in response['validation_errors']['non_field_errors'][0]
                    ):
                        raise excs.Error(
                            '`media_import_method` is set to `file`, but your Label Studio server is not configured '
                            'for local file storage.\nPlease set the `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED` '
                            'environment variable to `true` in the environment where your Label Studio server '
                            'is running.'
                        ) from exc
                raise  # Handle any other exception type normally

        if s3_configuration is not None:
            _logger.info(f'Setting up S3 import storage for Label Studio project: {title}')
            project.connect_s3_import_storage(**s3_configuration)

        project_id = project.get_params()['id']
        return LabelStudioProject(name, project_id, media_import_method, resolved_col_mapping)


@dataclass(frozen=True)
class _DataKey:
    name: Optional[str]  # The 'name' attribute of the data key; may differ from the field name
    column_type: ts.ColumnType


@dataclass(frozen=True)
class _RectangleLabel:
    to_name: str
    labels: list[str]


@dataclass(frozen=True)
class _LabelStudioConfig:
    data_keys: dict[str, _DataKey]
    rectangle_labels: dict[str, _RectangleLabel]

    def validate(self) -> None:
        data_key_names = {key.name for key in self.data_keys.values() if key.name is not None}
        for name, rl in self.rectangle_labels.items():
            if rl.to_name not in data_key_names:
                raise excs.Error(
                    f'Invalid Label Studio configuration: `toName` attribute of RectangleLabels `{name}` '
                    f'references an unknown data key: `{rl.to_name}`'
                )

    @property
    def export_columns(self) -> dict[str, ts.ColumnType]:
        data_key_cols = {key_id: key_info.column_type for key_id, key_info in self.data_keys.items()}
        rl_cols = {name: ts.JsonType() for name in self.rectangle_labels}
        return {**data_key_cols, **rl_cols}


ANNOTATIONS_COLUMN = 'annotations'
_PAGE_SIZE = 100  # This is the default used in the LS SDK
_LS_TAG_MAP = {
    'header': ts.StringType(),
    'text': ts.StringType(),
    'image': ts.ImageType(),
    'video': ts.VideoType(),
    'audio': ts.AudioType(),
}
