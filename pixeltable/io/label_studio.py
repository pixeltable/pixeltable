import itertools
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Literal
from xml.etree import ElementTree

import PIL.Image
import label_studio_sdk
from requests.exceptions import HTTPError

import pixeltable as pxt
import pixeltable.env as env
import pixeltable.exceptions as excs
from pixeltable import Table
from pixeltable.exprs import ColumnRef, DataRow
from pixeltable.io.external_store import Project, SyncStatus
from pixeltable.utils import coco

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

    def __init__(
            self,
            name: str,
            project_id: int,
            media_import_method: Literal['post', 'file', 'url'],
            col_mapping: Optional[dict[str, str]]
    ):
        """
        The constructor will NOT create a new Label Studio project; it is also used when loading
        metadata for existing projects.
        """
        self.project_id = project_id
        self.media_import_method = media_import_method
        self._project: Optional[label_studio_sdk.project.Project] = None
        super().__init__(name, col_mapping)

    @property
    def project(self) -> label_studio_sdk.project.Project:
        """The `Project` object corresponding to this Label Studio project."""
        if self._project is None:
            try:
                self._project = _label_studio_client().get_project(self.project_id)
            except HTTPError as exc:
                raise excs.Error(f'Could not locate Label Studio project: {self.project_id} '
                                 '(cannot connect to server or project no longer exists)') from exc
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

    def get_export_columns(self) -> dict[str, pxt.ColumnType]:
        """
        The data keys and preannotation fields specified in this Label Studio project.
        """
        return self.__project_config.export_columns

    def get_import_columns(self) -> dict[str, pxt.ColumnType]:
        """
        Always contains a single entry:

        ```
        {"annotations": pxt.JsonType(nullable=True)}
        ```
        """
        return {ANNOTATIONS_COLUMN: pxt.JsonType(nullable=True)}

    def sync(self, t: Table, export_data: bool, import_data: bool) -> 'SyncStatus':
        _logger.info(f'Syncing Label Studio project "{self.project_title}" with table `{t.get_name()}`'
                     f' (export: {export_data}, import: {import_data}).')
        # Collect all existing tasks into a dict with entries `rowid: task`
        tasks = {tuple(task['meta']['rowid']): task for task in self.__fetch_all_tasks()}
        sync_status = SyncStatus.empty()
        if export_data:
            export_sync_status = self.__update_tasks(t, tasks)
            sync_status = sync_status.combine(export_sync_status)
        if import_data:
            import_sync_status = self.__update_table_from_tasks(t, tasks)
            sync_status = sync_status.combine(import_sync_status)
        return sync_status

    def __fetch_all_tasks(self) -> Iterator[dict]:
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
                f'Skipped {unknown_task_count} unrecognized task(s) when syncing Label Studio project "{self.project_title}".'
            )

    def __update_tasks(self, t: Table, existing_tasks: dict[tuple, dict]) -> 'SyncStatus':
        config = self.__project_config

        # Columns in `t` that map to Label Studio data keys
        t_data_cols = [
            t_col_name for t_col_name, r_col_name in self.col_mapping.items()
            if r_col_name in config.data_keys
        ]

        if len(t_data_cols) == 0:
            return SyncStatus.empty()

        # Columns in `t` that map to `rectanglelabels` preannotations
        t_rl_cols = [
            t_col_name for t_col_name, r_col_name in self.col_mapping.items()
            if r_col_name in config.rectangle_labels
        ]

        # Destinations for `rectanglelabels` preannotations
        rl_info = list(config.rectangle_labels.values())

        _logger.debug('`t_data_cols`: %s', t_data_cols)
        _logger.debug('`t_rl_cols`: %s', t_rl_cols)
        _logger.debug('`rl_info`: %s', rl_info)

        if self.media_import_method == 'post':
            # Send media to Label Studio by HTTP post.
            return self.__update_tasks_by_post(t, existing_tasks, t_data_cols[0], t_rl_cols, rl_info)
        elif self.media_import_method == 'file' or self.media_import_method == 'url':
            # Send media to Label Studio by file reference (local file or URL).
            return self.__update_tasks_by_files(t, existing_tasks, t_data_cols, t_rl_cols, rl_info)
        else:
            assert False

    def __update_tasks_by_post(
            self,
            t: Table,
            existing_tasks: dict[tuple, dict],
            media_col_name: str,
            t_rl_cols: list[str],
            rl_info: list['_RectangleLabel']
    ) -> 'SyncStatus':
        is_stored = t[media_col_name].col.is_stored
        # If it's a stored column, we can use `localpath`
        localpath_col_opt = [t[media_col_name].localpath] if is_stored else []
        # Select the media column, rectanglelabels columns, and localpath (if appropriate)
        rows = t.select(t[media_col_name], *[t[col] for col in t_rl_cols], *localpath_col_opt)
        tasks_created = 0
        row_ids_in_pxt: set[tuple] = set()

        for row in rows._exec():
            media_col_idx = rows._select_list_exprs[0].slot_idx
            rl_col_idxs = [expr.slot_idx for expr in rows._select_list_exprs[1: 1 + len(t_rl_cols)]]
            row_ids_in_pxt.add(row.rowid)
            if row.rowid not in existing_tasks:
                # Upload the media file to Label Studio
                if is_stored:
                    # There is an existing localpath; use it!
                    localpath_col_idx = rows._select_list_exprs[-1].slot_idx
                    file = Path(row.vals[localpath_col_idx])
                    task_id: int = self.project.import_tasks(file)[0]
                else:
                    # No localpath; create a temp file and upload it
                    assert isinstance(row.vals[media_col_idx], PIL.Image.Image)
                    file = env.Env.get().create_tmp_path(extension='.png')
                    row.vals[media_col_idx].save(file, format='png')
                    task_id: int = self.project.import_tasks(file)[0]
                    os.remove(file)

                # Update the task with `rowid` metadata
                self.project.update_task(task_id, meta={'rowid': row.rowid})

                # Convert coco annotations to predictions
                coco_annotations = [row.vals[i] for i in rl_col_idxs]
                _logger.debug('`coco_annotations`: %s', coco_annotations)
                predictions = [
                    self.__coco_to_predictions(
                        coco_annotations[i], self.col_mapping[t_rl_cols[i]], rl_info[i], task_id=task_id
                    )
                    for i in range(len(coco_annotations))
                ]
                _logger.debug(f'`predictions`: %s', predictions)
                self.project.create_predictions(predictions)
                tasks_created += 1

        print(f'Created {tasks_created} new task(s) in {self}.')

        sync_status = SyncStatus(external_rows_created=tasks_created)

        deletion_sync_status = self.__delete_stale_tasks(existing_tasks, row_ids_in_pxt, tasks_created)

        return sync_status.combine(deletion_sync_status)

    def __update_tasks_by_files(
            self,
            t: Table,
            existing_tasks: dict[tuple, dict],
            t_data_cols: list[str],
            t_rl_cols: list[str],
            rl_info: list['_RectangleLabel']
    ) -> 'SyncStatus':
        r_data_cols = [self.col_mapping[col_name] for col_name in t_data_cols]
        col_refs = {}
        for col_name in t_data_cols:
            if self.media_import_method == 'url':
                col_refs[col_name] = t[col_name].fileurl
            else:
                assert self.media_import_method == 'file'
                if not t[col_name].col_type.is_media_type():
                    # Not a media column; query the data directly
                    col_refs[col_name] = t[col_name]
                elif t[col_name].col.stored_proxy:
                    # Media column that has a stored proxy; use it. We have to give it a name,
                    # since it's an anonymous column
                    col_refs[f'{col_name}_proxy'] = ColumnRef(t[col_name].col.stored_proxy).localpath
                else:
                    # Media column without a stored proxy; this means it's a stored computed column,
                    # and we can just use the localpath
                    col_refs[col_name] = t[col_name].localpath

        df = t.select(*[t[col] for col in t_rl_cols], **col_refs)
        rl_col_idxs: Optional[list[int]] = None  # We have to wait until we begin iterating to populate these
        data_col_idxs: Optional[list[int]] = None

        row_ids_in_pxt: set[tuple] = set()
        tasks_created = 0
        tasks_updated = 0
        page = []

        # Function that turns a `DataRow` into a `dict` for creating or updating a task in the
        # Label Studio SDK.
        def create_task_info(row: DataRow) -> dict:
            data_vals = [row.vals[idx] for idx in data_col_idxs]
            coco_annotations = [row.vals[idx] for idx in rl_col_idxs]
            for i in range(len(t_data_cols)):
                if t[t_data_cols[i]].col_type.is_media_type():
                    # Special handling for media columns
                    assert isinstance(data_vals[i], str)
                    if self.media_import_method == 'url':
                        data_vals[i] = self.__validate_fileurl(t_data_cols[i], data_vals[i])
                    else:
                        assert self.media_import_method == 'file'
                        data_vals[i] = self.__localpath_to_lspath(data_vals[i])
            predictions = [
                self.__coco_to_predictions(coco_annotations[i], self.col_mapping[t_rl_cols[i]], rl_info[i])
                for i in range(len(coco_annotations))
            ]
            return {
                'data': dict(zip(r_data_cols, data_vals)),
                'meta': {'rowid': row.rowid},
                'predictions': predictions
            }

        for row in df._exec():
            if rl_col_idxs is None:
                rl_col_idxs = [expr.slot_idx for expr in df._select_list_exprs[:len(t_rl_cols)]]
                data_col_idxs = [expr.slot_idx for expr in df._select_list_exprs[len(t_rl_cols):]]
            row_ids_in_pxt.add(row.rowid)
            # TODO(aaron-siegel) Implement update logic (the below logic is not correct)
            # if row.rowid in existing_tasks:
            #     # A task for this row already exists; see if it needs an update.
            #     # Get the v_min record from task metadata. Default to 0 if no v_min record is found
            #     old_v_min = int(existing_tasks[row.rowid]['meta'].get('v_min', 0))
            #     if row.v_min > old_v_min:
            #         _logger.debug(f'Updating task for rowid {row.rowid} ({row.v_min} > {old_v_min}).')
            #         task_info = create_task_info(row)
            #         self.project.update_task(existing_tasks[row.rowid]['id'], **task_info)
            #         tasks_updated += 1
            if row.rowid not in existing_tasks:
                # No task exists for this row; we need to create one.
                page.append(create_task_info(row))
                tasks_created += 1
                if len(page) == _PAGE_SIZE:
                    self.project.import_tasks(page)
                    page.clear()

        if len(page) > 0:
            self.project.import_tasks(page)

        print(f'Created {tasks_created} new task(s) and updated {tasks_updated} existing task(s) in {self}.')

        sync_status = SyncStatus(external_rows_created=tasks_created)

        deletion_sync_status = self.__delete_stale_tasks(existing_tasks, row_ids_in_pxt, tasks_created)

        return sync_status.combine(deletion_sync_status)

    @classmethod
    def __validate_fileurl(cls, col_name: str, url: str) -> Optional[str]:
        # Check that the URL is one that will be visible to Label Studio. If it isn't, log an info message
        # to help users debug the issue.
        if url.startswith('file://') or url.startswith('s3://'):
            _logger.info(
                'URL found in media column `col_name` will not render correctly in Label Studio, since '
                f'it is not an HTTP URL: {url}'
            )
        return url

    @classmethod
    def __localpath_to_lspath(cls, localpath: str) -> str:
        # Transform the local path into Label Studio's bespoke path format.
        relpath = Path(localpath).relative_to(env.Env.get().home)
        return f'/data/local-files/?d={str(relpath)}'

    def __delete_stale_tasks(self, existing_tasks: dict[tuple, dict], row_ids_in_pxt: set[tuple], tasks_created: int) -> 'SyncStatus':
        deleted_rowids = set(existing_tasks.keys()) - row_ids_in_pxt
        # Sanity check the math
        assert len(deleted_rowids) == len(existing_tasks) + tasks_created - len(row_ids_in_pxt)
        tasks_to_delete = [existing_tasks[rowid]['id'] for rowid in deleted_rowids]

        if len(tasks_to_delete) > 0:
            self.project.delete_tasks(tasks_to_delete)
            print(f'Deleted {len(tasks_to_delete)} tasks(s) in {self} that are no longer present in Pixeltable.')

        # Remove them from the `existing_tasks` dict so that future updates are applied correctly
        for rowid in deleted_rowids:
            del existing_tasks[rowid]

        return SyncStatus(external_rows_deleted=len(deleted_rowids))

    def __update_table_from_tasks(self, t: Table, tasks: dict[tuple, dict]) -> 'SyncStatus':
        if ANNOTATIONS_COLUMN not in self.col_mapping.values():
            return SyncStatus.empty()

        local_annotations_column = next(k for k, v in self.col_mapping.items() if v == ANNOTATIONS_COLUMN)
        annotations = {
            # Replace [] by None to indicate no annotations. We do want to sync rows with no annotations,
            # in order to properly handle the scenario where existing annotations have been deleted in
            # Label Studio.
            tuple(task['meta']['rowid']): task[ANNOTATIONS_COLUMN] if len(task[ANNOTATIONS_COLUMN]) > 0 else None
            for task in tasks.values()
        }

        # Prune the annotations down to just the ones that have actually changed.
        rows = t.select(t[local_annotations_column])
        for row in rows._exec():
            assert len(row.vals) == 1
            if row.rowid in annotations and annotations[row.rowid] == row.vals[0]:
                del annotations[row.rowid]

        # Apply updates
        updates = [{'_rowid': rowid, local_annotations_column: ann} for rowid, ann in annotations.items()]
        if len(updates) > 0:
            _logger.info(
                f'Updating table `{t.get_name()}`, column `{local_annotations_column}` with {len(updates)} total annotations.'
            )
            # batch_update currently doesn't propagate from views to base tables. As a workaround, we call
            # batch_update on the actual ancestor table that holds the annotations column.
            # TODO(aaron-siegel): Simplify this once propagation is properly implemented in batch_update
            ancestor = t
            while local_annotations_column not in ancestor.tbl_version_path.tbl_version.cols_by_name:
                assert ancestor.base is not None
                ancestor = ancestor.base
            ancestor.batch_update(updates)
            print(f'Updated annotation(s) from {len(updates)} task(s) in {self}.')

        return SyncStatus(pxt_rows_updated=len(updates))

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'project_id': self.project_id,
            'media_import_method': self.media_import_method,
            'col_mapping': self.col_mapping
        }

    @classmethod
    def from_dict(cls, md: dict[str, Any]) -> 'LabelStudioProject':
        return LabelStudioProject(md['name'], md['project_id'], md['media_import_method'], md['col_mapping'])

    def __repr__(self) -> str:
        name = self.project.get_params()['title']
        return f'LabelStudioProject `{name}`'

    @classmethod
    def __parse_project_config(cls, xml_config: str) -> '_LabelStudioConfig':
        """
        Parses a Label Studio XML config, extracting the names and Pixeltable types of
        all input variables.
        """
        root: ElementTree.Element = ElementTree.fromstring(xml_config)
        if root.tag.lower() != 'view':
            raise excs.Error('Root of Label Studio config must be a `View`')
        config = _LabelStudioConfig(
            data_keys=dict(cls.__parse_data_keys_config(root)),
            rectangle_labels=dict(cls.__parse_rectangle_labels_config(root))
        )
        config.validate()
        return config

    @classmethod
    def __parse_data_keys_config(cls, root: ElementTree.Element) -> Iterator[tuple[str, '_DataKey']]:
        for element in root:
            if 'value' in element.attrib and element.attrib['value'][0] == '$':
                external_col_name = element.attrib['value'][1:]
                data_key_name = element.attrib.get('name')
                element_type = _LS_TAG_MAP.get(element.tag.lower())
                if element_type is None:
                    raise excs.Error(
                        f'Unsupported Label Studio data type: `{element.tag}` (in data key `{external_col_name}`)'
                    )
                yield external_col_name, _DataKey(data_key_name, element_type)

    @classmethod
    def __parse_rectangle_labels_config(cls, root: ElementTree.Element) -> Iterator[tuple[str, '_RectangleLabel']]:
        for element in root:
            if element.tag.lower() == 'rectanglelabels':
                name = element.attrib['name']
                to_name = element.attrib['toName']
                labels = [
                    child.attrib['value']
                    for child in element if child.tag.lower() == 'label'
                ]
                for label in labels:
                    if label not in coco.COCO_2017_CATEGORIES.values():
                        raise excs.Error(f'Label in `rectanglelabels` config is not a valid COCO object name: {label}')
                yield name, _RectangleLabel(to_name=to_name, labels=labels)

    @classmethod
    def __coco_to_predictions(
            cls,
            coco_annotations: dict[str, Any],
            from_name: str,
            rl_info: '_RectangleLabel',
            task_id: Optional[int] = None
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
                    'rectanglelabels': [coco.COCO_2017_CATEGORIES[entry['category']]]
                }
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
        print(f'Deleted Label Studio project: {title}')

    def __eq__(self, other) -> bool:
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
            **kwargs: Any
    ) -> 'LabelStudioProject':
        """
        Creates a new Label Studio project, using the Label Studio client configured in Pixeltable.
        """
        # Check that the config is valid before creating the project
        config = cls.__parse_project_config(label_config)

        if name is None:
            # Create a default name that's unique to the table
            all_stores = t.list_external_stores()
            n = 0
            while f'ls_project_{n}' in all_stores:
                n += 1
            name = f'ls_project_{n}'

        if title is None:
            # `title` defaults to table name
            title = t.get_name()

        # Create a column to hold the annotations, if one does not yet exist
        if col_mapping is None or ANNOTATIONS_COLUMN in col_mapping.values():
            if col_mapping is None:
                local_annotations_column = ANNOTATIONS_COLUMN
            else:
                local_annotations_column = next(k for k, v in col_mapping.items() if v == ANNOTATIONS_COLUMN)
            if local_annotations_column not in t.column_names():
                t[local_annotations_column] = pxt.JsonType(nullable=True)

        cls.validate_column_names(t, config.export_columns, {ANNOTATIONS_COLUMN: pxt.JsonType(nullable=True)}, col_mapping)

        # Perform some additional validation
        if media_import_method == 'post' and len(config.data_keys) > 1:
            raise excs.Error('`media_import_method` cannot be `post` if there is more than one data key')

        project = _label_studio_client().start_project(title=title, label_config=label_config, **kwargs)

        if media_import_method == 'file':
            # We need to set up a local storage connection to receive media files
            os.environ['LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT'] = str(env.Env.get().home)
            try:
                project.connect_local_import_storage(local_store_path=str(env.Env.get().media_dir))
            except HTTPError as exc:
                if exc.errno == 400:
                    response: dict = json.loads(exc.response.text)
                    if 'validation_errors' in response and 'non_field_errors' in response['validation_errors'] \
                            and 'LOCAL_FILES_SERVING_ENABLED' in response['validation_errors']['non_field_errors'][0]:
                        raise excs.Error(
                            '`media_import_method` is set to `file`, but your Label Studio server is not configured '
                            'for local file storage.\nPlease set the `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED` '
                            'environment variable to `true` in the environment where your Label Studio server is running.'
                        ) from exc
                raise  # Handle any other exception type normally

        if col_mapping is None:
            col_mapping = {col: col for col in itertools.chain(config.export_columns.keys(), [ANNOTATIONS_COLUMN])}

        project_id = project.get_params()['id']
        return LabelStudioProject(name, project_id, media_import_method, col_mapping)


@dataclass(frozen=True)
class _DataKey:
    name: Optional[str]  # The 'name' attribute of the data key; may differ from the field name
    column_type: pxt.ColumnType


@dataclass(frozen=True)
class _RectangleLabel:
    to_name: str
    labels: list[str]


@dataclass(frozen=True)
class _LabelStudioConfig:
    data_keys: dict[str, _DataKey]
    rectangle_labels: dict[str, _RectangleLabel]

    def validate(self) -> None:
        data_key_names = set(key.name for key in self.data_keys.values() if key is not None)
        for name, rl in self.rectangle_labels.items():
            if rl.to_name not in data_key_names:
                raise excs.Error(
                    f'Invalid Label Studio configuration: `toName` attribute of RectangleLabels `{name}` '
                    f'references an unknown data key: `{rl.to_name}`'
                )

    @property
    def export_columns(self) -> dict[str, pxt.ColumnType]:
        data_key_cols = {key_id: key_info.column_type for key_id, key_info in self.data_keys.items()}
        rl_cols = {name: pxt.JsonType() for name in self.rectangle_labels.keys()}
        return {**data_key_cols, **rl_cols}


ANNOTATIONS_COLUMN = 'annotations'
_PAGE_SIZE = 100  # This is the default used in the LS SDK
_LS_TAG_MAP = {
    'header': pxt.StringType(),
    'text': pxt.StringType(),
    'image': pxt.ImageType(),
    'video': pxt.VideoType(),
    'audio': pxt.AudioType()
}
