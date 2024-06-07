import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional
from xml.etree import ElementTree

import PIL.Image
import label_studio_sdk
import more_itertools
from requests.exceptions import HTTPError

import pixeltable as pxt
import pixeltable.env as env
import pixeltable.exceptions as excs
from pixeltable import Table
from pixeltable.datatransfer.remote import Remote
from pixeltable.utils import coco

_logger = logging.getLogger('pixeltable')


@env.register_client('label_studio')
def _(api_key: str, url: str) -> label_studio_sdk.Client:
    return label_studio_sdk.Client(api_key=api_key, url=url)


def _label_studio_client() -> label_studio_sdk.Client:
    return env.Env.get().get_client('label_studio')


class LabelStudioProject(Remote):
    """
    A [`Remote`][pixeltable.datatransfer.Remote] that represents a Label Studio project, providing functionality
    for synchronizing between a Pixeltable table and a Label Studio project.

    Typically, applications will call [`create`][pixeltable.datatransfer.label_studio.LabelStudioProject.create]`()`
    to create a new project, then
    [`Table.link`][pixeltable.Table.link]`()` to establish a link between a Pixeltable table and
    the new project.

    The API key and URL for a valid Label Studio server must be specified in Pixeltable config. Either:

    * Set the `LABEL_STUDIO_API_KEY` and `LABEL_STUDIO_URL` environment variables; or
    * Specify `api_key` and `url` fields in the `label-studio` section of `$PIXELTABLE_HOME/config.yaml`.
    """
    # TODO(aaron-siegel): Add link in docstring to a Label Studio howto

    def __init__(self, project_id: int):
        self.project_id = project_id
        self._project: Optional[label_studio_sdk.project.Project] = None

    @classmethod
    def create(cls, title: str, label_config: str, **kwargs: Any) -> 'LabelStudioProject':
        """
        Creates a new Label Studio project, using the Label Studio client configured in Pixeltable.

        Args:
            title: The title of the project.
            label_config: The Label Studio project configuration, in XML format.
            **kwargs: Additional keyword arguments for the new project; these will be passed to `start_project`
                in the Label Studio SDK.
        """
        # Check that the config is valid before creating the project
        cls._parse_project_config(label_config)
        project = _label_studio_client().start_project(title=title, label_config=label_config, **kwargs)
        project_id = project.get_params()['id']
        return LabelStudioProject(project_id)

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
    def _project_config(self) -> '_LabelStudioConfig':
        return self._parse_project_config(self.project_params['label_config'])

    def get_export_columns(self) -> dict[str, pxt.ColumnType]:
        """
        The data keys and preannotation fields specified in this Label Studio project.
        """
        return self._project_config.export_columns

    def get_import_columns(self) -> dict[str, pxt.ColumnType]:
        """
        Always contains a single entry:

        ```
        {"annotations": pxt.JsonType(nullable=True)}
        ```
        """
        return {ANNOTATIONS_COLUMN: pxt.JsonType(nullable=True)}

    def sync(self, t: Table, col_mapping: dict[str, str], export_data: bool, import_data: bool) -> None:
        _logger.info(f'Syncing Label Studio project "{self.project_title}" with table `{t.get_name()}`'
                     f' (export: {export_data}, import: {import_data}).')
        # Collect all existing tasks into a dict with entries `rowid: task`
        tasks = {tuple(task['meta']['rowid']): task for task in self._fetch_all_tasks()}
        if export_data:
            self._create_tasks_from_table(t, col_mapping, tasks)
        if import_data:
            self._update_table_from_tasks(t, col_mapping, tasks)

    def _fetch_all_tasks(self) -> Iterator[dict]:
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

    def _update_table_from_tasks(self, t: Table, col_mapping: dict[str, str], tasks: dict[tuple, dict]) -> None:
        # `col_mapping` is guaranteed to be a one-to-one dict whose values are a superset
        # of `get_import_columns`
        assert ANNOTATIONS_COLUMN in col_mapping.values()
        annotations_column = next(k for k, v in col_mapping.items() if v == ANNOTATIONS_COLUMN)
        updates = [
            {
                '_rowid': task['meta']['rowid'],
                # Replace [] by None to indicate no annotations. We do want to sync rows with no annotations,
                # in order to properly handle the scenario where existing annotations have been deleted in
                # Label Studio.
                annotations_column: task[ANNOTATIONS_COLUMN] if len(task[ANNOTATIONS_COLUMN]) > 0 else None
            }
            for task in tasks.values()
        ]
        if len(updates) > 0:
            _logger.info(
                f'Updating table `{t.get_name()}`, column `{annotations_column}` with {len(updates)} total annotations.'
            )
            t.batch_update(updates)
            annotations_count = sum(len(task[ANNOTATIONS_COLUMN]) for task in tasks.values())
            print(f'Synced {annotations_count} annotation(s) from {len(updates)} existing task(s) in {self}.')

    def _create_tasks_from_table(self, t: Table, col_mapping: dict[str, str], existing_tasks: dict[tuple, dict]) -> None:
        t_col_types = t.column_types()
        config = self._project_config

        # Columns in `t` that map to Label Studio data keys
        t_data_cols = [
            t_col_name for t_col_name, r_col_name in col_mapping.items()
            if r_col_name in config.data_keys
        ]

        # Columns in `t` that map to `rectanglelabels` preannotations
        t_rl_cols = [
            t_col_name for t_col_name, r_col_name in col_mapping.items()
            if r_col_name in config.rectangle_labels
        ]

        # Destinations for `rectanglelabels` preannotations
        rl_info = list(config.rectangle_labels.values())

        _logger.debug('`t_data_cols`: %s', t_data_cols)
        _logger.debug('`t_rl_cols`: %s', t_rl_cols)
        _logger.debug('`rl_info`: %s', rl_info)

        if len(t_data_cols) == 1 and t_col_types[t_data_cols[0]].is_media_type():
            # With a single media column, we can post local files to Label Studio using
            # the file transfer API.
            self._create_tasks_by_post(t, col_mapping, existing_tasks, t_rl_cols, rl_info, t_data_cols[0])
        else:
            # Either a single non-media column or multiple columns. Either way, we can't
            # use the file upload API and need to rely on externally accessible URLs for
            # media columns.
            self._create_tasks_by_urls(t, col_mapping, existing_tasks, t_data_cols, t_col_types, t_rl_cols, rl_info)

    def _create_tasks_by_post(
            self,
            t: Table,
            col_mapping: dict[str, str],
            existing_tasks: dict[tuple, dict],
            t_rl_cols: list[str],
            rl_info: list['_RectangleLabel'],
            media_col_name: str
    ) -> None:
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
                    self._coco_to_predictions(
                        coco_annotations[i], col_mapping[t_rl_cols[i]], rl_info[i], task_id=task_id
                    )
                    for i in range(len(coco_annotations))
                ]
                _logger.debug(f'`predictions`: %s', predictions)
                self.project.create_predictions(predictions)
                tasks_created += 1

        print(f'Created {tasks_created} new task(s) in {self}.')

        self._delete_stale_tasks(existing_tasks, row_ids_in_pxt, tasks_created)

    def _create_tasks_by_urls(
            self,
            t: Table,
            col_mapping: dict[str, str],
            existing_tasks: dict[tuple, dict],
            t_data_cols: list[str],
            t_col_types: dict[str, pxt.ColumnType],
            t_rl_cols: list[str],
            rl_info: list['_RectangleLabel']
    ):
        # TODO(aaron-siegel): This is just a placeholder (implementation is not complete or tested!)
        selection = [
            t[col_name].fileurl if t_col_types[col_name].is_media_type() else t[col_name]
            for col_name in t_data_cols
        ]
        r_data_cols = [col_mapping[col_name] for col_name in t_data_cols]
        rows = t.select(*selection, *[t[col] for col in t_rl_cols])
        new_rows = filter(lambda row: row.rowid not in existing_tasks, rows._exec())
        tasks_created = 0
        row_ids_in_pxt: set[tuple] = set()

        for page in more_itertools.batched(new_rows, n=_PAGE_SIZE):
            data_col_idxs = [expr.slot_idx for expr in rows._select_list_exprs[:len(t_data_cols)]]
            rl_col_idxs = [expr.slot_idx for expr in rows._select_list_exprs[len(t_data_cols):]]
            tasks = []

            for row in page:
                row_ids_in_pxt.add(row.rowid)
                data_vals = [row.vals[i] for i in data_col_idxs]
                coco_annotations = [row.vals[i] for i in rl_col_idxs]
                predictions = [
                    self._coco_to_predictions(coco_annotations[i], col_mapping[t_rl_cols[i]], rl_info[i])
                    for i in range(len(coco_annotations))
                ]

                # Validate media columns
                # TODO Support this if label studio is running on localhost?
                for i in range(len(data_vals)):
                    if t[t_data_cols[i]].col_type.is_media_type() and data_vals[i].startswith("file://"):
                        raise excs.Error(
                            'Cannot use locally stored media files in a `LabelStudioProject` with more than one '
                            'data key. (This is a limitation of Label Studio; see warning here: '
                            'https://labelstud.io/guide/tasks.html)'
                        )

                tasks.append({
                    'data': zip(r_data_cols, data_vals),
                    'meta': {'rowid': row.rowid},
                    'predictions': predictions
                })

            self.project.import_tasks(tasks)
            tasks_created += len(tasks)

        print(f'Created {tasks_created} new task(s) in {self}.')

        self._delete_stale_tasks(existing_tasks, row_ids_in_pxt, tasks_created)

    def _delete_stale_tasks(self, existing_tasks: dict[tuple, dict], row_ids_in_pxt: set[tuple], tasks_created: int):
        tasks_to_delete = [
            task['id'] for rowid, task in existing_tasks.items()
            if rowid not in row_ids_in_pxt
        ]
        # Sanity check the math
        assert len(tasks_to_delete) == len(existing_tasks) + tasks_created - len(row_ids_in_pxt)

        if len(tasks_to_delete) > 0:
            self.project.delete_tasks(tasks_to_delete)
            print(f'Deleted {len(tasks_to_delete)} tasks(s) in {self} that are no longer present in Pixeltable.')

    def to_dict(self) -> dict[str, Any]:
        return {'project_id': self.project_id}

    @classmethod
    def from_dict(cls, md: dict[str, Any]) -> 'LabelStudioProject':
        return LabelStudioProject(md['project_id'])

    def __repr__(self) -> str:
        name = self.project.get_params()['title']
        return f'LabelStudioProject `{name}`'

    @classmethod
    def _parse_project_config(cls, xml_config: str) -> '_LabelStudioConfig':
        """
        Parses a Label Studio XML config, extracting the names and Pixeltable types of
        all input variables.
        """
        root: ElementTree.Element = ElementTree.fromstring(xml_config)
        if root.tag.lower() != 'view':
            raise excs.Error('Root of Label Studio config must be a `View`')
        config = _LabelStudioConfig(
            data_keys=dict(cls._parse_data_keys_config(root)),
            rectangle_labels=dict(cls._parse_rectangle_labels_config(root))
        )
        config.validate()
        return config

    @classmethod
    def _parse_data_keys_config(cls, root: ElementTree.Element) -> Iterator[tuple[str, '_DataKey']]:
        for element in root:
            if 'value' in element.attrib and element.attrib['value'][0] == '$':
                remote_col_name = element.attrib['value'][1:]
                if 'name' not in element.attrib:
                    raise excs.Error(f'Data key is missing `name` attribute: `{remote_col_name}`')
                element_type = _LS_TAG_MAP.get(element.tag.lower())
                if element_type is None:
                    raise excs.Error(
                        f'Unsupported Label Studio data type: `{element.tag}` (in data key `{remote_col_name}`)'
                    )
                yield remote_col_name, _DataKey(element.attrib['name'], element_type)

    @classmethod
    def _parse_rectangle_labels_config(cls, root: ElementTree.Element) -> Iterator[tuple[str, '_RectangleLabel']]:
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
    def _coco_to_predictions(
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


@dataclass(frozen=True)
class _DataKey:
    name: str  # The 'name' attribute of the data key; may differ from the field name
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
        data_key_names = set(key.name for key in self.data_keys.values())
        for name, rl in self.rectangle_labels.items():
            if rl.to_name not in data_key_names:
                raise excs.Error(
                    f'Invalid Label Studio configuration: `toName` attribute of RectangleLabels `{name}` '
                    f'references an unknown data key: `{rl.to_name}`'
                )

    @property
    def export_columns(self) -> dict[str, pxt.ColumnType]:
        data_key_cols = {key_name: key_info.column_type for key_name, key_info in self.data_keys.items()}
        rl_cols = {name: pxt.JsonType() for name in self.rectangle_labels.keys()}
        return {**data_key_cols, **rl_cols}


ANNOTATIONS_COLUMN = 'annotations'
_PAGE_SIZE = 100  # This is the default used in the LS SDK
_LS_TAG_MAP = {
    'text': pxt.StringType(),
    'image': pxt.ImageType(),
    'video': pxt.VideoType(),
    'audio': pxt.AudioType()
}
