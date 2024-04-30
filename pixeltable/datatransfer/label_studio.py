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


class LabelStudioProject(Remote):

    ANNOTATIONS_COLUMN = 'annotations'

    def __init__(self, project_id: int):
        self.project_id = project_id
        self.ls_client = env.Env.get().label_studio_client
        self._project: Optional[label_studio_sdk.project.Project] = None

    @property
    def project(self) -> label_studio_sdk.project.Project:
        if self._project is None:
            try:
                self._project = self.ls_client.get_project(self.project_id)
            except HTTPError as exc:
                raise excs.Error(f'Could not locate Label Studio project: {self.project_id} '
                                 '(cannot connect to server or project no longer exists)') from exc
        return self._project

    @property
    def project_params(self) -> dict[str, Any]:
        return self.project.get_params()

    @property
    def project_title(self) -> str:
        return self.project_params['title']

    @property
    def _project_config(self) -> '_LabelStudioConfig':
        return self._parse_project_config(self.project_params['label_config'])

    @classmethod
    def create(cls, title: str, label_config: str, **kwargs) -> 'LabelStudioProject':
        # Check that the config is valid before creating the project
        cls._parse_project_config(label_config)
        ls_client = env.Env.get().label_studio_client
        project = ls_client.start_project(title=title, label_config=label_config, **kwargs)
        project_id = project.get_params()['id']
        return LabelStudioProject(project_id)

    def get_push_columns(self) -> dict[str, pxt.ColumnType]:
        return self._project_config.push_columns

    def get_pull_columns(self) -> dict[str, pxt.ColumnType]:
        return {self.ANNOTATIONS_COLUMN: pxt.JsonType(nullable=True)}

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

    def _update_table_from_tasks(self, t: Table, col_mapping: dict[str, str], tasks: list[dict]) -> None:
        # `col_mapping` is guaranteed to be a one-to-one dict whose values are a superset
        # of `get_pull_columns`
        assert self.ANNOTATIONS_COLUMN in col_mapping.values()
        annotations_column = next(k for k, v in col_mapping.items() if v == self.ANNOTATIONS_COLUMN)
        updates = [
            {
                '_rowid': task['meta']['rowid'],
                # Replace [] by None to indicate no annotations. We do want to sync rows with no annotations,
                # in order to properly handle the scenario where existing annotations have been deleted in
                # Label Studio.
                annotations_column: task[self.ANNOTATIONS_COLUMN] if len(task[self.ANNOTATIONS_COLUMN]) > 0 else None
            }
            for task in tasks
        ]
        if len(updates) > 0:
            _logger.info(
                f'Updating table `{t.get_name()}`, column `{annotations_column}` with {len(updates)} total annotations.'
            )
            t.batch_update(updates)
            annotations_count = sum(len(task[self.ANNOTATIONS_COLUMN]) for task in tasks)
            print(f'Synced {annotations_count} annotation(s) from {len(updates)} existing task(s) in {self}.')

    def _create_tasks_from_table(self, t: Table, col_mapping: dict[str, str], existing_tasks: list[dict]) -> None:
        row_ids_in_ls = {tuple(task['meta']['rowid']) for task in existing_tasks}
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
        # Destinations for `rectanglelabels` annotations
        rl_to_names = [rl.to_name for rl in config.rectangle_labels.values()]

        _logger.debug('`t_data_cols`: %s', t_data_cols)
        _logger.debug('`t_rl_cols`: %s', t_rl_cols)
        _logger.debug('`rl_to_names`: %s', rl_to_names)

        # TODO Validate that `rl_to_names` are in the config!

        if len(t_data_cols) == 1 and t_col_types[t_data_cols[0]].is_media_type():
            # With a single media column, we can post local files to Label Studio using
            # the file transfer API.
            self._create_tasks_by_post(t, col_mapping, row_ids_in_ls, t_rl_cols, rl_to_names, t_data_cols[0])
        else:
            # Either a single non-media column or multiple columns. Either way, we can't
            # use the file upload API and need to rely on externally accessible URLs for
            # media columns.
            self._create_tasks_by_urls(t, col_mapping, row_ids_in_ls, t_data_cols, t_col_types, t_rl_cols, rl_to_names)

    def _create_tasks_by_post(
            self,
            t: Table,
            col_mapping: dict[str, str],
            row_ids_in_ls: set[tuple],
            t_rl_cols: list[str],
            rl_to_names: list[str],
            media_col_name: str
    ):
        is_stored = t[media_col_name].col.is_stored
        # If it's a stored column, we can use `localpath`
        localpath_col_opt = [t[media_col_name].localpath] if is_stored else []
        # Select the media column, rectanglelabels columns, and localpath (if appropriate)
        rows = t.select(t[media_col_name], *[t[col] for col in t_rl_cols], *localpath_col_opt)
        tasks_created = 0

        for row in rows._exec():
            media_col_idx = rows._select_list_exprs[0].slot_idx
            rl_col_idxs = [expr.slot_idx for expr in rows._select_list_exprs[1: 1 + len(t_rl_cols)]]
            if row.rowid not in row_ids_in_ls:
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
                        coco_annotations[i], col_mapping[t_rl_cols[i]], rl_to_names[i], task_id=task_id
                    )
                    for i in range(len(coco_annotations))
                ]
                _logger.debug(f'`predictions`: %s', predictions)
                self.project.create_predictions(predictions)
                tasks_created += 1

        print(f'Created {tasks_created} new task(s) in {self}.')

    def _create_tasks_by_urls(
            self,
            t: Table,
            col_mapping: dict[str, str],
            row_ids_in_ls: set[tuple],
            t_data_cols: list[str],
            t_col_types: dict[str, pxt.ColumnType],
            t_rl_cols: list[str],
            rl_to_names: list[str]
    ):
        selection = [
            t[col_name].fileurl if t_col_types[col_name].is_media_type() else t[col_name]
            for col_name in t_data_cols
        ]
        r_data_cols = [col_mapping[col_name] for col_name in t_data_cols]
        rows = t.select(*selection, *[t[col] for col in t_rl_cols])
        new_rows = filter(lambda row: row.rowid not in row_ids_in_ls, rows._exec())
        tasks_created = 0

        for page in more_itertools.batched(new_rows, n=_PAGE_SIZE):
            data_col_idxs = [expr.slot_idx for expr in rows._select_list_exprs[:len(t_data_cols)]]
            rl_col_idxs = [expr.slot_idx for expr in rows._select_list_exprs[len(t_data_cols):]]
            tasks = []

            for row in page:
                data_vals = [row.vals[i] for i in data_col_idxs]
                coco_annotations = [row.vals[i] for i in rl_col_idxs]
                predictions = [
                    self._coco_to_predictions(coco_annotations[i], col_mapping[t_rl_cols[i]], rl_to_names[i])
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
        # xml_config: str = self.project.get_params()['label_config']
        root: ElementTree.Element = ElementTree.fromstring(xml_config)
        if root.tag.lower() != 'view':
            raise excs.Error('Root of Label Studio config must be a `View`')
        return _LabelStudioConfig(
            data_keys=dict(cls._parse_data_keys_config(root)),
            rectangle_labels=dict(cls._parse_rectangle_labels_config(root))
        )

    @classmethod
    def _parse_data_keys_config(cls, root: ElementTree.Element) -> Iterator[tuple[str, pxt.ColumnType]]:
        for element in root:
            if 'value' in element.attrib and element.attrib['value'][0] == '$':
                element_type = _LS_TAG_MAP.get(element.tag.lower())
                if element_type is None:
                    raise excs.Error(f'Unsupported Label Studio data type: `{element.tag}`')
                yield element.attrib['value'][1:], element_type

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
            to_name: str,
            task_id: Optional[int] = None
    ) -> dict[str, Any]:
        width = coco_annotations['image']['width']
        height = coco_annotations['image']['height']
        result = [
            {
                'id': f'result_{i}',
                'type': 'rectanglelabels',
                'from_name': from_name,
                'to_name': to_name,
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
        ]
        if task_id is not None:
            return {'task': task_id, 'result': result}
        else:
            return {'result': result}


@pxt.udf
def detr_to_rectangle_labels(image: PIL.Image.Image, detr_info: dict[str, Any]) -> dict[str, Any]:
    bboxes = detr_info['boxes']
    scores = detr_info['scores']
    labels = detr_info['labels']
    result = [
        {
            'id': f'result{i}',
            'type': 'rectanglelabels',
            'image_rotation': 0,
            'original_width': image.width,
            'original_height': image.height,
            'value': {
                'rotation': 0,
                # Label Studio expects image coordinates as % of image dimensions
                'x': bboxes[i][0] * 100.0 / image.width,
                'y': bboxes[i][1] * 100.0 / image.height,
                'width': (bboxes[i][2] - bboxes[i][0]) * 100.0 / image.width,
                'height': (bboxes[i][3] - bboxes[i][1]) * 100.0 / image.height,
                'rectanglelabels': [coco.COCO_2017_CATEGORIES[labels[i]]]
            }
        }
        for i in range(len(bboxes))
    ]
    return {
        'score': min(scores) if len(scores) > 0 else 0.0,
        'result': result
    }


@dataclass(frozen=True)
class _RectangleLabel:
    to_name: str
    labels: list[str]


@dataclass(frozen=True)
class _LabelStudioConfig:
    data_keys: dict[str, pxt.ColumnType]
    rectangle_labels: dict[str, _RectangleLabel]

    @property
    def push_columns(self) -> dict[str, pxt.ColumnType]:
        return {**self.data_keys, **{name: pxt.JsonType() for name in self.rectangle_labels.keys()}}


_PAGE_SIZE = 100  # This is the default used in the LS SDK
_LS_TAG_MAP = {
    'text': pxt.StringType(),
    'image': pxt.ImageType(),
    'video': pxt.VideoType(),
    'audio': pxt.AudioType()
}
