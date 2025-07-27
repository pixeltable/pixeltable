import json
import os
from pathlib import Path
from typing import Any, Optional, Union

import PIL.Image

import pixeltable as pxt
import pixeltable.exceptions as excs


def import_coco(
    tbl_name: str,
    coco_json_path: Union[str, os.PathLike[str]],
    images_dir: Union[str, os.PathLike[str]],
    *,
    schema_overrides: Optional[dict[str, Any]] = None,
    primary_key: Optional[Union[str, list[str]]] = None,
    num_retained_versions: int = 10,
    comment: str = '',
) -> Any:
    """
    Creates a new base table from a COCO dataset annotation file and images directory.

    This function imports data from the standard COCO (Common Objects in Context) annotation format,
    which is a widely adopted JSON-based standard for storing and sharing image and video annotations,
    particularly for object detection, instance segmentation, and human pose estimation tasks.

    The COCO format consists of:
    - A JSON annotation file containing image metadata and annotations
    - A directory containing the actual image files

    The resulting table will have the following columns:
    - `image`: PIL.Image containing the loaded image
    - `annotations`: JSON containing the annotations in Pixeltable's expected format
    - `image_id`: Integer ID of the image from the COCO dataset
    - `file_name`: String filename of the image
    - `width`: Integer width of the image
    - `height`: Integer height of the image

    Args:
        tbl_name: The name of the table to create.
        coco_json_path: Path to the COCO annotation JSON file.
        images_dir: Path to the directory containing the image files.
        schema_overrides: If specified, then for each (name, type) pair in `schema_overrides`,
            the column with name `name` will be given type `type`, instead of being inferred.
        primary_key: The primary key of the table (see [`create_table()`][pixeltable.create_table]).
        num_retained_versions: The number of retained versions of the table.
        comment: A comment to attach to the table.

    Returns:
        A handle to the newly created Table.

    Raises:
        Error: If the COCO JSON file is malformed or images cannot be loaded.

    Example:
        Create a table from a COCO dataset:

        >>> tbl = pxt.io.import_coco(
        ...     'my_dataset',
        ...     'annotations/instances_train2017.json',
        ...     'images/train2017/'
        ... )

        The resulting table can be used with other Pixeltable COCO utilities:

        >>> # Export as COCO dataset
        >>> query = tbl.select({'image': tbl.image, 'annotations': tbl.annotations})
        >>> path = query.to_coco_dataset()
    """
    coco_json_path = Path(coco_json_path)
    images_dir = Path(images_dir)

    # Validate inputs
    if not coco_json_path.exists():
        raise excs.Error(f'COCO JSON file not found: {coco_json_path}')
    if not images_dir.exists():
        raise excs.Error(f'Images directory not found: {images_dir}')
    if not images_dir.is_dir():
        raise excs.Error(f'Images path is not a directory: {images_dir}')

    # Load and parse COCO JSON
    try:
        with open(coco_json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
    except json.JSONDecodeError as e:
        raise excs.Error(f'Invalid JSON in COCO file: {e}') from e
    except Exception as e:
        raise excs.Error(f'Error reading COCO file: {e}') from e

    # Validate COCO format
    _validate_coco_format(coco_data)

    # Convert COCO data to Pixeltable format
    table_data = _convert_coco_to_pixeltable_format(coco_data, images_dir)

    # Create the table
    return pxt.create_table(
        tbl_name,
        source=table_data,
        schema_overrides=schema_overrides,
        primary_key=primary_key,
        num_retained_versions=num_retained_versions,
        comment=comment,
    )


def _validate_coco_format(coco_data: dict[str, Any]) -> None:
    """Validate that the loaded JSON follows COCO format."""
    required_fields = ['images', 'annotations', 'categories']
    for field in required_fields:
        if field not in coco_data:
            raise excs.Error(f'Missing required field "{field}" in COCO JSON')
        if not isinstance(coco_data[field], list):
            raise excs.Error(f'Field "{field}" must be a list in COCO JSON')

    # Validate images structure
    for i, image in enumerate(coco_data['images']):
        if not isinstance(image, dict):
            raise excs.Error(f'Image at index {i} is not a dictionary')
        required_image_fields = ['id', 'file_name', 'width', 'height']
        for field in required_image_fields:
            if field not in image:
                raise excs.Error(f'Missing required field "{field}" in image at index {i}')

    # Validate categories structure
    for i, category in enumerate(coco_data['categories']):
        if not isinstance(category, dict):
            raise excs.Error(f'Category at index {i} is not a dictionary')
        required_category_fields = ['id', 'name']
        for field in required_category_fields:
            if field not in category:
                raise excs.Error(f'Missing required field "{field}" in category at index {i}')

    # Validate annotations structure
    for i, annotation in enumerate(coco_data['annotations']):
        if not isinstance(annotation, dict):
            raise excs.Error(f'Annotation at index {i} is not a dictionary')
        required_annotation_fields = ['id', 'image_id', 'category_id', 'bbox']
        for field in required_annotation_fields:
            if field not in annotation:
                raise excs.Error(f'Missing required field "{field}" in annotation at index {i}')

        # Validate bbox format
        bbox = annotation['bbox']
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise excs.Error(f'Invalid bbox format in annotation at index {i}. Expected [x, y, width, height]')


def _convert_coco_to_pixeltable_format(coco_data: dict[str, Any], images_dir: Path) -> list[dict[str, Any]]:
    """Convert COCO data to Pixeltable table format."""

    # Create mappings
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Group annotations by image_id
    image_annotations: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(annotation)

    table_rows = []

    for image_id, image_info in image_id_to_info.items():
        file_name = image_info['file_name']
        image_path = images_dir / file_name

        # Try to load the image
        try:
            if not image_path.exists():
                raise excs.Error(f'Image file not found: {image_path}')
            with PIL.Image.open(image_path) as pil_image:
                # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
                if pil_image.mode != 'RGB':
                    image = pil_image.convert('RGB')
                else:
                    image = pil_image.copy()
        except Exception as e:
            raise excs.Error(f'Error loading image {file_name}: {e}') from e

        # Convert annotations to Pixeltable format
        pxt_annotations = []
        if image_id in image_annotations:
            for annotation in image_annotations[image_id]:
                bbox = annotation['bbox']
                category_id = annotation['category_id']

                # Convert COCO bbox format [x, y, width, height] to Pixeltable format [x, y, width, height]
                # Note: COCO bbox is already in the format Pixeltable expects
                pxt_annotation = {
                    'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    'category': category_id_to_name.get(category_id, f'unknown_{category_id}'),
                }

                # Add optional fields if present
                if 'area' in annotation:
                    pxt_annotation['area'] = annotation['area']
                if 'iscrowd' in annotation:
                    pxt_annotation['iscrowd'] = annotation['iscrowd']

                pxt_annotations.append(pxt_annotation)

        # Create the row in the format expected by Pixeltable
        row = {
            'image': image,
            'annotations': pxt_annotations,
            'image_id': image_id,
            'file_name': file_name,
            'width': image_info['width'],
            'height': image_info['height'],
        }

        # Add optional image fields if present
        if 'date_captured' in image_info:
            row['date_captured'] = image_info['date_captured']
        if 'license' in image_info:
            row['license'] = image_info['license']

        table_rows.append(row)

    return table_rows
