import json
from pathlib import Path
from typing import Any

import PIL

import pixeltable as pxt
import pixeltable.exceptions as excs

format_msg = """

Required format:
{
    'image': PIL.Image.Image,
    'annotations': [
        {
            'bbox': [x: int, y: int, w: int, h: int],
            'category': str | int,
        },
        ...
    ],
}
"""


def _verify_input_dict(input_dict: dict[str, Any]) -> None:
    """Verify that input_dict is a valid input dict for write_coco_dataset()"""
    if not isinstance(input_dict, dict):
        raise excs.Error(f'Expected dict, got {input_dict}{format_msg}')
    if 'image' not in input_dict:
        raise excs.Error(f'Missing key "image" in input dict: {input_dict}{format_msg}')
    if not isinstance(input_dict['image'], PIL.Image.Image):
        raise excs.Error(f'Value for "image" is not a PIL.Image.Image: {input_dict}{format_msg}')
    if 'annotations' not in input_dict:
        raise excs.Error(f'Missing key "annotations" in input dict: {input_dict}{format_msg}')
    if not isinstance(input_dict['annotations'], list):
        raise excs.Error(f'Value for "annotations" is not a list: {input_dict}{format_msg}')
    for annotation in input_dict['annotations']:
        if not isinstance(annotation, dict):
            raise excs.Error(f'Annotation is not a dict: {annotation}{format_msg}')
        if 'bbox' not in annotation:
            raise excs.Error(f'Missing key "bbox" in annotation: {annotation}{format_msg}')
        if not isinstance(annotation['bbox'], list):
            raise excs.Error(f'Value for "bbox" is not a list [x, y, w, h]: {annotation}{format_msg}')
        if len(annotation['bbox']) != 4 or not all(isinstance(x, int) for x in annotation['bbox']):
            raise excs.Error(f'Key "bbox" is not a list [x, y, w, h] of ints: {annotation}{format_msg}')
        if 'category' not in annotation:
            raise excs.Error(f'Missing key "category" in annotation: {annotation}{format_msg}')
        if not isinstance(annotation['category'], (str, int)):
            raise excs.Error(f'Value for "category" is not a str or int: {annotation}{format_msg}')


def write_coco_dataset(df: pxt.DataFrame, dest_path: Path) -> Path:
    """Export a DataFrame result set as a COCO dataset in dest_path and return the path of the data.json file."""
    # TODO: validate schema
    if len(df._select_list_exprs) != 1 or not df._select_list_exprs[0].col_type.is_json_type():
        raise excs.Error(f'Expected exactly one json-typed column in select list: {df._select_list_exprs}')
    input_dict_slot_idx = -1  # df._select_list_exprs[0].slot_idx isn't valid until _exec()

    # create output dir
    assert not dest_path.exists()
    dest_path.mkdir(parents=False)
    images_dir = dest_path / 'images'
    images_dir.mkdir()

    images: list[dict[str, Any]] = []
    img_id = -1
    annotations: list[dict[str, Any]] = []
    ann_id = -1
    categories: set[Any] = set()
    for input_row in df._exec():
        if input_dict_slot_idx == -1:
            input_dict_expr = df._select_list_exprs[0]
            input_dict_slot_idx = input_dict_expr.slot_idx
            input_dict = input_row[input_dict_slot_idx]
            _verify_input_dict(input_dict)

            # we want to know the slot idx of the image used in the input dict, so that we can check whether we
            # already have a local path for it
            input_dict_dependencies = input_dict_expr.dependencies()
            img_slot_idx = next((e.slot_idx for e in input_dict_dependencies if e.col_type.is_image_type()), None)
            assert img_slot_idx is not None
        else:
            input_dict = input_row[input_dict_slot_idx]
            _verify_input_dict(input_dict)

        # create image record
        img_id += 1

        # get a local path for the image
        img = input_dict['image']
        if input_row.file_paths[img_slot_idx] is not None:
            # we already have a local path
            img_path = Path(input_row.file_paths[img_slot_idx])
            # TODO: if the path leads to our tmp dir, we need to move the file
        else:
            # we need to create a local path
            img_path = images_dir / f'{img_id}.jpg'
            img.save(img_path)

        images.append({'id': img_id, 'file_name': str(img_path), 'width': img.width, 'height': img.height})

        # create annotation records for this image
        for annotation in input_dict['annotations']:
            ann_id += 1
            _, _, w, h = annotation['bbox']
            category = annotation['category']
            categories.add(category)
            annotations.append(
                {
                    'id': ann_id,
                    'image_id': img_id,
                    # we use the category name here and fix it up at the end, when we have assigned category ids
                    'category_id': category,
                    'bbox': annotation['bbox'],
                    'area': w * h,
                    'iscrowd': 0,
                }
            )

    # replace category names with ids
    category_ids = {category: id for id, category in enumerate(sorted(categories))}
    for annotation in annotations:
        annotation['category_id'] = category_ids[annotation['category_id']]

    result = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': id, 'name': category} for category, id in category_ids.items()],
    }
    output_path = dest_path / 'data.json'
    with open(output_path, 'w', encoding='utf-8') as fp:
        json.dump(result, fp)
    return output_path


COCO_2017_CATEGORIES = {
    0: 'N/A',
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    12: 'N/A',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    26: 'N/A',
    27: 'backpack',
    28: 'umbrella',
    29: 'N/A',
    30: 'N/A',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    45: 'N/A',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    66: 'N/A',
    67: 'dining table',
    68: 'N/A',
    69: 'N/A',
    70: 'toilet',
    71: 'N/A',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    83: 'N/A',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush',
}
