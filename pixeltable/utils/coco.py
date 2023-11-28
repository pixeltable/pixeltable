from typing import List, Dict, Any, Set
from pathlib import Path
import json

import PIL

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

def _verify_input_dict(input_dict: Dict[str, Any]) -> None:
    """Verify that input_dict is a valid input dict for write_coco_dataset()"""
    if not isinstance(input_dict, dict):
        raise excs.Error(f'Expected dict, got {input_dict}{format_msg}')
    if 'image' not in input_dict:
        raise excs.Error(f'Missing key "image" in input dict: {input_dict}{format_msg}')
    if not isinstance(input_dict['image'], PIL.Image.Image):
        raise excs.Error(f'Value for "image" is not a PIL.Image.Image: {input_dict}{format_msg}')
    if  'annotations' not in input_dict:
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

def write_coco_dataset(df: 'pixeltable.DataFrame', dest_path: Path) -> Path:
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

    images: List[Dict[str, Any]] = []
    img_id = -1
    annotations: List[Dict[str, Any]] = []
    ann_id = -1
    categories: Set[Any] = set()
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

        images.append({
            'id': img_id,
            'file_name': str(img_path),
            'width': img.width,
            'height': img.height,
        })

        # create annotation records for this image
        for annotation in input_dict['annotations']:
            ann_id += 1
            x, y, w, h = annotation['bbox']
            category = annotation['category']
            categories.add(category)
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                # we use the category name here and fix it up at the end, when we have assigned category ids
                'category_id': category,
                'bbox': annotation['bbox'],
                'area': w * h,
                'iscrowd': 0,
            })

    # replace category names with ids
    category_ids = {category: id for id, category in enumerate(sorted(list(categories)))}
    for annotation in annotations:
        annotation['category_id'] = category_ids[annotation['category_id']]

    result = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': id, 'name': category} for category, id in category_ids.items()],
    }
    output_path = dest_path / 'data.json'
    with open(output_path, 'w') as f:
        json.dump(result, f)
    return output_path

