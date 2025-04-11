import logging
from typing import TYPE_CHECKING

import PIL.Image

import pixeltable as pxt
from pixeltable.func import Batch
from pixeltable.functions.util import normalize_image_mode
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    from yolox.models import Yolox, YoloxProcessor  # type: ignore[import-untyped]

_logger = logging.getLogger('pixeltable')


@pxt.udf(batch_size=4)
def yolox(images: Batch[PIL.Image.Image], *, model_id: str, threshold: float = 0.5) -> Batch[dict]:
    """
    Computes YOLOX object detections for the specified image. `model_id` should reference one of the models
    defined in the [YOLOX documentation](https://github.com/Megvii-BaseDetection/YOLOX).

    YOLOX is part of the `pixeltable.ext` package: long-term support in Pixeltable is not guaranteed.

    __Requirements__:

    - `pip install pixeltable-yolox`

    Args:
        model_id: one of: `yolox_nano`, `yolox_tiny`, `yolox_s`, `yolox_m`, `yolox_l`, `yolox_x`
        threshold: the threshold for object detection

    Returns:
        A dictionary containing the output of the object detection model.

    Examples:
        Add a computed column that applies the model `yolox_m` to an existing
        Pixeltable column `tbl.image` of the table `tbl`:

        >>> tbl.add_computed_column(detections=yolox(tbl.image, model_id='yolox_m', threshold=0.8))
    """
    import torch

    model = _lookup_model(model_id, 'cpu')
    processor = _lookup_processor(model_id)
    normalized_images = [normalize_image_mode(image) for image in images]
    with torch.no_grad():
        tensor = processor(normalized_images)
        output = model(tensor)
        return processor.postprocess(normalized_images, output, threshold=threshold)


@pxt.udf
def yolo_to_coco(detections: dict) -> list:
    """
    Converts the output of a YOLOX object detection model to COCO format.

    YOLOX is part of the `pixeltable.ext` package: long-term support in Pixeltable is not guaranteed.

    Args:
        detections: The output of a YOLOX object detection model, as returned by `yolox`.

    Returns:
        A dictionary containing the data from `detections`, converted to COCO format.

    Examples:
        Add a computed column that converts the output `tbl.detections` to COCO format, where `tbl.image`
        is the image for which detections were computed:

        >>> tbl.add_computed_column(detections=yolox(tbl.image, model_id='yolox_m', threshold=0.8))
        ... tbl.add_computed_column(detections_coco=yolo_to_coco(tbl.detections))
    """
    bboxes, labels = detections['bboxes'], detections['labels']
    num_annotations = len(detections['bboxes'])
    assert num_annotations == len(detections['labels'])
    result = []
    for i in range(num_annotations):
        bbox = bboxes[i]
        ann = {
            'bbox': [round(bbox[0]), round(bbox[1]), round(bbox[2] - bbox[0]), round(bbox[3] - bbox[1])],
            'category': labels[i],
        }
        result.append(ann)
    return result


def _lookup_model(model_id: str, device: str) -> 'Yolox':
    from yolox.models import Yolox

    key = (model_id, device)
    if key not in _model_cache:
        _model_cache[key] = Yolox.from_pretrained(model_id, device=device)

    return _model_cache[key]


def _lookup_processor(model_id: str) -> 'YoloxProcessor':
    from yolox.models import YoloxProcessor

    if model_id not in _processor_cache:
        _processor_cache[model_id] = YoloxProcessor(model_id)

    return _processor_cache[model_id]


_model_cache: dict[tuple[str, str], 'Yolox'] = {}
_processor_cache: dict[str, 'YoloxProcessor'] = {}


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
