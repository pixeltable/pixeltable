import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator
from urllib.request import urlretrieve

import numpy as np
import PIL.Image

import pixeltable as pxt
from pixeltable import env
from pixeltable.func import Batch
from pixeltable.functions.util import normalize_image_mode
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    import torch
    from yolox.exp import Exp  # type: ignore[import-untyped]
    from yolox.models import YOLOX  # type: ignore[import-untyped]

_logger = logging.getLogger('pixeltable')


@pxt.udf(batch_size=4)
def yolox(images: Batch[PIL.Image.Image], *, model_id: str, threshold: float = 0.5) -> Batch[dict]:
    """
    Computes YOLOX object detections for the specified image. `model_id` should reference one of the models
    defined in the [YOLOX documentation](https://github.com/Megvii-BaseDetection/YOLOX).

    YOLOX is part of the `pixeltable.ext` package: long-term support in Pixeltable is not guaranteed.

    __Requirements__:

    - `pip install git+https://github.com/Megvii-BaseDetection/YOLOX`

    Args:
        model_id: one of: `yolox_nano`, `yolox_tiny`, `yolox_s`, `yolox_m`, `yolox_l`, `yolox_x`
        threshold: the threshold for object detection

    Returns:
        A dictionary containing the output of the object detection model.

    Examples:
        Add a computed column that applies the model `yolox_m` to an existing
        Pixeltable column `tbl.image` of the table `tbl`:

        >>> tbl['detections'] = yolox(tbl.image, model_id='yolox_m', threshold=0.8)
    """
    import torch
    from yolox.utils import postprocess  # type: ignore[import-untyped]

    model, exp = _lookup_model(model_id, 'cpu')
    image_tensors = list(_images_to_tensors(images, exp))
    batch_tensor = torch.stack(image_tensors)

    with torch.no_grad():
        output_tensor = model(batch_tensor)

    outputs = postprocess(output_tensor, 80, threshold, exp.nmsthre, class_agnostic=False)

    results: list[dict] = []
    for image in images:
        ratio = min(exp.test_size[0] / image.height, exp.test_size[1] / image.width)
        if outputs[0] is None:
            results.append({'bboxes': [], 'scores': [], 'labels': []})
        else:
            results.append(
                {
                    'bboxes': [(output[:4] / ratio).tolist() for output in outputs[0]],
                    'scores': [output[4].item() * output[5].item() for output in outputs[0]],
                    'labels': [int(output[6]) for output in outputs[0]],
                }
            )
    return results


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

        >>> tbl['detections'] = yolox(tbl.image, model_id='yolox_m', threshold=0.8)
        ... tbl['detections_coco'] = yolo_to_coco(tbl.detections)
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


def _images_to_tensors(images: Iterable[PIL.Image.Image], exp: 'Exp') -> Iterator['torch.Tensor']:
    import torch
    from yolox.data import ValTransform  # type: ignore[import-untyped]

    _val_transform = ValTransform(legacy=False)
    for image in images:
        image = normalize_image_mode(image)
        image_transform, _ = _val_transform(np.array(image), None, exp.test_size)
        yield torch.from_numpy(image_transform)


def _lookup_model(model_id: str, device: str) -> tuple['YOLOX', 'Exp']:
    import torch
    from yolox.exp import get_exp

    key = (model_id, device)
    if key in _model_cache:
        return _model_cache[key]

    weights_url = f'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{model_id}.pth'
    weights_file = Path(f'{env.Env.get().tmp_dir}/{model_id}.pth')
    if not weights_file.exists():
        _logger.info(f'Downloading weights for YOLOX model {model_id}: from {weights_url} -> {weights_file}')
        urlretrieve(weights_url, weights_file)

    exp = get_exp(exp_name=model_id)
    model = exp.get_model().to(device)

    model.eval()
    model.head.training = False
    model.training = False

    # Load in the weights from training
    weights = torch.load(weights_file, map_location=torch.device(device))
    model.load_state_dict(weights['model'])

    _model_cache[key] = (model, exp)
    return model, exp


_model_cache: dict[tuple[str, str], tuple['YOLOX', 'Exp']] = {}


__all__ = local_public_names(__name__)


def __dir__():
    return __all__
