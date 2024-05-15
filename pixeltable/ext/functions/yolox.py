import logging
from pathlib import Path
from typing import Iterable, Iterator
from urllib.request import urlretrieve

import PIL.Image
import numpy as np
import torch
from yolox.data import ValTransform
from yolox.exp import get_exp, Exp
from yolox.models import YOLOX
from yolox.utils import postprocess

import pixeltable as pxt
from pixeltable import env
from pixeltable.func import Batch
from pixeltable.functions.util import resolve_torch_device

_logger = logging.getLogger('pixeltable')


@pxt.udf(batch_size=4)
def yolox(images: Batch[PIL.Image.Image], *, model_id: str, threshold: float = 0.5) -> Batch[dict]:
    """
    Runs the specified YOLOX object detection model on an image.

    YOLOX support is part of the `pixeltable.ext` package: long-term support is not guaranteed, and it is not
    intended for use in production applications.

    Parameters:
    - `model_id` - one of: `yolox_nano, `yolox_tiny`, `yolox_s`, `yolox_m`, `yolox_l`, `yolox_x`
    - `threshold` - the threshold for object detection
    """
    model, exp = _lookup_model(model_id, 'cpu')
    image_tensors = list(_images_to_tensors(images, exp))
    batch_tensor = torch.stack(image_tensors)

    with torch.no_grad():
        output_tensor = model(batch_tensor)

    outputs = postprocess(
        output_tensor, 80, threshold, exp.nmsthre, class_agnostic=False
    )

    results: list[dict] = []
    for image in images:
        ratio = min(exp.test_size[0] / image.height, exp.test_size[1] / image.width)
        if outputs[0] is None:
            results.append({'bboxes': [], 'scores': [], 'labels': []})
        else:
            results.append({
                'bboxes': [(output[:4] / ratio).tolist() for output in outputs[0]],
                'scores': [output[4].item() * output[5].item() for output in outputs[0]],
                'labels': [int(output[6]) for output in outputs[0]]
            })
    return results


def _images_to_tensors(images: Iterable[PIL.Image.Image], exp: Exp) -> Iterator[torch.Tensor]:
    for image in images:
        image_transform, _ = _val_transform(np.array(image), None, exp.test_size)
        yield torch.from_numpy(image_transform)


def _lookup_model(model_id: str, device: str) -> (YOLOX, Exp):
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


_model_cache = {}
_val_transform = ValTransform(legacy=False)
