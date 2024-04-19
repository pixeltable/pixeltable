from pathlib import Path
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
from pixeltable.functions.util import resolve_torch_device


@pxt.udf
def yolox(image: PIL.Image.Image, *, model_id: str, threshold: float = 0.5, device: str = 'cpu') -> dict:
    device = resolve_torch_device(device)
    model, exp = __lookup_model(model_id, device)
    image_transform, _ = __val_transform(np.array(image), None, exp.test_size)
    image_tensor = torch.from_numpy(image_transform).unsqueeze(0).float()
    image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        outputs = postprocess(
            outputs, 80, threshold, exp.nmsthre, class_agnostic=False
        )

    ratio = min(exp.test_size[0] / image.height, exp.test_size[1] / image.width)
    return {
        'boxes': [(output[:4] / ratio).tolist() for output in outputs[0]],
        'scores': [output[4].item() * output[5].item() for output in outputs[0]],
        'labels': [int(output[6]) for output in outputs[0]]
    }


def __lookup_model(model_id: str, device: str) -> (YOLOX, Exp):
    key = (model_id, device)
    if key in __model_cache:
        return __model_cache[key]

    weights_url = f'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{model_id}.pth'
    weights_file = Path(f'{env.Env.get().tmp_dir}/{model_id}.pth')
    if not weights_file.exists():
        print(f'Downloading weights for YOLOX model {model_id}')
        print(f'Downloading from {weights_url} to {weights_file}')
        urlretrieve(weights_url, weights_file)

    exp = get_exp(exp_name=model_id)
    model = exp.get_model()

    model.eval()
    model.head.training = False
    model.training = False

    # Load in the weights from training
    weights = torch.load(weights_file, map_location=torch.device(device))
    model.load_state_dict(weights['model'])

    __model_cache[key] = (model, exp)
    return model, exp


__model_cache = {}
__val_transform = ValTransform(legacy=False)
