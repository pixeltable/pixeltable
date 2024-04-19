from pathlib import Path
from urllib.request import urlretrieve

import PIL.Image
import numpy as np
import torch
from yolox.data import ValTransform
from yolox.exp import get_exp, Exp
from yolox.models import YOLOX
from yolox.utils import postprocess

from pixeltable import env


def yolox(image: PIL.Image.Image, *, model_id: str, device: str = 'cpu') -> dict:
    model, exp = __lookup_model(model_id)

    image_transform, _ = __val_transform(np.array(image), None, exp.test_size)
    image_tensor = torch.from_numpy(image_transform).unsqueeze(0).float()
    image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        outputs = postprocess(
            outputs, 1, exp.test_conf, exp.nmsthre, class_agnostic=True
        )
    return outputs


def __lookup_model(model_id: str) -> (YOLOX, Exp):
    if model_id in __model_cache:
        return __model_cache[model_id]

    weights_url = f'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{model_id}.pth'
    weights_file = Path(f'./{model_id}.pth')
    if not weights_file.exists():
        print(f'Downloading weights for YOLOX model {model_id}')
        print(f'Downloading from {weights_url} to {weights_file}')
        urlretrieve(weights_url, weights_file)

    exp = get_exp(exp_name=model_id)
    model = exp.get_model()

    # Inference on gpu
    # model.cuda()

    model.eval()
    model.head.training = False
    model.training = False

    # Load in the weights from training
    weights = torch.load(weights_file, map_location=torch.device('cpu'))
    model.load_state_dict(weights['model'])

    __model_cache[model_id] = (model, exp)
    return model, exp


__model_cache = {}
__val_transform = ValTransform(legacy=False)
