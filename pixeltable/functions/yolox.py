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


def yolox(img: PIL.Image.Image, *, model_id: str, device: str = 'cpu') -> dict:
    model, exp = get_model(model_id)
    val_transform = ValTransform(legacy=False)

    img, _ = val_transform(np.array(img), None, exp.test_size)
    img = torch.from_numpy(img).unsqueeze(0).float()
    if device == "gpu":
        img = img.cuda()

    with torch.no_grad():
        outputs = model(img)
        outputs = postprocess(
            outputs, 1, exp.test_conf,
            exp.nmsthre, class_agnostic=True
        )
    return outputs


def get_model(name: str) -> (YOLOX, Exp):
    weights_url = f'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{name}.pth'
    weights_file = Path(f'{env.Env.get().tmp_dir}/{name}.pth')
    if not weights_file.exists():
        print(f'Downloading weights for YOLOX model {name}')
        print(f'Downloading from {weights_url} to {weights_file}')
        urlretrieve(weights_url, weights_file)

    exp = get_exp(exp_name=name)
    model = exp.get_model()

    # Inference on gpu
    # model.cuda()

    model.eval()
    model.head.training = False
    model.training = False

    # Load in the weights from training
    weights = torch.load(weights_file, map_location=torch.device('cpu'))
    model.load_state_dict(weights['model'])

    return model, exp
