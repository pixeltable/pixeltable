import numpy as np
import PIL.Image
import clip
import torch
import PIL.Image


_device = 'cuda' if torch.cuda.is_available() else 'cpu'
_model, _preprocess = clip.load("ViT-B/32", device=_device)

def encode_image(img: PIL.Image.Image) -> np.ndarray:
    preprocessed = _preprocess(img).unsqueeze(0).to(_device)
    features = _model.encode_image(preprocessed)
    val = features.numpy(force=True).squeeze()
    return val

def encode_text(txt: str) -> np.ndarray:
    preprocessed = clip.tokenize([txt]).to(_device)
    features = _model.encode_text(preprocessed)
    val = features.numpy(force=True).squeeze()
    return val
