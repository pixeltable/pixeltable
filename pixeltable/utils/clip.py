import numpy as np
import PIL.Image


_model = None
_preprocess = None
_tokenizer = None

def _load_model():
    global _model
    global _tokenizer
    global _preprocess
    import open_clip
    _model, _, _preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
    _tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')

def encode_image(img: PIL.Image.Image) -> np.ndarray:
    if _model is None:
        _load_model()
    preprocessed = _preprocess(img).unsqueeze(0)
    features = _model.encode_image(preprocessed)
    val = features.numpy(force=True).squeeze()
    return val

def encode_text(txt: str) -> np.ndarray:
    if _model is None:
        _load_model()
    preprocessed = _tokenizer([txt])
    features = _model.encode_text(preprocessed)
    val = features.numpy(force=True).squeeze()
    return val
