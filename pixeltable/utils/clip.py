import numpy as np
import PIL.Image

from pixeltable.function import FunctionRegistry
from pixeltable.env import Env


def embed_image(img: PIL.Image.Image) -> np.ndarray:
    from pixeltable.functions.image_embedding import openai_clip
    model_info = FunctionRegistry.get().get_nos_info(openai_clip)
    assert model_info.default_method == "encode_image"
    result = Env.get().nos_client.Run(model_info.name, inputs={"images": [img.resize((224, 224))]}, method=model_info.default_method, shm=True)
    return result['embedding'].squeeze(0)

def embed_text(text: str) -> np.ndarray:
    from pixeltable.functions.text_embedding import openai_clip
    model_info, method = FunctionRegistry.get().get_nos_info(openai_clip)
    assert model_info.default_method == "encode_text"
    result = Env.get().nos_client.Run(model_info.name, inputs={"texts": [text]}, method=model_info.default_method, shm=True)
    return result['embedding'].squeeze(0)
