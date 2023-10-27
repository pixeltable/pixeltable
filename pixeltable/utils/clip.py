import numpy as np
import PIL.Image

from pixeltable.function import FunctionRegistry
from pixeltable.env import Env


def embed_image(img: PIL.Image.Image) -> np.ndarray:
    from pixeltable.functions.image_embedding import openai_clip
    model_info = FunctionRegistry.get().get_nos_info(openai_clip)
    result = Env.get().nos_client.Run(task=model_info.task, model_name=model_info.name, inputs={"images": [img.resize((224, 224))]})
    return result['embedding'].squeeze(0)

def embed_text(text: str) -> np.ndarray:
    from pixeltable.functions.text_embedding import openai_clip
    model_info = FunctionRegistry.get().get_nos_info(openai_clip)
    result = Env.get().nos_client.Run(task=model_info.task, model_name=model_info.name, inputs={"texts": [text]})
    return result['embedding'].squeeze(0)
