import numpy as np
import PIL.Image

import pixeltable.func as func
from pixeltable.env import Env


def embed_image(img: PIL.Image.Image) -> np.ndarray:
    from pixeltable.functions.nos.image_embedding import openai_clip
    model_info = openai_clip.model_spec
    result = Env.get().nos_client.Run(task=model_info.task, model_name=model_info.name, images=[img.resize((224, 224))])
    return result['embedding'].squeeze(0)

def embed_text(text: str) -> np.ndarray:
    from pixeltable.functions.nos.text_embedding import openai_clip
    model_info = openai_clip.model_spec
    result = Env.get().nos_client.Run(task=model_info.task, model_name=model_info.name, texts=[text])
    return result['embedding'].squeeze(0)
