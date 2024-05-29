import numpy as np
import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer

@pxt.expr_udf
def e5_embed(text: str) -> np.ndarray:
    return sentence_transformer(text, model_id='intfloat/e5-large-v2')

