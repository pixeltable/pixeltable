import numpy as np
from pixeltable.functions.huggingface import sentence_transformer
import pixeltable as pxt


@pxt.expr_udf
def e5_embed(text: str) -> np.ndarray:
    return sentence_transformer(text, model_id='intfloat/e5-large-v2')

@pxt.expr_udf
def minilm(text: str) -> np.ndarray:
    return sentence_transformer(text, model_id="all-MiniLM-L6-v2")
