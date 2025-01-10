import numpy as np

import pixeltable as pxt


# TODO This can go away once we have the ability to inline expr_udf's
@pxt.expr_udf
def clip_text_embed(txt: str) -> np.ndarray:
    return pxt.functions.huggingface.clip(txt, model_id='openai/clip-vit-base-patch32')  # type: ignore[return-value]
