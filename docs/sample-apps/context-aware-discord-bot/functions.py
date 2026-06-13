import config
import numpy as np

import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer


@pxt.expr_udf
def get_embeddings(text: str) -> np.ndarray:
    return sentence_transformer(text, model_id=config.EMBEDDING_MODEL_ID)


@pxt.udf
def create_prompt(context: list[dict], question: str) -> str:
    context_str = '\n'.join(
        f'{msg["username"]}: {msg["text"]}' for msg in context if msg['sim'] > config.SIMILARITY_THRESHOLD
    )
    return f'Context:\n{context_str}\n\nQuestion: {question}'
