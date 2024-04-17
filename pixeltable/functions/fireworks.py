from typing import Optional

import fireworks.client

import pixeltable as pxt
from pixeltable import env


def fireworks_client() -> fireworks.client.Fireworks:
    return env.Env.get().get_client('fireworks', lambda api_key: fireworks.client.Fireworks(api_key=api_key))


@pxt.udf
def chat_completions(
        messages: list[dict[str, str]],
        *,
        model: str,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None
) -> dict:
    kwargs = {
        'max_tokens': max_tokens,
        'top_k': top_k,
        'top_p': top_p,
        'temperature': temperature
    }
    kwargs_not_none = dict(filter(lambda x: x[1] is not None, kwargs.items()))
    return fireworks_client().chat.completions.create(
        model=model,
        messages=messages,
        **kwargs_not_none
    ).dict()
