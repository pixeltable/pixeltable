from typing import Optional

import together

import pixeltable as pxt
from pixeltable import env


def together_client() -> together.Together:
    return env.Env.get().get_client('together', lambda api_key: together.Together(api_key=api_key))


@pxt.udf
def completions(
        prompt: str,
        model: str,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[list] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None
) -> dict:
    return together_client().completions.create(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        n=n,
        repetition_penalty=repetition_penalty,
        stop=stop,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature
    ).dict()
