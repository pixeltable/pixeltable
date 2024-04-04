from typing import Optional

import pixeltable as pxt


@pxt.udf
def completions(
        prompt: str,
        model: str,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[list] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None
) -> dict:
    import together
    return together.Complete.create(
        prompt,
        model,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        stop=stop,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature
    )
