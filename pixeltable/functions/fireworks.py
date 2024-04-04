import logging
import os
from typing import Optional

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable import env


@pxt.udf
def chat_completions(
        prompt: str,
        model: str,
        *,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None
) -> dict:
    initialize()
    kwargs = {
        'max_tokens': max_tokens,
        'repetition_penalty': repetition_penalty,
        'top_k': top_k,
        'top_p': top_p,
        'temperature': temperature
    }
    kwargs_not_none = dict(filter(lambda x: x[1] is not None, kwargs.items()))
    import fireworks.client
    return fireworks.client.Completion.create(
        model=model,
        prompt_or_messages=prompt,
        **kwargs_not_none
    ).dict()


def initialize():
    global _is_fireworks_initialized
    if _is_fireworks_initialized:
        return

    _logger.info('Initializing Fireworks client.')

    config = pxt.env.Env.get().config

    if 'fireworks' in config and 'api_key' in config['fireworks']:
        api_key = config['fireworks']['api_key']
    else:
        api_key = os.environ.get('FIREWORKS_API_KEY')
    if api_key is None or api_key == '':
        raise excs.Error('Fireworks client not initialized (no API key configured).')

    import fireworks.client

    fireworks.client.api_key = api_key
    _is_fireworks_initialized = True


_logger = logging.getLogger('pixeltable')
_is_fireworks_initialized = False
