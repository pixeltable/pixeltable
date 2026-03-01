"""
Pixeltable UDFs for vLLM models.

Provides integration with vLLM for high-throughput inference with large language models,
supporting chat completions and text generation with HuggingFace models.
"""

from typing import TYPE_CHECKING, Any

import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    import vllm


@pxt.udf(is_deterministic=False)
def chat_completions(
    messages: list[dict],
    *,
    model: str,
    engine_kwargs: dict[str, Any] | None = None,
    sampling_params: dict[str, Any] | None = None,
) -> dict:
    """
    Generate a chat completion from a list of messages using vLLM.

    Uses vLLM's high-throughput inference engine for efficient local LLM serving.
    Models are loaded from HuggingFace and cached for reuse across calls.

    For additional details, see the
    [vLLM documentation](https://docs.vllm.ai/en/stable/).

    __Requirements:__

    - `pip install vllm`

    Args:
        messages: A list of messages to generate a response for. Each message should be a dict
            with `role` and `content` keys, following the OpenAI chat format.
        model: The HuggingFace model identifier (e.g., `'Qwen/Qwen2.5-0.5B-Instruct'`).
        engine_kwargs: Additional keyword args for the vLLM `LLM` constructor, such as `dtype`,
            `max_model_len`, `gpu_memory_utilization`, `tensor_parallel_size`. For details, see the
            [vLLM engine args documentation](https://docs.vllm.ai/en/stable/serving/engine_args.html).
        sampling_params: Keyword args for vLLM `SamplingParams`, such as `max_tokens`,
            `temperature`, `top_p`, `top_k`. For details, see the
            [vLLM sampling params documentation](https://docs.vllm.ai/en/stable/dev/sampling_params.html).

    Returns:
        A dict containing the vLLM `RequestOutput` in its native format.

    Examples:
        Add a computed column that generates chat completions:

        >>> t.add_computed_column(
        ...     result=chat_completions(
        ...         t.messages, model='Qwen/Qwen2.5-0.5B-Instruct'
        ...     )
        ... )

        With custom sampling parameters:

        >>> t.add_computed_column(
        ...     result=chat_completions(
        ...         t.messages,
        ...         model='Qwen/Qwen2.5-0.5B-Instruct',
        ...         sampling_params={'max_tokens': 256, 'temperature': 0.7},
        ...     )
        ... )
    """
    Env.get().require_package('vllm')
    import vllm

    llm = _lookup_model(model, engine_kwargs or {})
    sp = vllm.SamplingParams(**(sampling_params or {})) if sampling_params else None

    chat_kwargs: dict[str, Any] = {'use_tqdm': False}
    if sp is not None:
        chat_kwargs['sampling_params'] = sp

    chat_messages: Any = [messages]
    outputs = llm.chat(chat_messages, **chat_kwargs)
    return _request_output_to_dict(outputs[0])


@pxt.udf(is_deterministic=False)
def generate(
    prompt: str,
    *,
    model: str,
    engine_kwargs: dict[str, Any] | None = None,
    sampling_params: dict[str, Any] | None = None,
) -> dict:
    """
    Generate text completion for a given prompt using vLLM.

    Uses vLLM's high-throughput inference engine for efficient local LLM serving.
    Models are loaded from HuggingFace and cached for reuse across calls.

    For additional details, see the
    [vLLM documentation](https://docs.vllm.ai/en/stable/).

    __Requirements:__

    - `pip install vllm`

    Args:
        prompt: The text prompt to generate a completion for.
        model: The HuggingFace model identifier (e.g., `'Qwen/Qwen2.5-0.5B-Instruct'`).
        engine_kwargs: Additional keyword args for the vLLM `LLM` constructor, such as `dtype`,
            `max_model_len`, `gpu_memory_utilization`, `tensor_parallel_size`. For details, see the
            [vLLM engine args documentation](https://docs.vllm.ai/en/stable/serving/engine_args.html).
        sampling_params: Keyword args for vLLM `SamplingParams`, such as `max_tokens`,
            `temperature`, `top_p`, `top_k`. For details, see the
            [vLLM sampling params documentation](https://docs.vllm.ai/en/stable/dev/sampling_params.html).

    Returns:
        A dict containing the vLLM `RequestOutput` in its native format.

    Examples:
        Add a computed column that generates text completions:

        >>> t.add_computed_column(
        ...     result=generate(t.prompt, model='Qwen/Qwen2.5-0.5B-Instruct')
        ... )
    """
    Env.get().require_package('vllm')
    import vllm

    llm = _lookup_model(model, engine_kwargs or {})
    sp = vllm.SamplingParams(**(sampling_params or {})) if sampling_params else None

    gen_kwargs: dict[str, Any] = {'use_tqdm': False}
    if sp is not None:
        gen_kwargs['sampling_params'] = sp

    outputs = llm.generate([prompt], **gen_kwargs)
    return _request_output_to_dict(outputs[0])


def _lookup_model(model: str, engine_kwargs: dict[str, Any]) -> 'vllm.LLM':
    import vllm

    kwargs_key = tuple(sorted(engine_kwargs.items())) if engine_kwargs else ()
    key = (model, kwargs_key)
    if key not in _model_cache:
        _model_cache[key] = vllm.LLM(model=model, **engine_kwargs)
    return _model_cache[key]


def _request_output_to_dict(output: Any) -> dict:
    """Convert a vLLM RequestOutput to a JSON-serializable dict, preserving native structure."""
    return {
        'request_id': output.request_id,
        'prompt': output.prompt,
        'prompt_token_ids': list(output.prompt_token_ids) if output.prompt_token_ids else None,
        'outputs': [
            {
                'index': o.index,
                'text': o.text,
                'token_ids': list(o.token_ids) if o.token_ids else [],
                'cumulative_logprob': o.cumulative_logprob,
                'finish_reason': o.finish_reason,
                'stop_reason': o.stop_reason,
            }
            for o in output.outputs
        ],
        'finished': output.finished,
    }


_model_cache: dict[tuple, 'vllm.LLM'] = {}


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
