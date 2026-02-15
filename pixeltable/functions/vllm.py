"""
Pixeltable UDFs for vLLM models.

Provides integration with vLLM for high-throughput inference with large language models,
supporting chat completions and text generation with HuggingFace models.
"""

from typing import TYPE_CHECKING, Any

import pixeltable as pxt
from pixeltable import exprs
from pixeltable.env import Env
from pixeltable.func.tools import Tools
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    import vllm


@pxt.udf(is_deterministic=False)
def chat_completions(
    messages: list[dict],
    *,
    model: str,
    model_kwargs: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
) -> dict:
    """
    Generate a chat completion from a list of messages using vLLM.

    Uses vLLM's high-throughput inference engine for efficient local LLM serving.
    Models are loaded from HuggingFace and cached for reuse across calls.

    For additional details, see the
    [vLLM documentation](https://docs.vllm.ai/en/stable/).

    Args:
        messages: A list of messages to generate a response for. Each message should be a dict
            with `role` and `content` keys, following the OpenAI chat format.
        model: The HuggingFace model identifier (e.g., `'Qwen/Qwen2.5-0.5B-Instruct'`).
        model_kwargs: Additional keyword args passed to vLLM. Supports both
            `LLM` constructor args (such as `dtype`, `max_model_len`, `gpu_memory_utilization`,
            `tensor_parallel_size`) and `SamplingParams` args (such as `max_tokens`,
            `temperature`, `top_p`, `top_k`). For details, see the
            [vLLM engine args documentation](https://docs.vllm.ai/en/stable/serving/engine_args.html)
            and
            [vLLM sampling params documentation](https://docs.vllm.ai/en/stable/dev/sampling_params.html).
        tools: List of tools available to the model, following the OpenAI format. Each tool should
            be a dict with `name`, `description`, and `parameters` keys.
        tool_choice: Controls which (if any) tool is called by the model.

    Returns:
        A dict containing the chat completion result in OpenAI-compatible format:

        ```python
        {
            'choices': [
                {'index': 0, 'message': {'role': 'assistant', 'content': '...'}}
            ],
            'model': '...',
            'usage': {
                'prompt_tokens': ...,
                'completion_tokens': ...,
                'total_tokens': ...,
            },
        }
        ```

    Example:
        ```python
        import pixeltable as pxt
        from pixeltable.functions import vllm

        t = pxt.create_table('my_table', {'input': pxt.String})

        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': t.input},
        ]

        t.add_computed_column(
            result=vllm.chat_completions(
                messages, model='Qwen/Qwen2.5-0.5B-Instruct'
            )
        )
        ```
    """
    Env.get().require_package('vllm')

    if model_kwargs is None:
        model_kwargs = {}

    import vllm as vllm_lib

    # Separate LLM constructor kwargs from SamplingParams kwargs
    sampling_param_names = _get_sampling_param_names()
    sampling_kwargs: dict[str, Any] = {}
    engine_kwargs: dict[str, Any] = {}
    for k, v in model_kwargs.items():
        if k in sampling_param_names:
            sampling_kwargs[k] = v
        else:
            engine_kwargs[k] = v

    llm = _lookup_model(model, engine_kwargs)
    sampling_params = vllm_lib.SamplingParams(**sampling_kwargs) if sampling_kwargs else None

    # Build chat kwargs
    chat_kwargs: dict[str, Any] = {'use_tqdm': False}
    if sampling_params is not None:
        chat_kwargs['sampling_params'] = sampling_params

    # Handle tools
    if tools is not None:
        chat_kwargs['tools'] = [{'type': 'function', 'function': tool} for tool in tools]

    outputs = llm.chat([messages], **chat_kwargs)
    output = outputs[0]

    return _format_chat_response(output, model)


@pxt.udf(is_deterministic=False)
def generate(prompt: str, *, model: str, model_kwargs: dict[str, Any] | None = None) -> dict:
    """
    Generate text completion for a given prompt using vLLM.

    Uses vLLM's high-throughput inference engine for efficient local LLM serving.
    Models are loaded from HuggingFace and cached for reuse across calls.

    For additional details, see the
    [vLLM documentation](https://docs.vllm.ai/en/stable/).

    Args:
        prompt: The text prompt to generate a completion for.
        model: The HuggingFace model identifier (e.g., `'Qwen/Qwen2.5-0.5B-Instruct'`).
        model_kwargs: Additional keyword args passed to vLLM. Supports both
            `LLM` constructor args (such as `dtype`, `max_model_len`, `gpu_memory_utilization`,
            `tensor_parallel_size`) and `SamplingParams` args (such as `max_tokens`,
            `temperature`, `top_p`, `top_k`). For details, see the
            [vLLM engine args documentation](https://docs.vllm.ai/en/stable/serving/engine_args.html)
            and
            [vLLM sampling params documentation](https://docs.vllm.ai/en/stable/dev/sampling_params.html).

    Returns:
        A dict containing the generation result in OpenAI-compatible format:

        ```python
        {
            'choices': [{'index': 0, 'text': '...'}],
            'model': '...',
            'usage': {
                'prompt_tokens': ...,
                'completion_tokens': ...,
                'total_tokens': ...,
            },
        }
        ```

    Example:
        ```python
        import pixeltable as pxt
        from pixeltable.functions import vllm

        t = pxt.create_table('my_table', {'prompt': pxt.String})

        t.add_computed_column(
            result=vllm.generate(t.prompt, model='Qwen/Qwen2.5-0.5B-Instruct')
        )
        ```
    """
    Env.get().require_package('vllm')

    if model_kwargs is None:
        model_kwargs = {}

    import vllm as vllm_lib

    # Separate LLM constructor kwargs from SamplingParams kwargs
    sampling_param_names = _get_sampling_param_names()
    sampling_kwargs: dict[str, Any] = {}
    engine_kwargs: dict[str, Any] = {}
    for k, v in model_kwargs.items():
        if k in sampling_param_names:
            sampling_kwargs[k] = v
        else:
            engine_kwargs[k] = v

    llm = _lookup_model(model, engine_kwargs)
    sampling_params = vllm_lib.SamplingParams(**sampling_kwargs) if sampling_kwargs else None

    gen_kwargs: dict[str, Any] = {'use_tqdm': False}
    if sampling_params is not None:
        gen_kwargs['sampling_params'] = sampling_params

    outputs = llm.generate([prompt], **gen_kwargs)
    output = outputs[0]

    return _format_generate_response(output, model)


def _get_sampling_param_names() -> set[str]:
    """Return the set of valid SamplingParams field names for disambiguation."""
    global _SAMPLING_PARAM_NAMES  # noqa: PLW0603
    if _SAMPLING_PARAM_NAMES is None:
        import inspect

        import vllm as vllm_lib

        sig = inspect.signature(vllm_lib.SamplingParams)
        _SAMPLING_PARAM_NAMES = set(sig.parameters.keys())
    return _SAMPLING_PARAM_NAMES


_SAMPLING_PARAM_NAMES: set[str] | None = None


def _lookup_model(model: str, engine_kwargs: dict[str, Any]) -> 'vllm.LLM':
    import vllm as vllm_lib

    # Create a hashable key from model name and kwargs
    kwargs_key = tuple(sorted(engine_kwargs.items())) if engine_kwargs else ()
    key = (model, kwargs_key)
    if key not in _model_cache:
        llm = vllm_lib.LLM(model=model, **engine_kwargs)
        _model_cache[key] = llm
    return _model_cache[key]


def _format_chat_response(output: Any, model: str) -> dict:
    """Format a vLLM RequestOutput into an OpenAI-compatible chat completion response."""
    result_output = output.outputs[0]
    prompt_tokens = len(output.prompt_token_ids) if output.prompt_token_ids else 0
    completion_tokens = len(result_output.token_ids) if result_output.token_ids else 0

    message: dict[str, Any] = {'role': 'assistant', 'content': result_output.text}

    return {
        'choices': [{'index': 0, 'message': message}],
        'model': model,
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
        },
    }


def _format_generate_response(output: Any, model: str) -> dict:
    """Format a vLLM RequestOutput into an OpenAI-compatible completion response."""
    result_output = output.outputs[0]
    prompt_tokens = len(output.prompt_token_ids) if output.prompt_token_ids else 0
    completion_tokens = len(result_output.token_ids) if result_output.token_ids else 0

    return {
        'choices': [{'index': 0, 'text': result_output.text}],
        'model': model,
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
        },
    }


def invoke_tools(tools: Tools, response: exprs.Expr) -> exprs.InlineDict:
    """Converts a vLLM response dict to Pixeltable tool invocation format and calls `tools._invoke()`.

    vLLM returns responses in OpenAI-compatible format, so this reuses the OpenAI tool call parser.
    """
    from .openai import _openai_response_to_pxt_tool_calls

    return tools._invoke(_openai_response_to_pxt_tool_calls(response))


_model_cache: dict[tuple, 'vllm.LLM'] = {}


def cleanup() -> None:
    """Clean up cached vLLM models and free resources."""
    _model_cache.clear()
    global _SAMPLING_PARAM_NAMES  # noqa: PLW0603
    _SAMPLING_PARAM_NAMES = None


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
