"""Token-usage extraction from provider response payloads."""

from __future__ import annotations

from typing import Any

# usage field name pairs in the wild: OpenAI chat completions / embeddings use prompt/completion_tokens;
# OpenAI responses API and Anthropic use input/output_tokens
_INPUT_KEYS = ('prompt_tokens', 'input_tokens')
_OUTPUT_KEYS = ('completion_tokens', 'output_tokens')


def extract_usage(result: Any) -> tuple[str | None, dict[str, int]]:
    """Extract (model, token counts by type) from a provider response payload.

    Returns (None, {}) for unrecognized shapes. Batched results (lists) are summed.
    """
    if isinstance(result, list):
        model: str | None = None
        totals: dict[str, int] = {}
        for item in result:
            item_model, item_tokens = extract_usage(item)
            model = model or item_model
            for k, v in item_tokens.items():
                totals[k] = totals.get(k, 0) + v
        return model, totals

    if not isinstance(result, dict):
        return None, {}
    model = result.get('model') if isinstance(result.get('model'), str) else None
    usage = result.get('usage')
    if not isinstance(usage, dict):
        return model, {}
    tokens: dict[str, int] = {}
    for key in _INPUT_KEYS:
        if isinstance(usage.get(key), int):
            tokens['input'] = usage[key]
            break
    for key in _OUTPUT_KEYS:
        if isinstance(usage.get(key), int):
            tokens['output'] = usage[key]
            break
    return model, tokens
