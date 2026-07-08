from typing import Any


def non_none_dict_factory(d: list[tuple[str, Any]]) -> dict:
    return {k: v for (k, v) in d if v is not None}


# (input_key, output_key) pairs across the provider usage-dict shapes, tried in order:
# openai chat / deepseek / groq / mistralai / openrouter / fireworks (prompt_tokens, completion_tokens);
# openai responses / anthropic (input_tokens, output_tokens); bedrock converse (inputTokens, outputTokens);
# gemini (prompt_token_count, candidates_token_count); ollama, top-level (prompt_eval_count, eval_count).
_TOKEN_KEY_PAIRS = (
    ('prompt_tokens', 'completion_tokens'),
    ('input_tokens', 'output_tokens'),
    ('inputTokens', 'outputTokens'),
    ('prompt_token_count', 'candidates_token_count'),
    ('prompt_eval_count', 'eval_count'),
)


def extract_token_usage(result: Any) -> tuple[int, int] | None:
    """Best-effort (input, output) token counts from a UDF result; None when nothing is recognized.

    Provider UDFs return dict responses that surface usage under a 'usage'/'usage_metadata' key or
    (ollama) at the top level, under one of several key-pair conventions. A batched result is a list of
    such dicts, summed. Anything unrecognized or malformed yields None; this function never raises.
    """
    try:
        if isinstance(result, (list, tuple)):
            total_in = 0
            total_out = 0
            found = False
            for item in result:
                pair = extract_token_usage(item)
                if pair is not None:
                    total_in += pair[0]
                    total_out += pair[1]
                    found = True
            return (total_in, total_out) if found else None
        if not isinstance(result, dict):
            return None
        usage = result.get('usage') or result.get('usage_metadata') or result
        if not isinstance(usage, dict):
            return None
        for in_key, out_key in _TOKEN_KEY_PAIRS:
            n_in = usage.get(in_key)
            n_out = usage.get(out_key)
            if isinstance(n_in, int) or isinstance(n_out, int):
                return (n_in if isinstance(n_in, int) else 0, n_out if isinstance(n_out, int) else 0)
        return None
    except Exception:
        return None
