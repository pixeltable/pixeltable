"""
Performance test for chat completion endpoint integrations in Pixeltable.
"""

import argparse
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pixeltable as pxt
import pixeltable.functions as pxtf


@dataclass
class ProviderConfig:
    """Configuration for a provider."""

    prompt_udf: pxt.Function
    udf: pxt.Function
    default_model: str
    kwargs: dict[str, Any]


def get_random_words(wordlist: list[str], k: int = 2) -> list[str]:
    """Get k random words from the wordlist."""
    return random.sample(wordlist, k=k)


def instructions(count: int, word1: str, word2: str) -> str:
    return (
        f'Generate a roughly {count}-token essay that uses both of these words: {word1} and {word2}. '
        f'The prose should be natural and make sense.'
    )


@pxt.udf
def create_chatgpt_prompt(count: int, word1: str, word2: str) -> list[dict[str, str]]:
    """Create a prompt in ChatGPT message format."""
    return [
        {'role': 'system', 'content': 'You are a creative writer who creates natural-sounding prose.'},
        {'role': 'user', 'content': instructions(count, word1, word2)},
    ]


@pxt.udf
def create_simple_messages_prompt(count: int, word1: str, word2: str) -> list[dict[str, str]]:
    """Create a prompt in Claude message format (no system message in messages)."""
    return [{'role': 'user', 'content': instructions(count, word1, word2)}]


@pxt.udf
def create_simple_prompt(count: int, word1: str, word2: str) -> str:
    """Create a simple string prompt."""
    return f'You are a creative writer who creates natural-sounding prose. {instructions(count, word1, word2)}'


def create_provider_configs(max_tokens: int) -> dict[str, ProviderConfig]:
    """Create configuration for each supported provider."""
    from google.genai.types import GenerateContentConfigDict

    return {
        'openai': ProviderConfig(
            prompt_udf=create_chatgpt_prompt,
            udf=pxtf.openai.chat_completions,
            default_model='gpt-4o-mini',
            kwargs={'model_kwargs': {'max_tokens': max_tokens, 'temperature': 0.7}},
        ),
        'anthropic': ProviderConfig(
            prompt_udf=create_simple_messages_prompt,
            udf=pxtf.anthropic.messages,
            default_model='claude-3-haiku-20240307',
            kwargs={
                'max_tokens': max_tokens,
                'model_kwargs': {
                    'temperature': 0.7,
                    'system': 'You are a creative writer who creates natural-sounding sentences.',
                },
            },
        ),
        'gemini': ProviderConfig(
            prompt_udf=create_simple_prompt,
            udf=pxtf.gemini.generate_content,
            default_model='gemini-2.0-flash',
            kwargs={
                'config': GenerateContentConfigDict(
                    candidate_count=3,
                    stop_sequences=['\n'],
                    max_output_tokens=300,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=40,
                    response_mime_type='text/plain',
                    presence_penalty=0.6,
                    frequency_penalty=0.6,
                )
            },
        ),
        'fireworks': ProviderConfig(
            prompt_udf=create_simple_messages_prompt,
            udf=pxtf.fireworks.chat_completions,
            default_model='accounts/fireworks/models/mixtral-8x22b-instruct',
            kwargs={'model_kwargs': {'max_tokens': max_tokens, 'top_k': 40, 'top_p': 0.9, 'temperature': 0.7}},
        ),
        'groq': ProviderConfig(
            prompt_udf=create_chatgpt_prompt,
            udf=pxtf.groq.chat_completions,
            default_model='llama3-8b-8192',
            kwargs={'model_kwargs': {'max_tokens': max_tokens, 'temperature': 0.7}},
        ),
        'mistralai': ProviderConfig(
            prompt_udf=create_chatgpt_prompt,
            udf=pxtf.mistralai.chat_completions,
            default_model='mistral-tiny',
            kwargs={'model_kwargs': {'max_tokens': max_tokens, 'temperature': 0.7}},
        ),
        'together': ProviderConfig(
            prompt_udf=create_simple_prompt,
            udf=pxtf.together.completions,
            default_model='mistralai/Mixtral-8x7B-Instruct-v0.1',
            kwargs={'model_kwargs': {'max_tokens': max_tokens, 'temperature': 0.7}},
        ),
        'deepseek': ProviderConfig(
            prompt_udf=create_chatgpt_prompt,
            udf=pxtf.deepseek.chat_completions,
            default_model='deepseek-chat',
            kwargs={'model_kwargs': {'max_tokens': max_tokens, 'temperature': 0.7}},
        ),
        'bedrock': ProviderConfig(
            prompt_udf=create_simple_messages_prompt,
            udf=pxtf.bedrock.converse,
            default_model='anthropic.claude-3-haiku-20240307-v1:0',
            kwargs={'model_kwargs': {'max_tokens': max_tokens, 'temperature': 0.7}},
        ),
    }


def main() -> None:
    """Main function to run the test."""
    parser = argparse.ArgumentParser(
        description='Test LLM endpoint providers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --provider openai --n 10 --t 100
  %(prog)s --provider gemini --n 5 --model gemini-1.5-pro
  %(prog)s --provider anthropic --n 20
        """,
    )

    parser.add_argument('--provider', required=True, help='AI provider to use for sentence generation')
    parser.add_argument('--n', type=int, required=True, help='Number of word pairs')
    parser.add_argument('--t', type=int, default=10, help='Length of output in tokens')
    parser.add_argument('--model', help='Model to use (overrides provider default)')
    parser.add_argument('--log-level', type=int, default=10, help='Logging level (default: 10)')
    args = parser.parse_args()

    provider_configs = create_provider_configs(args.t)
    # Load wordlist
    with open('/usr/share/dict/american-english', encoding='utf-8') as f:
        wordlist = [word.strip() for word in f]

    pxt.configure_logging(level=args.log_level)
    provider_config = provider_configs[args.provider]
    model = args.model or provider_config.default_model

    print(f'Using provider: {args.provider}')
    print(f'Using model: {model}')
    print(f'Generating {args.n} rows x {args.t} tokens')

    t = pxt.create_table('sentence_tbl', {'word1': pxt.String, 'word2': pxt.String}, if_exists='replace')
    t.add_computed_column(prompt=provider_config.prompt_udf(args.t, t.word1, t.word2))
    t.add_computed_column(response=provider_config.udf(t.prompt, model=model, **provider_config.kwargs))

    rows = ({'word1': word1, 'word2': word2} for word1, word2 in (random.sample(wordlist, k=2) for _ in range(args.n)))
    start = datetime.now()
    status = t.insert(rows, on_error='ignore')
    # make sure we're not testing a service that's experiencing an outage?
    # assert status.num_excs <= int(args.n * 0.01), status
    end = datetime.now()

    print(status)
    print(f'Total time: {(end - start).total_seconds():.2f} seconds')


if __name__ == '__main__':
    main()
