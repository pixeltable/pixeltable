"""
Performance test for chat completion endpoint integrations in Pixeltable.
"""

import argparse
import random
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

import pixeltable as pxt
import pixeltable.functions as pxtf


class ProviderConfig(TypedDict):
    """Configuration for a provider."""
    function: pxt.Function
    default_model: str
    prompt_type: Literal['messages', 'string']
    prompt_udf: pxt.Function
    params: Dict[str, Any]
    response_path: Callable[[Any], Any]
    system_prompt: Optional[str]


def get_random_words(wordlist: List[str], k: int = 2) -> List[str]:
    """Get k random words from the wordlist."""
    return random.sample(wordlist, k=k)


@pxt.udf
def create_chatgpt_prompt(word1: str, word2: str) -> List[Dict[str, str]]:
    """Create a prompt in ChatGPT message format."""
    return [
        {'role': 'system', 'content': 'You are a creative writer who creates natural-sounding sentences.'},
        {
            'role': 'user',
            'content': f'Generate a single sentence that uses both of these words: {word1} and {word2}. '
                       f'The sentence should be natural and make sense.',
        },
    ]


@pxt.udf
def create_claude_prompt(word1: str, word2: str) -> List[Dict[str, str]]:
    """Create a prompt in Claude message format (no system message in messages)."""
    return [
        {
            'role': 'user',
            'content': f'Generate a single sentence that uses both of these words: {word1} and {word2}. '
                       f'The sentence should be natural and make sense.',
        }
    ]


@pxt.udf
def create_simple_prompt(word1: str, word2: str) -> str:
    """Create a simple string prompt."""
    return (
        f'You are a creative writer who creates natural-sounding sentences. '
        f'Generate a single sentence that uses both of these words: {word1} and {word2}. '
        f'The sentence should be natural and make sense.'
    )


@pxt.udf
def create_simple_messages_prompt(word1: str, word2: str) -> List[Dict[str, str]]:
    """Create a simple messages format prompt (user message only)."""
    return [
        {
            'role': 'user',
            'content': f'Generate a single sentence that uses both of these words: {word1} and {word2}. '
                       f'The sentence should be natural and make sense.',
        }
    ]


def create_provider_configs() -> Dict[str, ProviderConfig]:
    """Create configuration for each supported provider."""
    return {
        'openai': {
            'function': pxtf.openai.chat_completions,
            'default_model': 'gpt-4o-mini',
            'prompt_type': 'messages',
            'prompt_udf': create_chatgpt_prompt,
            'params': {
                'max_tokens': 100,
                'temperature': 0.7,
            },
            'response_path': lambda r: r.choices[0].message.content,
            'system_prompt': None,
        },
        'anthropic': {
            'function': pxtf.anthropic.messages,
            'default_model': 'claude-3-haiku-20240307',
            'prompt_type': 'messages',
            'prompt_udf': create_claude_prompt,
            'params': {
                'max_tokens': 100,
                'temperature': 0.7,
            },
            'response_path': lambda r: r.content[0].text,
            'system_prompt': 'You are a creative writer who creates natural-sounding sentences.',
        },
        'gemini': {
            'function': pxtf.gemini.generate_content,
            'default_model': 'gemini-2.0-flash',
            'prompt_type': 'string',
            'prompt_udf': create_simple_prompt,
            'params': {},
            'response_path': lambda r: r,
            'system_prompt': None,
        },
        'fireworks': {
            'function': pxtf.fireworks.chat_completions,
            'default_model': 'accounts/fireworks/models/mixtral-8x22b-instruct',
            'prompt_type': 'messages',
            'prompt_udf': create_simple_messages_prompt,
            'params': {
                'max_tokens': 300,
                'top_k': 40,
                'top_p': 0.9,
                'temperature': 0.7,
            },
            'response_path': lambda r: r,
            'system_prompt': None,
        },
        'groq': {
            'function': pxtf.groq.chat_completions,
            'default_model': 'llama3-8b-8192',
            'prompt_type': 'messages',
            'prompt_udf': create_chatgpt_prompt,
            'params': {
                'max_tokens': 100,
                'temperature': 0.7,
            },
            'response_path': lambda r: r.choices[0].message.content,
            'system_prompt': None,
        },
        'mistralai': {
            'function': pxtf.mistralai.chat_completions,
            'default_model': 'mistral-tiny',
            'prompt_type': 'messages',
            'prompt_udf': create_chatgpt_prompt,
            'params': {
                'max_tokens': 100,
                'temperature': 0.7,
            },
            'response_path': lambda r: r.choices[0].message.content,
            'system_prompt': None,
        },
        'together': {
            'function': pxtf.together.completions,
            'default_model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
            'prompt_type': 'string',
            'prompt_udf': create_simple_prompt,
            'params': {
                'max_tokens': 100,
                'temperature': 0.7,
            },
            'response_path': lambda r: r.choices[0].text,
            'system_prompt': None,
        },
        'deepseek': {
            'function': pxtf.deepseek.chat_completions,
            'default_model': 'deepseek-chat',
            'prompt_type': 'messages',
            'prompt_udf': create_chatgpt_prompt,
            'params': {
                'max_tokens': 100,
                'temperature': 0.7,
            },
            'response_path': lambda r: r.choices[0].message.content,
            'system_prompt': None,
        },
        'bedrock': {
            'function': pxtf.bedrock.converse,
            'default_model': 'anthropic.claude-3-haiku-20240307-v1:0',
            'prompt_type': 'messages',
            'prompt_udf': create_claude_prompt,
            'params': {
                'max_tokens': 100,
                'temperature': 0.7,
            },
            'response_path': lambda r: r.output.message.content[0].text,
            'system_prompt': None,
        },
    }


def main() -> None:
    """Main function to run the test."""
    parser = argparse.ArgumentParser(
        description='Test endpoint providers for sentence generation in Pixeltable',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --provider openai --n 10
  %(prog)s --provider gemini --n 5 --model gemini-1.5-pro
  %(prog)s --provider anthropic --n 20
        '''
    )
    
    provider_configs = create_provider_configs()
    
    parser.add_argument(
        '--provider', 
        required=True,
        choices=list(provider_configs.keys()),
        help='AI provider to use for sentence generation'
    )
    parser.add_argument(
        '--n', 
        type=int, 
        required=True,
        help='Number of word pairs to generate sentences for'
    )
    parser.add_argument(
        '--model',
        help='Model to use (overrides provider default)'
    )
    parser.add_argument(
        '--log-level',
        type=int,
        default=10,
        help='Logging level (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    pxt.configure_logging(level=args.log_level)
    
    # Get provider configuration
    provider_config = provider_configs[args.provider]
    model = args.model or provider_config['default_model']
    
    print(f"Using provider: {args.provider}")
    print(f"Using model: {model}")
    print(f"Generating {args.n} sentences...")
    
    t = pxt.create_table('sentence_tbl', {'word1': pxt.String, 'word2': pxt.String}, if_exists='replace')
    
    # Load wordlist  
    with open('/usr/share/dict/american-english') as f:
        wordlist = [word.strip() for word in f]

    t.add_computed_column(prompt=provider_config['prompt_udf'](t.word1, t.word2))
    
    # Create provider-specific computed column
    if provider_config['prompt_type'] == 'messages':
        # Handle system prompt for providers that support it
        if 'system_prompt' in provider_config and args.provider == 'anthropic':
            t.add_computed_column(
                response=provider_config['function'](
                    messages=t.prompt,
                    model=model,
                    system=provider_config['system_prompt'],
                    model_kwargs=provider_config['params']
                )
            )
        else:
            t.add_computed_column(
                response=provider_config['function'](
                    messages=t.prompt,
                    model=model,
                    model_kwargs=provider_config['params']
                )
            )
    else:  # prompt_type == 'string'
        t.add_computed_column(
            response=provider_config['function'](
                t.prompt,
                model=model,
                model_kwargs=provider_config['params']
            )
        )
    
    # Generate rows
    rows = ({'word1': w1, 'word2': w2} for _ in range(args.n) for w1, w2 in [random.sample(wordlist, k=2)])
    
    # Insert and time the operation
    start = datetime.now()
    status = t.insert(rows, on_error='ignore')
    end = datetime.now()
    
    print(status)
    print(f'Total time: {(end - start).total_seconds():.2f} seconds')
    
    # # Show a few examples
    # print("\nExample generated sentences:")
    # results = t.select(t.word1, t.word2, t.response).limit(3).collect()
    # for row in results:
    #     try:
    #         # Extract content based on provider response format
    #         response_path = provider_config.get('response_path', lambda x: str(x))
    #         content = response_path(row['response'])
    #         print(f"Words: {row['word1']}, {row['word2']}")
    #         print(f"Sentence: {content}")
    #         print()
    #     except Exception as e:
    #         print(f"Error extracting response: {e}")


if __name__ == '__main__':
    main()