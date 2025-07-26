# OpenRouter Integration for Pixeltable

This document describes how to use the OpenRouter integration in Pixeltable. OpenRouter provides a unified API that gives you access to hundreds of AI models through a single endpoint, while automatically handling fallbacks and selecting the most cost-effective options.

## Overview

OpenRouter is a platform that aggregates multiple AI model providers (OpenAI, Anthropic, Meta, Google, and many others) into a single, unified API. This integration allows you to:

- Access hundreds of AI models through one API
- Automatically handle model fallbacks and cost optimization
- Use familiar OpenAI SDK patterns
- Support for tools, streaming, and all standard parameters
- Optional site attribution for OpenRouter rankings

## Prerequisites

1. **Install the OpenAI package**: `pip install openai`
2. **Get an OpenRouter API key**: Visit [https://openrouter.ai/keys](https://openrouter.ai/keys)
3. **Configure Pixeltable**: Set up your API key in Pixeltable

## Configuration

```python
import pixeltable as pxt

# Configure Pixeltable to use OpenRouter
pxt.configure_client('openrouter', api_key='your_openrouter_api_key_here')

# Or use environment variable
import os
pxt.configure_client('openrouter', api_key=os.getenv('OPENROUTER_API_KEY'))
```

## Basic Usage

### Chat Completions

```python
from pixeltable.functions import openrouter

# Create a table
t = pxt.create_table('chat_demo', {'prompt': pxt.String})
t.insert([
    {'prompt': 'What is the capital of France?'},
    {'prompt': 'Explain quantum computing in simple terms'}
])

# Add a computed column using OpenAI's GPT-4o-mini through OpenRouter
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': t.prompt}
]

t.add_computed_column(
    response=openrouter.chat_completions(messages, model='openai/gpt-4o-mini')
)

# Query results
results = t.select(t.prompt, t.response).collect()
for row in results:
    content = row['response']['choices'][0]['message']['content']
    print(f"Q: {row['prompt']}")
    print(f"A: {content}\n")
```

### Using Different Model Providers

OpenRouter gives you access to models from various providers:

```python
# OpenAI models
t.add_computed_column(
    gpt_response=openrouter.chat_completions(messages, model='openai/gpt-4o')
)

# Anthropic models
t.add_computed_column(
    claude_response=openrouter.chat_completions(messages, model='anthropic/claude-3-sonnet')
)

# Meta models
t.add_computed_column(
    llama_response=openrouter.chat_completions(messages, model='meta-llama/llama-3.1-70b-instruct')
)

# Google models
t.add_computed_column(
    gemini_response=openrouter.chat_completions(messages, model='google/gemini-pro')
)
```

### Site Attribution

OpenRouter provides rankings and analytics. You can attribute your usage to your site:

```python
t.add_computed_column(
    attributed_response=openrouter.chat_completions(
        messages, 
        model='openai/gpt-4o-mini',
        site_url='https://yourapp.com',
        site_name='Your App Name'
    )
)
```

### Custom Model Parameters

You can pass any OpenAI-compatible parameters:

```python
t.add_computed_column(
    creative_response=openrouter.chat_completions(
        messages, 
        model='openai/gpt-4o-mini',
        model_kwargs={
            'temperature': 0.9,      # More creative
            'max_tokens': 500,       # Longer responses
            'top_p': 0.9,           # Nucleus sampling
            'frequency_penalty': 0.1 # Reduce repetition
        }
    )
)
```

### Tool Usage

OpenRouter supports function calling/tools just like OpenAI:

```python
# Define a Pixeltable UDF as a tool
@pxt.udf
def calculator(operation: str, a: float, b: float) -> dict:
    """Perform basic arithmetic operations"""
    operations = {
        'add': lambda x, y: x + y,
        'subtract': lambda x, y: x - y,
        'multiply': lambda x, y: x * y,
        'divide': lambda x, y: x / y if y != 0 else 'Error: Division by zero'
    }
    
    if operation in operations:
        result = operations[operation](a, b)
        return {'result': result}
    else:
        return {'result': 'Error: Unknown operation'}

# Create tools object
tools = pxt.func.Tools([calculator])

t.add_computed_column(
    tool_response=openrouter.chat_completions(
        messages, 
        model='openai/gpt-4o-mini',
        tools=tools.tool_specs,
        tool_choice={'auto': True, 'required': False, 'tool': None, 'parallel_tool_calls': True}
    )
)

# Actually invoke the tools to get results
t.add_computed_column(tool_results=openrouter.invoke_tools(tools, t.tool_response))
```

## Available Models

OpenRouter provides access to hundreds of models from various providers. Some popular options include:

### OpenAI
- `openai/gpt-4o`
- `openai/gpt-4o-mini`
- `openai/gpt-3.5-turbo`

### Anthropic
- `anthropic/claude-3-opus`
- `anthropic/claude-3-sonnet`
- `anthropic/claude-3-haiku`

### Meta
- `meta-llama/llama-3.1-70b-instruct`
- `meta-llama/llama-3.1-8b-instruct`
- `meta-llama/llama-3.1-8b-instruct:free` (free tier)

### Google
- `google/gemini-pro`
- `google/gemini-pro-vision`

### Other Providers
- Cohere, Mistral, Perplexity, and many more

For the complete and up-to-date list of available models, visit: [https://openrouter.ai/models](https://openrouter.ai/models)

## API Reference

### `openrouter.chat_completions()`

Creates a model response for the given chat conversation using OpenRouter.

**Parameters:**
- `messages` (list): List of messages for chat completion
- `model` (str): Model to use (e.g., 'openai/gpt-4o-mini')
- `model_kwargs` (dict, optional): Additional parameters for the API
- `tools` (list, optional): List of Pixeltable tools to use
- `tool_choice` (dict, optional): Tool choice configuration
- `site_url` (str, optional): Site URL for OpenRouter rankings
- `site_name` (str, optional): Site name for OpenRouter rankings

**Returns:**
- `dict`: Response containing choices, usage, and metadata

### `openrouter.invoke_tools()`

Invokes tools based on an OpenRouter chat completion response.

**Parameters:**
- `tools` (pxt.func.Tools): Pixeltable Tools object containing the available functions
- `response` (exprs.Expr): The response from `openrouter.chat_completions()`

**Returns:**
- `exprs.InlineDict`: Dictionary containing the results of tool invocations

## Error Handling

The integration uses the same error handling as the OpenAI SDK. Common errors include:

- **Authentication Error**: Invalid API key
- **Rate Limit Error**: Too many requests
- **Model Not Found**: Invalid model name
- **Insufficient Credits**: Not enough credits in OpenRouter account

## Best Practices

1. **Model Selection**: Choose the right model for your use case (cost vs. performance)
2. **Rate Limiting**: OpenRouter handles rate limiting, but be mindful of your usage
3. **Error Handling**: Implement proper error handling for production use
4. **Site Attribution**: Use site attribution to appear in OpenRouter rankings
5. **Model Fallbacks**: OpenRouter automatically handles fallbacks, but you can specify multiple models

## Pricing

OpenRouter uses a pay-per-use model with competitive pricing. Each model has different costs:
- Free tier models available (marked with `:free`)
- Premium models with various pricing tiers
- Credits system for billing

Check current pricing at: [https://openrouter.ai/models](https://openrouter.ai/models)

## Support

For issues with the Pixeltable integration:
- Check the Pixeltable documentation
- Open an issue on the Pixeltable GitHub repository

For OpenRouter-specific issues:
- Visit [OpenRouter documentation](https://openrouter.ai/docs)
- Join the OpenRouter Discord community