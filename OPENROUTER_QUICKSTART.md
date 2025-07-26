# OpenRouter Integration - Quick Start

This guide shows you how to quickly get started with the OpenRouter integration in Pixeltable.

## What is OpenRouter?

OpenRouter provides a unified API that gives you access to hundreds of AI models from providers like OpenAI, Anthropic, Meta, Google, and many others through a single endpoint. It automatically handles fallbacks and selects cost-effective options.

## Setup (2 minutes)

### 1. Install Dependencies
```bash
pip install openai
```

### 2. Get Your API Key
Visit [https://openrouter.ai/keys](https://openrouter.ai/keys) to get your free API key.

### 3. Configure Pixeltable
```python
import pixeltable as pxt
import os

# Set your API key
pxt.configure_client('openrouter', api_key='your_api_key_here')
# Or use environment variable: pxt.configure_client('openrouter', api_key=os.getenv('OPENROUTER_API_KEY'))
```

## Basic Example (30 seconds)

```python
from pixeltable.functions import openrouter

# Create a table
t = pxt.create_table('my_chat', {'question': pxt.String})
t.insert([{'question': 'What is the capital of France?'}])

# Add AI response using OpenAI's GPT-4o-mini through OpenRouter
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': t.question}
]

t.add_computed_column(
    answer=openrouter.chat_completions(messages, model='openai/gpt-4o-mini')
)

# Get the result
result = t.select(t.question, t.answer).collect()
print(result[0]['answer']['choices'][0]['message']['content'])
# Output: "The capital of France is Paris."
```

## Try Different Models

```python
# Use Claude instead
t.add_computed_column(
    claude_answer=openrouter.chat_completions(messages, model='anthropic/claude-3-haiku')
)

# Use Llama (free tier)
t.add_computed_column(
    llama_answer=openrouter.chat_completions(messages, model='meta-llama/llama-3.1-8b-instruct:free')
)

# Compare all responses
results = t.select(t.question, t.answer, t.claude_answer, t.llama_answer).collect()
```

## Popular Models

| Provider | Model | Description |
|----------|-------|-------------|
| OpenAI | `openai/gpt-4o-mini` | Fast, cost-effective |
| OpenAI | `openai/gpt-4o` | Most capable |
| Anthropic | `anthropic/claude-3-haiku` | Fast, good reasoning |
| Anthropic | `anthropic/claude-3-sonnet` | Balanced performance |
| Meta | `meta-llama/llama-3.1-8b-instruct:free` | Free tier |
| Google | `google/gemini-pro` | Google's flagship |

See all models at: [https://openrouter.ai/models](https://openrouter.ai/models)

## Advanced Features

### Custom Parameters
```python
t.add_computed_column(
    creative_answer=openrouter.chat_completions(
        messages, 
        model='openai/gpt-4o-mini',
        model_kwargs={'temperature': 0.9, 'max_tokens': 200}
    )
)
```

### Site Attribution (for rankings)
```python
t.add_computed_column(
    attributed_answer=openrouter.chat_completions(
        messages, 
        model='openai/gpt-4o-mini',
        site_url='https://myapp.com',
        site_name='My App'
    )
)
```

### Function Calling/Tools
```python
# Define a tool as a Pixeltable UDF
@pxt.udf
def calculator(operation: str, a: float, b: float) -> dict:
    ops = {'add': lambda x,y: x+y, 'multiply': lambda x,y: x*y}
    return {'result': ops.get(operation, lambda x,y: 'Unknown')(a, b)}

tools = pxt.func.Tools([calculator])

t.add_computed_column(
    tool_response=openrouter.chat_completions(
        messages, 
        model='openai/gpt-4o-mini',
        tools=tools.tool_specs
    )
)

# Actually invoke the tools
t.add_computed_column(tool_results=openrouter.invoke_tools(tools, t.tool_response))
```

## Benefits

✅ **Access 100+ models** through one API  
✅ **Automatic fallbacks** if a model is down  
✅ **Cost optimization** - OpenRouter picks cheaper alternatives  
✅ **Familiar OpenAI syntax** - same parameters and responses  
✅ **Free tier models** available  
✅ **No vendor lock-in** - easily switch between providers  

## Next Steps

- Check out the full [OpenRouter Integration Documentation](docs/openrouter_integration.md)
- Run the [complete example](examples/openrouter_example.py)
- Browse available models at [https://openrouter.ai/models](https://openrouter.ai/models)
- Join the OpenRouter community on Discord

Happy coding! 🚀