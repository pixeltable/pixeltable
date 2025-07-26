#!/usr/bin/env python3
"""
OpenRouter Integration Example

This example demonstrates how to use the OpenRouter integration in Pixeltable.
OpenRouter provides access to hundreds of AI models through a single API endpoint.

Prerequisites:
1. Install the OpenAI package: pip install openai
2. Set your OpenRouter API key: export OPENROUTER_API_KEY=your_api_key_here
3. Configure Pixeltable to use OpenRouter (shown below)

For more information about OpenRouter, visit: https://openrouter.ai/docs
"""

import os
import pixeltable as pxt
from pixeltable.functions import openrouter

def main():
    # Configure Pixeltable to use OpenRouter
    # You can get your API key from https://openrouter.ai/keys
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Please set your OPENROUTER_API_KEY environment variable")
        return
    
    pxt.configure_client('openrouter', api_key=api_key)
    
    # Create a simple table for testing
    if pxt.list_tables('openrouter_demo'):
        pxt.drop_table('openrouter_demo')
    
    t = pxt.create_table('openrouter_demo', {
        'prompt': pxt.String,
        'context': pxt.String
    })
    
    # Insert some sample data
    sample_data = [
        {
            'prompt': 'What is the capital of France?',
            'context': 'Geography question'
        },
        {
            'prompt': 'Explain quantum computing in simple terms',
            'context': 'Science education'
        },
        {
            'prompt': 'Write a haiku about spring',
            'context': 'Creative writing'
        }
    ]
    
    t.insert(sample_data)
    
    print("=== Basic OpenRouter Chat Completion ===")
    
    # Add a computed column using OpenAI's GPT-4o-mini through OpenRouter
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': t.prompt}
    ]
    
    t.add_computed_column(
        gpt4_response=openrouter.chat_completions(
            messages, 
            model='openai/gpt-4o-mini'
        )
    )
    
    # Query and display results
    results = t.select(t.prompt, t.gpt4_response).collect()
    for row in results:
        response_content = row['gpt4_response']['choices'][0]['message']['content']
        print(f"Prompt: {row['prompt']}")
        print(f"Response: {response_content}\n")
    
    print("=== Using Different Models ===")
    
    # Add columns for different model providers available through OpenRouter
    t.add_computed_column(
        claude_response=openrouter.chat_completions(
            messages, 
            model='anthropic/claude-3-haiku'
        )
    )
    
    t.add_computed_column(
        llama_response=openrouter.chat_completions(
            messages, 
            model='meta-llama/llama-3.1-8b-instruct:free'
        )
    )
    
    # Compare responses from different models
    comparison_results = t.select(
        t.prompt, 
        t.gpt4_response, 
        t.claude_response, 
        t.llama_response
    ).limit(1).collect()
    
    if comparison_results:
        row = comparison_results[0]
        print(f"Prompt: {row['prompt']}")
        print(f"GPT-4o-mini: {row['gpt4_response']['choices'][0]['message']['content']}")
        print(f"Claude-3-Haiku: {row['claude_response']['choices'][0]['message']['content']}")
        print(f"Llama-3.1-8B: {row['llama_response']['choices'][0]['message']['content']}\n")
    
    print("=== Using OpenRouter with Site Attribution ===")
    
    # Add a column with site attribution for OpenRouter rankings
    t.add_computed_column(
        attributed_response=openrouter.chat_completions(
            messages, 
            model='openai/gpt-4o-mini',
            site_url='https://pixeltable.com',
            site_name='Pixeltable OpenRouter Demo'
        )
    )
    
    print("Added attributed response column (helps with OpenRouter rankings)")
    
    print("=== Using Custom Model Parameters ===")
    
    # Example with custom parameters
    creative_messages = [
        {'role': 'system', 'content': 'You are a creative writing assistant.'},
        {'role': 'user', 'content': t.prompt}
    ]
    
    t.add_computed_column(
        creative_response=openrouter.chat_completions(
            creative_messages, 
            model='openai/gpt-4o-mini',
            model_kwargs={
                'temperature': 0.9,  # More creative
                'max_tokens': 150,   # Shorter responses
                'top_p': 0.9
            }
        )
    )
    
    creative_results = t.select(t.prompt, t.creative_response).limit(1).collect()
    if creative_results:
        row = creative_results[0]
        response_content = row['creative_response']['choices'][0]['message']['content']
        print(f"Creative prompt: {row['prompt']}")
        print(f"Creative response: {response_content}\n")
    
    print("=== Using Tools/Function Calling ===")
    
    # Define a simple calculator tool
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
            return {'result': result, 'operation': operation, 'operands': [a, b]}
        else:
            return {'result': 'Error: Unknown operation', 'operation': operation}
    
    # Create tools object
    tools = pxt.func.Tools([calculator])
    
    # Add a math question to test tools
    t.insert({'prompt': 'What is 25 * 4?', 'context': 'Math question'})
    
    # Add computed column that uses tools
    tool_messages = [
        {'role': 'system', 'content': 'You are a helpful assistant. Use the calculator tool for math questions.'},
        {'role': 'user', 'content': t.prompt}
    ]
    
    t.add_computed_column(
        tool_response=openrouter.chat_completions(
            tool_messages, 
            model='openai/gpt-4o-mini',
            tools=tools.tool_specs,
            tool_choice={'auto': True, 'required': False, 'tool': None, 'parallel_tool_calls': True}
        )
    )
    
    # Invoke the tools to get actual results
    t.add_computed_column(tool_results=openrouter.invoke_tools(tools, t.tool_response))
    
    # Show tool usage results
    tool_results = t.select(t.prompt, t.tool_response, t.tool_results).where(t.prompt.contains('25 * 4')).collect()
    if tool_results:
        row = tool_results[0]
        print(f"Math question: {row['prompt']}")
        if row['tool_results'] and 'calculator' in row['tool_results']:
            calc_result = row['tool_results']['calculator'][0]['result']
            print(f"Calculator result: {calc_result}")
        print()
    
    print("=== Available Models Information ===")
    print("OpenRouter provides access to models from various providers:")
    print("- OpenAI: openai/gpt-4o, openai/gpt-4o-mini, openai/gpt-3.5-turbo")
    print("- Anthropic: anthropic/claude-3-opus, anthropic/claude-3-sonnet, anthropic/claude-3-haiku")
    print("- Meta: meta-llama/llama-3.1-70b-instruct, meta-llama/llama-3.1-8b-instruct")
    print("- Google: google/gemini-pro, google/gemini-pro-vision")
    print("- And many more! Check https://openrouter.ai/models for the full list")
    
    print("\n=== Example Complete ===")
    print("OpenRouter integration allows you to:")
    print("1. Access hundreds of AI models through a single API")
    print("2. Automatically handle fallbacks and cost optimization")
    print("3. Use familiar OpenAI SDK patterns")
    print("4. Support for tools, streaming, and all standard parameters")


if __name__ == '__main__':
    main()