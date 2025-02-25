import logging
from typing import List, Dict, Any
import pixeltable as pxt
from pixeltable.functions.anthropic import messages
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# System prompts as global variables
GENERATION_SYSTEM_PROMPT = """
Your task is to Generate the best content possible for the user's request.
If the user provides critique, respond with a revised version of your previous attempt.
You must always output the revised content.
"""

REFLECTION_SYSTEM_PROMPT = """
You are tasked with generating critique and recommendations to the user's generated content.
If the user content has something wrong or something to be improved, output a list of recommendations
and critiques. If the user content is ok and there's nothing to change, output this: <OK>
"""

# Drop and recreate directory
if 'reflection_agent' in pxt.list_tables():
    logger.info("Dropping and recreating reflection_agent directory")
    pxt.drop_dir('reflection_agent', force=True)

print(pxt.list_tables())
pxt.create_dir('reflection_agent')

# Create base table for storing conversations
conversations = pxt.create_table(
    'reflection_agent.conversations',
    {
        'prompt': pxt.String,
        'timestamp': pxt.Timestamp,
        'request_id': pxt.String
    }
)

@pxt.udf
def create_generation_messages(prompt: str) -> List[Dict[str, str]]:
    """Creates the message structure for generation."""
    return [
        {'role': 'system', 'content': GENERATION_SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt}
    ]

@pxt.udf
def create_reflection_messages(content: str) -> List[Dict[str, str]]:
    """Creates the message structure for reflection."""
    return [
        {'role': 'system', 'content': REFLECTION_SYSTEM_PROMPT},
        {'role': 'user', 'content': content}
    ]

# Add computed columns for the generation phase
conversations.add_computed_column(
    generation_messages=create_generation_messages(conversations.prompt)
)

conversations.add_computed_column(
    generation_response=messages(
        model='claude-3-sonnet-20240229',
        messages=conversations.generation_messages,
        temperature=0.7
    )
)

conversations.add_computed_column(
    generated_content=conversations.generation_response.content[0].text
)

# Add computed columns for the reflection phase
conversations.add_computed_column(
    reflection_messages=create_reflection_messages(conversations.generated_content)
)

conversations.add_computed_column(
    reflection_response=messages(
        model='claude-3-sonnet-20240229',
        messages=conversations.reflection_messages,
        temperature=0.7
    )
)

conversations.add_computed_column(
    reflection_content=conversations.reflection_response.content[0].text
)

@pxt.udf
def update_generation_messages(
    previous_messages: List[Dict[str, str]],
    generation: str,
    critique: str
) -> List[Dict[str, str]]:
    """Updates the message history with the latest generation and critique."""
    messages = previous_messages.copy()
    messages.append({'role': 'assistant', 'content': generation})
    messages.append({'role': 'user', 'content': critique})
    return messages

def run_reflection_loop(prompt: str, max_iterations: int = 5) -> Dict[str, Any]:
    """
    Runs the reflection loop using Pixeltable's computed columns.
    
    Args:
        prompt: The initial user prompt
        max_iterations: Maximum number of reflection iterations
        
    Returns:
        Dict containing the final response and metadata
    """
    request_id = f'req_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    # Insert initial prompt
    conversations.insert([{
        'prompt': prompt,
        'timestamp': datetime.now(),
        'request_id': request_id
    }])
    
    current_result = conversations.where(
        conversations.request_id == request_id
    ).select(
        conversations.generated_content,
        conversations.reflection_content
    ).collect()
    
    generation = current_result['generated_content'][0]
    critique = current_result['reflection_content'][0]
    
    iteration = 1
    while iteration < max_iterations:
        # Insert new iteration with updated messages
        conversations.insert([{
            'prompt': critique,  # Use critique as the new prompt
            'timestamp': datetime.now(),
            'request_id': f"{request_id}_iter_{iteration}"
        }])
        
        # Get latest results
        current_result = conversations.where(
            conversations.request_id == f"{request_id}_iter_{iteration}"
        ).select(
            conversations.generated_content,
            conversations.reflection_content
        ).collect()
        
        generation = current_result['generated_content'][0]
        critique = current_result['reflection_content'][0]
        iteration += 1
    
    return {
        'final_response': generation,
        'iterations': iteration,
        'request_id': request_id
    }

def main():
    """Main function to demonstrate the reflection agent."""
    test_prompt = "Tell me a 4 word joke."
    result = run_reflection_loop(test_prompt)
    
    print(f"\nFinal response after {result['iterations']} iterations:")
    print(result['final_response'])
    
    # Print full conversation history
    history = conversations.where(
        conversations.request_id.like(f"{result['request_id']}%")
    ).select(
        conversations.request_id,
        conversations.generated_content,
        conversations.reflection_content
    ).collect()
    
    print("\nFull conversation history:")
    for _, row in history.iterrows():
        print(f"\nIteration {row['request_id']}:")
        print(f"Generated: {row['generated_content']}")
        print(f"Reflection: {row['reflection_content']}")

if __name__ == "__main__":
    main() 