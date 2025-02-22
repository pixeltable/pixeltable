import pixeltable as pxt
from datetime import datetime
from pixeltable.functions import openai
from pixeltable.functions.huggingface import sentence_transformer
from typing import List, Dict

# Initialize Pixeltable
pxt.drop_dir("agent", force=True)
pxt.create_dir("agent")

############################################################
# Create Memory Store
############################################################

memory = pxt.create_table(
    path_str="agent.memory",
    schema_or_df={
        "role": pxt.String,
        "content": pxt.String,
        "timestamp": pxt.Timestamp,
    },
    if_exists="ignore",
)

# Add embedding for semantic search
embed_model = sentence_transformer.using(model_id="intfloat/e5-large-v2")
memory.add_embedding_index(column="content", string_embed=embed_model)

############################################################
# Memory Retrieval Tools
############################################################

@pxt.query
def get_recent_memory():
    """Fetch the 5 most recent messages from memory."""
    return (
        memory.order_by(memory.timestamp, asc=False)
        .select(role=memory.role, content=memory.content)
        .limit(5)
    )

@pxt.query
def get_semantic_memory(query_text: str):
    """Search for 5 semantically relevant memories."""
    sim = memory.content.similarity(query_text)
    return (
        memory.order_by(sim, asc=False)
        .select(role=memory.role, content=memory.content)
        .limit(5)
    )

# Create tools
tools = pxt.tools(get_recent_memory, get_semantic_memory)

############################################################
# Create Agent
############################################################

chat_session = pxt.create_table(
    path_str="agent.chat_session",
    schema_or_df={"user_message": pxt.String, "timestamp": pxt.Timestamp},
    if_exists="ignore",
)

@pxt.udf
def create_messages(recent_context: List[Dict], semantic_context: Dict, current_message: str) -> List[Dict]:
    """Create structured message list combining both types of memory"""
    messages = [
        {
            "role": "system",
            "content": """You are an assistant with access to both recent and semantic memory.
            Use both recent context and relevant past interactions to provide informed responses."""
        }
    ]

    # Add recent context
    if recent_context:
        messages.extend([{"role": msg["role"], "content": msg["content"]} for msg in recent_context])

    # Add semantic context if available
    if semantic_context and 'get_semantic_memory' in semantic_context:
        semantic_messages = semantic_context['get_semantic_memory']
        if semantic_messages:
            for msg in semantic_messages:
                msg_dict = {"role": msg["role"], "content": msg["content"]}
                if msg_dict not in messages:
                    messages.append(msg_dict)

    # Add current message
    messages.append({"role": "user", "content": current_message})

    return messages

# Add computed columns
chat_session.add_computed_column(recent_context=get_recent_memory())

chat_session.add_computed_column(
    semantic_tool_response=openai.chat_completions(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": chat_session.user_message}],
        tools=tools,
        tool_choice=tools.choice(required=True, parallel_tool_calls=True),
    )
)

chat_session.add_computed_column(
    semantic_context=openai.invoke_tools(tools, chat_session.semantic_tool_response)
)

chat_session.add_computed_column(
    messages=create_messages(
        chat_session.recent_context,
        chat_session.semantic_context,
        chat_session.user_message
    )
)

chat_session.add_computed_column(
    llm_response=openai.chat_completions(
        messages=chat_session.messages,
        model="gpt-4o-mini"
    )
)

chat_session.add_computed_column(
    assistant_response=chat_session.llm_response.choices[0].message.content
)

############################################################
# Chat Interface
############################################################

def chat(message: str) -> str:
    """Process a message through the memory-enabled agent"""
    # Store user message in memory
    memory.insert([{"role": "user", "content": message, "timestamp": datetime.now()}])
    
    # Process through chat session
    chat_session.insert([{"user_message": message, "timestamp": datetime.now()}])
    
    # Get response
    result = (
        chat_session.select(chat_session.assistant_response)
        .where(chat_session.user_message == message)
        .collect()
    )
    
    response = result["assistant_response"][0]
    
    # Store assistant response in memory
    memory.insert([{
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now()
    }])
    
    return response

if __name__ == "__main__":
    # Basic introduction and recall
    responses = [
        chat("Hi! My name is Alice Smith."),
        chat("I work as a software engineer at TechCorp."),
        chat("Do you remember where I work?"),
        chat("What's my full name?"),
    ]

    # Context-dependent questions
    responses.extend([
        chat("I have a dog named Max."),
        chat("He's a 3-year-old Golden Retriever."),
        chat("Who am I talking about?"),
        chat("How old is he and what breed?"),
    ])

    # Testing longer-term memory
    responses.extend([
        chat("I'm planning a trip to Japan next month."),
        chat("I'll be visiting Tokyo and Kyoto."),
        chat("I'm really excited about trying the local cuisine!"),
        chat("What destinations did I mention earlier for my trip?"),
        chat("By the way, what was the name of my dog again?"),  # Testing recall of older information
    ])

    # Complex contextual questions
    responses.extend([
        chat("I'm thinking of bringing Max to my parents' house while I'm in Japan."),
        chat("They live in Boston."),
        chat("Could you remind me of my travel plans and where my parents live?"),
    ])

    # Testing memory of multiple conversation threads
    responses.extend([
        chat("I'm also taking Japanese lessons to prepare for my trip."),
        chat("What details do you remember about: 1) my job, 2) my dog, and 3) my upcoming trip?"),
    ])

    # Print the conversation
    for i, response in enumerate(responses, 1):
        print(f"\nExchange {i}:")
        print(f"Bot: {response}")