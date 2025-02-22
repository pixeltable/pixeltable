from typing import Annotated
import time

from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


llm = ChatAnthropic(model='claude-3-5-sonnet-20240620')


def chatbot(state: State):
    return {'messages': [llm.invoke(state['messages'])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node('chatbot', chatbot)
graph_builder.set_entry_point('chatbot')
graph_builder.set_finish_point('chatbot')
graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    start_time = time.time()
    print('\n' + '=' * 50)
    print(f"Processing: '{user_input}'")
    print('=' * 50 + '\n')

    for event in graph.stream({'messages': [{'role': 'user', 'content': user_input}]}):
        for value in event.values():
            print('Assistant:', value['messages'][-1].content)

    end_time = time.time()
    print('\n' + '-' * 50)
    print(f'Time taken: {end_time - start_time:.2f} seconds')
    print('-' * 50 + '\n')


while True:
    try:
        user_input = input('User: ')
        if user_input.lower() in ['quit', 'exit', 'q']:
            print('\nGoodbye! 👋')
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = 'What do you know about LangGraph?'
        print('User: ' + user_input)
        stream_graph_updates(user_input)
        break

# Tell me a 4 word joke
