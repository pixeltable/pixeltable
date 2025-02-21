# pip install pixeltable openai
import pixeltable as pxt
from pixeltable.functions.openai import chat_completions, invoke_tools

# Setup
pxt.drop_dir("agents", force=True)
pxt.create_dir("agents")


# Tool Definition
@pxt.udf
def calculator(operation: str, x: float, y: float) -> str:
    """Simple calculator for basic arithmetic operations."""
    if operation == "add":
        return str(x + y)
    elif operation == "subtract":
        return str(x - y)
    elif operation == "multiply":
        return str(x * y)
    elif operation == "divide":
        return str(x / y) if y != 0 else "Cannot divide by zero"
    elif operation == "compare":
        if x > y:
            return f"{x} is greater than {y}"
        elif x < y:
            return f"{y} is greater than {x}"
        else:
            return f"{x} is equal to {y}"


tools = pxt.tools(calculator)

# Agent Table Creation and Initial Response
calc_agent = pxt.create_table("agents.calc", {"prompt": pxt.String}, if_exists="ignore")
messages = [{"role": "user", "content": calc_agent.prompt}]

calc_agent.add_computed_column(
    initial_response=chat_completions(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice=tools.choice(required=True),
    )
)

calc_agent.add_computed_column(
    tool_output=invoke_tools(tools, calc_agent.initial_response)
)


# Add tool response to prompt
@pxt.udf
def create_prompt(question: str, tool_outputs: list[dict]) -> str:
    return f"""
   QUESTION:

   {question}

   TOOL OUTPUTS:

   {tool_outputs}
  """


calc_agent.add_computed_column(
    tool_response_prompt=create_prompt(calc_agent.prompt, calc_agent.tool_output)
)

# Send back to OpenAI for final response
messages = [
    {
        "role": "system",
        "content": "Answer the users question based on the provided tool outputs.",
    },
    {"role": "user", "content": calc_agent.tool_response_prompt},
]

calc_agent.add_computed_column(
    final_response=chat_completions(model="gpt-4o-mini", messages=messages)
)

calc_agent.add_computed_column(
    answer=calc_agent.final_response.choices[0].message.content
)

# Usage Example
calc_agent.insert(prompt="Which is bigger 9.11 or 9.8?")
print(calc_agent.select(calc_agent.tool_output, calc_agent.answer).collect())
