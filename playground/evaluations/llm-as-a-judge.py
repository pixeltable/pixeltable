# schema.py
import pixeltable as pxt
from pixeltable.functions import openai

# Initialize app structure
pxt.drop_dir("chatbot", force=True)
pxt.create_dir("chatbot")

# Define data schema
conversations = pxt.create_table(
    path_str="chatbot.conversations", 
    schema_or_df={
        "prompt": pxt.String,
        "expected_criteria": pxt.String  # Add criteria for evaluation
    }, 
    if_exists="ignore"
)

# Configure processing pipeline
conversations.add_computed_column(
    messages=[{"role": "user", "content": conversations.prompt}]
)

conversations.add_computed_column(
    response=openai.chat_completions(
        messages=conversations.messages,
        model="gpt-4o-mini",
    )
)

conversations.add_computed_column(
    answer=conversations.response.choices[0].message.content
)

# Add judge evaluation pipeline
judge_prompt_template = """
You are an expert judge evaluating AI responses. Your task is to evaluate the following response based on the given criteria.

Original Prompt: {prompt}
Expected Criteria: {criteria}
AI Response: {response}

Please evaluate the response on a scale of 1-10 and provide a brief explanation.
Format your response as:
Score: [1-10]
Explanation: [Your explanation]
"""

conversations.add_computed_column(
    judge_prompt=judge_prompt_template.format(
        prompt=conversations.prompt,
        criteria=conversations.expected_criteria,
        response=conversations.answer
    )
)

conversations.add_computed_column(
    judge_response=openai.chat_completions(
        messages=[
            {"role": "system", "content": "You are an expert judge evaluating AI responses."},
            {"role": "user", "content": conversations.judge_prompt}
        ],
        model="gpt-4o-mini",
    )
)

conversations.add_computed_column(
    evaluation=conversations.judge_response.choices[0].message.content
)

# Add helper function to parse score
@pxt.udf
def extract_score(evaluation: str) -> float:
    try:
        score_line = [line for line in evaluation.split('\n') if line.startswith('Score:')][0]
        return float(score_line.split(':')[1].strip())
    except:
        return 0.0

conversations.add_computed_column(
    score=extract_score(conversations.evaluation)
)

# app.py
import pixeltable as pxt

def run_evaluation():
    # Connect to your app
    conversations = pxt.get_table("chatbot.conversations")

    # Example prompts with evaluation criteria
    test_cases = [
        {
            "prompt": "Write a haiku about dogs.",
            "expected_criteria": "The response should: 1) Follow 5-7-5 syllable pattern, 2) Be about dogs, 3) Use vivid imagery"
        },
        {
            "prompt": "Explain quantum computing to a 10-year-old.",
            "expected_criteria": "The response should: 1) Use age-appropriate language, 2) Use relevant analogies, 3) Be engaging and clear"
        }
    ]

    # Insert test cases
    conversations.insert(test_cases)

    # Get results with evaluations
    results = conversations.select(
        conversations.prompt,
        conversations.answer,
        conversations.evaluation,
        conversations.score
    ).collect().to_pandas()

    # Print results
    for idx, row in results.iterrows():
        print(f"\nTest Case {idx + 1}")
        print("=" * 50)
        print(f"Prompt: {row['prompt']}")
        print(f"\nResponse: {row['answer']}")
        print(f"\nEvaluation:\n{row['evaluation']}")
        print(f"Score: {row['score']}")
        print("=" * 50)

if __name__ == "__main__":
    run_evaluation()