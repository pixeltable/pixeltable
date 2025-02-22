import pixeltable as pxt
from pixeltable.functions.gemini import generate_content

DIRECTORY = 'gemini'
MODEL = 'gemini-1.5-flash'
RECREATE = False

table_name = f'{DIRECTORY}.chatbot'

# Drop the table if we want to recreate it
if RECREATE:
    pxt.drop_table(table_name)

# Create the table if it doesn't exist
if table_name not in pxt.list_tables():
    # Create a fresh directory
    pxt.create_dir(DIRECTORY, if_exists='ignore')

    # Create the table
    conversations = pxt.create_table(path_str=table_name, schema_or_df={'prompt': pxt.String}, if_exists='ignore')

    # Create the OpenAI response column
    conversations.add_computed_column(response=generate_content(contents=conversations.prompt, model_name=MODEL))

    # Create the answer column
    conversations.add_computed_column(answer=conversations.response['candidates'][0]['content']['parts'][0]['text'])

else:
    # Load the persistent table
    conversations = pxt.get_table(table_name)

# # Lets test it!!
conversations.insert([{'prompt': 'Tell me a 4 word joke.'}])

# Print the results
print(conversations.select(conversations.prompt, conversations.answer).collect())
