from config import MODEL, DIRECTORY

import pixeltable as pxt
from pixeltable.functions import gemini

# Create a fresh directory
pxt.drop_dir(DIRECTORY, force=True)
pxt.create_dir(DIRECTORY, if_exists='ignore')

# Create the table
print('Creating conversations table...')
conversations = pxt.create_table(
    path_str=f'{DIRECTORY}.conversations',
    schema_or_df={'prompt': pxt.String},
    if_exists='ignore',
)

# Create the Gemini response column
conversations.add_computed_column(
    response=gemini.generate_content(
        messages=conversations.prompt,
        model=MODEL,
    )
)

# Create the answer column
conversations.add_computed_column(answer=conversations.response['candidates'][0]['content']['parts'][0]['text'])
print('Setup complete!')
