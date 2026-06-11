"""JFK Files MCP Server — schema definition.

python schema.py                    # idempotent init
RESET_SCHEMA=true python schema.py  # wipe and recreate
"""

import os

from config import DIRECTORY, MISTRAL_MODEL

import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.functions.mistralai import chat_completions


def setup_schema() -> pxt.Table:
    """Create the documents table, computed columns, and embedding index."""
    if os.getenv('RESET_SCHEMA', 'false').lower() == 'true':
        pxt.drop_dir(DIRECTORY, force=True)

    pxt.create_dir(DIRECTORY, if_exists='ignore')

    documents = pxt.create_table(f'{DIRECTORY}.documents', {'document_url': pxt.String}, if_exists='ignore')

    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': "Create a detailed summary of the PDF. Extract the key points for the user. Don't skip out.",
                },
                {'type': 'document_url', 'document_url': documents.document_url},
            ],
        }
    ]

    documents.add_computed_column(
        api_response=chat_completions(model=MISTRAL_MODEL, messages=messages), if_exists='ignore'
    )
    documents.add_computed_column(
        document_summary=documents.api_response.choices[0].message.content.astype(pxt.String), if_exists='ignore'
    )
    documents.add_embedding_index(
        column=documents.document_summary,
        string_embed=sentence_transformer.using(model_id='intfloat/e5-large-v2'),
        if_exists='ignore',
    )

    return documents


if __name__ == '__main__':
    setup_schema()
    print('Schema setup complete.')
