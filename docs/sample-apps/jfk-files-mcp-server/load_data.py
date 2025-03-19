import os
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

import pixeltable as pxt
from pixeltable.functions.mistralai import chat_completions

# Retrieve the API key from environment variables
api_key = os.environ.get('MISTRAL_API_KEY')
if not api_key:
    raise ValueError('MISTRAL_API_KEY not found in environment variables')


def setup_pixeltable_table(directory: str):
    # Initialize Pixeltable
    pxt.drop_dir(directory, force=True)
    pxt.create_dir(directory)

    # Create table
    documents = pxt.create_table(f'{directory}.documents', {'document_url': pxt.String})

    # Define the messages for the chat
    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'Create a detailed summary of the PDF. Extract the key points for the user. Dont Skip out.',
                },
                {'type': 'document_url', 'document_url': documents.document_url},
            ],
        }
    ]
    documents.add_computed_column(api_response=chat_completions(model='mistral-small-latest', messages=messages))
    documents.add_computed_column(document_summary=documents.api_response.choices[0].message.content.astype(pxt.String))
    documents.add_embedding_index(
        column=documents.document_summary,
        embed=pxt.functions.huggingface.sentence_transformer.using(model_id='intfloat/e5-large-v2'),
    )
    return documents


def scrape_jfk_pdf_links() -> list:
    try:
        # Make HTTP request to the webpage
        response = requests.get('https://www.archives.gov/research/jfk/release-2025')

        # Check if request was successful
        if response.status_code != 200:
            raise ValueError(f'Failed to retrieve the webpage: {response.status_code}')

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all table cells with PDF links
        pdf_data = []
        for td in soup.find_all('td'):
            link = td.find('a', href=lambda href: href and href.endswith('.pdf'))
            if link:
                relative_url = link.get('href')
                full_url = urljoin('https://www.archives.gov', relative_url)
                pdf_data.append(full_url)

        return pdf_data
    except Exception as e:
        raise ValueError(f'Error scraping PDF links: {e}')


def populate_pixeltable(directory: str, num_docs: int = 2, load_all: bool = False):
    documents = setup_pixeltable_table(directory)
    pdf_links = scrape_jfk_pdf_links()
    
    # Apply num_docs limit if not loading all
    if not load_all:
        pdf_links = pdf_links[:num_docs]
    
    # Insert the documents
    documents.insert({'document_url': pdf_link for pdf_link in pdf_links})
