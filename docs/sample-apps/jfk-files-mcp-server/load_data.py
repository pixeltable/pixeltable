"""
JFK Files Data Loader Tutorial
=============================

This script demonstrates how to build a semantic search system for JFK document files using Pixeltable.
It shows how to:
1. Set up a Pixeltable table with computed columns for document summaries
2. Scrape PDF links from the National Archives website
3. Generate AI-powered summaries using Mistral
4. Create embeddings for semantic search

The end result is a searchable database of JFK documents with AI-generated summaries
that can be queried using natural language.
"""

import os
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from config import MISTRAL_MODEL

import pixeltable as pxt
from pixeltable.functions.mistralai import chat_completions

# We'll use Mistral AI for generating document summaries
# Make sure to set your API key in the environment variables
api_key = os.environ.get('MISTRAL_API_KEY')
if not api_key:
    raise ValueError('MISTRAL_API_KEY not found in environment variables')

# This function gets called in server.py so that it intializes sets up Pixeltable and loads the data
def setup_pixeltable_table(directory: str):
    """
    Sets up a Pixeltable table for storing and orchestrating JFK documents.
    
    This function demonstrates several key Pixeltable features:
    1. Table creation with computed columns
    2. Integration with Mistral AI for PDF OCR summarization
    3. Creation of embedding indices for semantic search
    
    Args:
        directory: The Pixeltable directory where the table will be stored
    
    Returns:
        A Pixeltable table configured for document storage and semantic search
    """
    # Start fresh by removing any existing directory
    pxt.drop_dir(directory, force=True)
    pxt.create_dir(directory)

    # Create a table with a column for document URLs
    documents = pxt.create_table(f'{directory}.documents', {'document_url': pxt.String})

    # Define the message structure for Mistral AI to generate document summaries
    # We structure it as a chat message that asks for detailed key points
    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'Create a detailed summary of the PDF. Extract the key points for the user. Dont Skip out.',
                },
                # Note that we are passing the table.column
                {'type': 'document_url', 'document_url': documents.document_url},
            ],
        }
    ]
    
    # Add computed columns that will automatically:
    # 1. Generate AI summaries for each document
    # 2. Extract the summary text from the API response
    documents.add_computed_column(api_response=chat_completions(model=MISTRAL_MODEL, messages=messages))
    documents.add_computed_column(document_summary=documents.api_response.choices[0].message.content.astype(pxt.String))
    
    # Create an embedding index for semantic search using the E5 model
    # This enables natural language queries over the document summaries
    documents.add_embedding_index(
        column=documents.document_summary,
        string_embed=pxt.functions.huggingface.sentence_transformer.using(model_id='intfloat/e5-large-v2'),
    )

    # As a recap:
    # 1. We created a workflow that creates a table
    # 2. Add computed columns that automatically call Mistral for OCR summarries on PDFs
    # 3. Create an embedding index for semantic search using a local embedding model
    # Important: this entire workflow is managed for you upon insert. Now all we have to do is add documents.
    # Pixeltable handles the orchestration, storage, and updates for you!
    return documents


def scrape_jfk_pdf_links() -> list:
    """
    Scrapes PDF links from the National Archives JFK document release page.
    
    This function demonstrates web scraping best practices:
    1. Error handling for HTTP requests
    2. HTML parsing with BeautifulSoup
    3. URL manipulation for converting relative to absolute URLs
    
    Returns:
        list: A list of absolute URLs to JFK document PDFs
    """
    try:
        # Fetch the webpage containing links to JFK documents
        response = requests.get('https://www.archives.gov/research/jfk/release-2025')

        # Ensure the request was successful
        if response.status_code != 200:
            raise ValueError(f'Failed to retrieve the webpage: {response.status_code}')

        # Parse the HTML and extract PDF links
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all table cells containing PDF links and convert to absolute URLs
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
    """
    Populates the Pixeltable with JFK documents and generates their summaries.
    
    This function ties everything together:
    1. Sets up the table structure
    2. Scrapes the document URLs
    3. Loads documents into Pixeltable
    
    Args:
        directory: The Pixeltable directory to use
        num_docs: Number of documents to load (for testing/demo purposes)
        load_all: If True, loads all available documents
    """
    documents = setup_pixeltable_table(directory)
    pdf_links = scrape_jfk_pdf_links()

    # Control the number of documents to process
    if not load_all:
        pdf_links = pdf_links[:num_docs]

    # Insert the documents - Pixeltable will automatically:
    # 1. Generate summaries using Mistral AI
    # 2. Create embeddings for semantic search
    documents.insert({'document_url': pdf} for pdf in pdf_links)
