import logging
import os
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from mistralai import Mistral

import pixeltable as pxt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

logger.info('Starting JFK documents loading script')
load_dotenv()

# Retrieve the API key from environment variables
api_key = os.environ.get('MISTRAL_API_KEY')
if not api_key:
    logger.error('MISTRAL_API_KEY not found in environment variables')
    raise ValueError('MISTRAL_API_KEY not found in environment variables')
logger.info('API key retrieved successfully')


@pxt.udf
async def extract_summary(document_url: str) -> str:
    logger.info(f'Extracting summary for document: {document_url}')
    # Define the messages for the chat
    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'Create a detailed summary of the PDF. Extract the key points for the user. Dont Skip out.',
                },
                {'type': 'document_url', 'document_url': document_url},
            ],
        }
    ]

    try:
        # Get the chat response
        client = Mistral(api_key=api_key)
        chat_response = client.chat.complete(model='mistral-small-latest', messages=messages)

        # Print the content of the response
        summary = chat_response.choices[0].message.content
        logger.info(f'Summary extracted successfully ({len(summary)} chars)')
        return summary
    except Exception as e:
        logger.error(f'Error extracting summary: {e}')
        return f'Error extracting summary: {str(e)}'


def setup_pixeltable():
    # Initialize Pixeltable
    logger.info('Initializing Pixeltable directory')
    pxt.drop_dir('jfk', force=True)
    pxt.create_dir('jfk')
    logger.info('Pixeltable directory created')

    # Create table
    logger.info('Creating documents table')
    documents = pxt.create_table('jfk.documents', {'document_url': pxt.String})
    logger.info('Table created successfully')

    logger.info('Adding computed column for document summaries')
    documents.add_computed_column(
        document_summary=extract_summary(document_url=documents.document_url), if_exists='ignore'
    )
    logger.info('Computed column added successfully')

    # Add embedding index
    logger.info('Adding embedding index')
    documents.add_embedding_index(
        column='document_summary',
        string_embed=pxt.functions.huggingface.sentence_transformer.using(model_id='intfloat/e5-large-v2'),
    )
    logger.info('Embedding index added successfully')


def scrape_jfk_pdf_links(url: str) -> list:
    logger.info(f'Scraping PDF links from: {url}')
    # Make HTTP request to the webpage
    try:
        response = requests.get(url)

        # Check if request was successful
        if response.status_code != 200:
            logger.error(f'Failed to retrieve the webpage: {response.status_code}')
            return []

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all table cells with PDF links matching the format from the example
        pdf_data = []

        # Look for table cells containing links to PDF files
        for td in soup.find_all('td'):
            link = td.find('a', href=lambda href: href and href.endswith('.pdf'))
            if link:
                relative_url = link.get('href')
                full_url = urljoin('https://www.archives.gov', relative_url)
                filename = os.path.basename(relative_url)

                pdf_data.append({'filename': filename, 'url': full_url})

        logger.info(f'Found {len(pdf_data)} PDF links')
        return pdf_data
    except Exception as e:
        logger.error(f'Error scraping PDF links: {e}')
        return []
