import os
from mistralai import Mistral

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import pixeltable as pxt

from dotenv import load_dotenv

load_dotenv()

base_url = "https://www.archives.gov/research/jfk/release-2025"

# Initialize Pixeltable
pxt.drop_dir('jfk', force=True)
pxt.create_dir('jfk')

# Retrieve the API key from environment variables
api_key = os.environ["MISTRAL_API_KEY"]

# Create table
documents = pxt.create_table(
    'jfk.documents', {'document_url': pxt.String}
)

@pxt.udf
async def extract_summary(document_url: str) -> str:
    # Define the messages for the chat
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Create a detailed summary of the PDF. Extract the key points for the user. Dont Skip out."
                },
                {
                    "type": "document_url",
                    "document_url": document_url
                }
            ]
        }
    ]

    # Get the chat response
    client = Mistral(api_key=api_key)
    chat_response = client.chat.complete(
        model="mistral-small-latest",
        messages=messages
    )

    # Print the content of the response
    return chat_response.choices[0].message.content

documents.add_computed_column(
    document_summary=extract_summary(document_url=documents.document_url),
    if_exists="ignore",
)

# Add embedding index
documents.add_embedding_index(
    column='document_summary',
    string_embed=pxt.functions.huggingface.sentence_transformer.using(model_id='intfloat/e5-large-v2')
)
   
def scrape_jfk_pdf_links(url: str) -> list:
    # Make HTTP request to the webpage
    response = requests.get(url)
    
    # Check if request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve the webpage: {response.status_code}")
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
            full_url = urljoin("https://www.archives.gov", relative_url)
            filename = os.path.basename(relative_url)
            
            pdf_data.append({
                'filename': filename,
                'url': full_url
            })
    
    return pdf_data

pdf_links = scrape_jfk_pdf_links(base_url)

for pdf in pdf_links:
    documents.insert([{
        'document_url': pdf['url'],
    }])

print(documents.collect())