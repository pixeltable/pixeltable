"""
JFK Files Data Loader
=====================

Scrapes JFK PDF links from the National Archives and loads them into Pixeltable.
Run after schema.py: python load_data.py
"""

from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from config import DIRECTORY
from schema import setup_schema

import pixeltable as pxt


def scrape_jfk_pdf_links() -> list[str]:
    """Scrape PDF links from the National Archives JFK document release page."""
    response = requests.get('https://www.archives.gov/research/jfk/release-2025', timeout=60)
    if response.status_code != 200:
        raise ValueError(f'Failed to retrieve the webpage: {response.status_code}')

    soup = BeautifulSoup(response.content, 'html.parser')
    pdf_data: list[str] = []
    for td in soup.find_all('td'):
        link = td.find('a', href=lambda href: href and href.endswith('.pdf'))
        if link:
            relative_url = link.get('href')
            full_url = urljoin('https://www.archives.gov', relative_url)
            pdf_data.append(full_url)

    return pdf_data


def populate_pixeltable(num_docs: int = 2, load_all: bool = False) -> pxt.Table:
    """Populate the Pixeltable with JFK documents."""
    documents = setup_schema()
    pdf_links = scrape_jfk_pdf_links()

    if not load_all:
        pdf_links = pdf_links[:num_docs]

    if pdf_links:
        documents.insert({'document_url': pdf} for pdf in pdf_links)

    return documents


if __name__ == '__main__':
    table = populate_pixeltable(num_docs=5)
    print(f'Loaded documents into {DIRECTORY}.documents ({table.count()} rows).')
