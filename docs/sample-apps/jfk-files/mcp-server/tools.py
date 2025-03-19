import logging

from load_data import scrape_jfk_pdf_links, setup_pixeltable
from mcp.server.fastmcp import FastMCP

import pixeltable as pxt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

mcp = FastMCP('JFK_Files')

logger.info('Setting up Pixeltable')
setup_pixeltable()

logger.info('Loading documents table from Pixeltable')
documents = pxt.get_table('jfk.documents')

logger.info('Starting PDF link scraping')
pdf_links = scrape_jfk_pdf_links('https://www.archives.gov/research/jfk/release-2025')
logger.info(f'Scraped {len(pdf_links)} PDF links in total')

logger.info('Inserting documents into the table (first 10 only)')
count = 0
for pdf in pdf_links[:10]:
    try:
        documents.insert([{'document_url': pdf['url']}])
        count += 1
        logger.info(f'Inserted document {count}/10: {pdf["filename"]}')
    except Exception as e:
        logger.error(f'Error inserting document {pdf["url"]}: {e}')

try:
    result = documents.collect()
    logger.info(f'Successfully collected {len(result)} documents')
    print(result)
except Exception as e:
    logger.error(f'Error collecting documents: {e}')

logger.info('JFK documents loading script completed')


@mcp.tool()
def query_document(query_text: str, top_n: int = 5) -> str:
    """Query the specified document index with a text question.

    Args:
        query_text: The question or text to search for in the document content.
        top_n: Number of top results to return (default is 5).

    Returns:
        A string containing the top matching text chunks and their similarity scores.
    """
    try:
        documents = pxt.get_table('jfk.documents')

        # Calculate similarity scores
        sim = documents.document_summary.similarity(query_text)

        # Get top results
        results = documents.order_by(sim, asc=False).select(documents.document_summary, sim=sim).limit(top_n).collect()

        # Format the results
        result_str = f"Query Results for '{query_text}' in 'jfk.documents':\n\n"
        for i, row in enumerate(results.to_pandas().itertuples(), 1):
            result_str += f'{i}. Score: {row.sim:.4f}\n'
            result_str += f'   Text: {row.document_summary}\n\n'

        return result_str if result_str else 'No results found.'
    except Exception as e:
        return f"Error querying document index 'jfk.documents': {str(e)}"
