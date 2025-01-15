import os
import logging
from typing import Dict, Any
import yfinance as yf
import requests
import pandas as pd
import tempfile
import pixeltable as pxt
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

@pxt.udf
def web_search_and_ingest(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search web using DuckDuckGo and ingest PDFs into Pixeltable

    Args:
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        Dict containing ingestion results and status
    """
    ingested_docs = []

    try:
        docs_table = pxt.get_table('research.documents')

        # Search PDFs using DuckDuckGo
        with DDGS() as ddgs:
            search_results = list(ddgs.text(
                f"{query} filetype:pdf",
                max_results=max_results
            ))

        # Process each search result
        for result in search_results:
            pdf_url = result['link']
            try:
                # Download PDF content with timeout
                pdf_response = requests.get(pdf_url, timeout=10)
                if pdf_response.status_code != 200:
                    logger.warning(f"Failed to download PDF from {pdf_url}: Status {pdf_response.status_code}")
                    continue

                # Save to temp file with unique suffix
                suffix = pdf_url.split('/')[-1][-20:] if '/' in pdf_url else 'document.pdf'
                with tempfile.NamedTemporaryFile(suffix=f'-{suffix}', delete=False) as tmp:
                    tmp.write(pdf_response.content)
                    tmp_path = tmp.name

                # Insert into documents table
                docs_table.insert([{
                    'document': tmp_path
                }])

                # Track successful ingestion
                ingested_docs.append({
                    'url': pdf_url,
                    'local_path': tmp_path,
                    'title': result.get('title', 'Unknown')
                })

                logger.info(f"Successfully ingested document from {pdf_url}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error for {pdf_url}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Error processing document from {pdf_url}: {str(e)}")
                continue

        return {
            'success': True,
            'ingested_documents': ingested_docs,
            'total_ingested': len(ingested_docs),
            'message': f"Successfully ingested {len(ingested_docs)} documents"
        }

    except Exception as e:
        error_msg = f"Web search and ingest error: {str(e)}"
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg,
            'ingested_documents': ingested_docs,
            'total_ingested': len(ingested_docs)
        }

@pxt.udf
def get_stock_data(symbol: str) -> Dict[str, Any]:
    """Get current stock data and company information.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dict with stock data or error information
    """
    try:
        stock = yf.Ticker(symbol)
        current_data = stock.history(period='1d')

        # Check if we have data and it's not empty
        if not current_data.empty and len(current_data) > 0:
            return {
                'symbol': symbol,
                'current_price': float(current_data['Close'].iloc[-1]),  # Explicit float conversion
                'company_name': stock.info.get('longName', symbol),
                'success': True
            }
        return {
            'error': f"No data found for {symbol}",
            'success': False
        }
    except Exception as e:
        logger.error(f"Stock data error for {symbol}: {str(e)}")
        return {
            'error': f"Failed to fetch stock data: {str(e)}",
            'success': False
        }

@pxt.udf
def search_news(name: str) -> Dict[str, Any]:
    """Search recent news articles.

    Args:
        name: Search query

    Returns:
        Dict with news articles or error information
    """
    try:
        if not (api_key := os.getenv('NEWS_API_KEY')):
            return {'error': 'NEWS_API_KEY not configured', 'success': False}

        response = requests.get(
            'https://newsapi.org/v2/everything',
            params={
                'q': name,
                'sortBy': 'publishedAt',
                'pageSize': 5,
                'apiKey': api_key
            }
        )

        if response.status_code != 200:
            return {'error': f"API error: {response.status_code}", 'success': False}

        articles = response.json().get('articles', [])[:5]
        return {
            'articles': [{
                'title': article['title'],
                'description': article['description'],
                'url': article['url'],
                'published': article['publishedAt']
            } for article in articles],
            'success': True
        }
    except Exception as e:
        logger.error(f"News API error: {e}")
        return {'error': str(e), 'success': False}

@pxt.udf
def search_documents(name: str) -> Dict[str, Any]:
    """Search documents using semantic similarity.

    Args:
        name: Search query

    Returns:
        Dict containing relevant passages
    """
    try:
        chunks_view = pxt.get_table('research.chunks')
        sim = chunks_view.text.similarity(name)

        results = (chunks_view
                  .where(sim >= 0.6)
                  .order_by(sim, asc=False)
                  .select(chunks_view.text, similarity=sim)
                  .limit(10)
                  .collect())

        return {
            'passages': [{
                'text': r['text'],
                'similarity': float(r['similarity'])
            } for r in results]
        }
    except Exception as e:
        logger.error(f"Document search error: {e}")
        return {'error': str(e), 'success': False}
