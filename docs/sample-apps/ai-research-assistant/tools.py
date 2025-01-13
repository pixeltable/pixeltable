import os
import logging
from typing import Dict, Any
import yfinance as yf
import requests
import pandas as pd
import pixeltable as pxt

logger = logging.getLogger(__name__)

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