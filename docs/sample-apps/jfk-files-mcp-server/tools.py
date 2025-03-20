"""
This module implements the semantic search functionality for the JFK Files demo using FastMCP and Pixeltable.
It allows users to perform natural language queries against a database of JFK document summaries,
leveraging Pixeltable's vector similarity search capabilities.
"""

from config import DIRECTORY
from mcp.server.fastmcp import FastMCP
import pixeltable as pxt

# Initialize FastMCP with our application name
mcp = FastMCP('JFK_Files')


@mcp.tool()
def query_document(query_text: str, top_n: int = 5) -> str:
    """
    Perform semantic search over JFK document summaries using natural language queries.
    
    This function:
    1. Takes a natural language query and converts it to a vector representation
    2. Compares this vector against our document embeddings using similarity search
    3. Returns the top N most semantically similar document summaries
    
    Args:
        query_text: The natural language query to search for
        top_n: Number of most similar documents to return (default: 5)
        
    Returns:
        A formatted string containing the search results with similarity scores
    """
    try:
        # Access our documents table in Pixeltable
        documents = pxt.get_table(f'{DIRECTORY}.documents')
        
        # Calculate similarity between query and document summaries
        sim = documents.document_summary.similarity(query_text)
        
        # Get top N results ordered by similarity score
        results = documents.order_by(sim, asc=False).select(documents.document_summary, sim=sim).limit(top_n).collect()

        # Format the results into a readable string
        result_str = f"Query Results for '{query_text}' in '{DIRECTORY}.documents':\n\n"
        for i, row in enumerate(results):
            result_str += f'{i}. Score: {row["sim"]:.4f}\n'
            result_str += f'   Text: {row["document_summary"]}\n\n'

        return result_str if result_str else 'No results found.'
    except Exception as e:
        return f"Error querying document index '{DIRECTORY}.documents': {str(e)}"
