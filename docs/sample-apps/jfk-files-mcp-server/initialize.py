from config import DIRECTORY
from load_data import populate_pixeltable
from mcp.server.fastmcp import FastMCP

mcp = FastMCP('JFK_Files')

# Load a smaller set of documents for quick testing and development
documents = populate_pixeltable(num_docs=5)


@mcp.tool()
def pxt_query_document(query_text: str, top_n: int = 5) -> str:
    """
    Perform semantic search over JFK document summaries using natural language queries.

    Args:
        query_text: The natural language query to search for
        top_n: Number of most similar documents to return (default: 5)

    Returns:
        A formatted string containing the search results with similarity scores
    """
    try:
        sim = documents.document_summary.similarity(string=query_text)
        results = documents.order_by(sim, asc=False).select(documents.document_summary, sim=sim).limit(top_n).collect()

        result_str = f"Query Results for '{query_text}' in '{DIRECTORY}.documents':\n\n"
        for i, row in enumerate(results):
            result_str += f'{i}. Score: {row["sim"]:.4f}\n'
            result_str += f'   Text: {row["document_summary"]}\n\n'

        return result_str if result_str else 'No results found.'
    except Exception as e:
        return f"Error querying document index '{DIRECTORY}.documents': {e!s}"
