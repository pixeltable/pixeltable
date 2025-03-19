from mcp.server.fastmcp import FastMCP

mcp = FastMCP("JFK_Files")

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
        results = (documents.order_by(sim, asc=False)
                  .select(documents.document_summary, sim=sim)
                  .limit(top_n)
                  .collect())

        # Format the results
        result_str = f"Query Results for '{query_text}' in 'jfk.documents':\n\n"
        for i, row in enumerate(results.to_pandas().itertuples(), 1):
            result_str += f"{i}. Score: {row.sim:.4f}\n"
            result_str += f"   Text: {row.document_summary}\n\n"
        
        return result_str if result_str else "No results found."
    except Exception as e:
        return f"Error querying document index 'jfk.documents': {str(e)}"
