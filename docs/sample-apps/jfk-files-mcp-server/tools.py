from mcp.server.fastmcp import FastMCP

import pixeltable as pxt

mcp = FastMCP('JFK_Files')


@mcp.tool()
def query_document(query_text: str, top_n: int = 5) -> str:
    try:
        documents = pxt.get_table('jfk.documents')
        sim = documents.document_summary.similarity(query_text)
        results = documents.order_by(sim, asc=False).select(documents.document_summary, sim=sim).limit(top_n).collect()

        result_str = f"Query Results for '{query_text}' in 'jfk.documents':\n\n"
        for i, row in enumerate(results):
            result_str += f'{i}. Score: {row["sim"]:.4f}\n'
            result_str += f'   Text: {row["document_summary"]}\n\n'

        return result_str if result_str else 'No results found.'
    except Exception as e:
        return f"Error querying document index 'jfk.documents': {str(e)}"
