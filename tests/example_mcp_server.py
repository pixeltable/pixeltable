from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP('PixeltableDemo', stateless_http=True, debug=True, log_level='DEBUG')


@mcp.tool()
def pixelmultiple(a: int, b: int) -> int:
    """Computes the Pixelmultiple of two integers."""
    return (a + 22) * b


@mcp.tool()
def pixeldict(d: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Returns the Pixeldict of a dictionary."""
    if d is None:
        d = {}
    else:
        d = d.copy()
    d['pixelkey'] = 'pixeltable'
    return d


if __name__ == '__main__':
    mcp.run(transport='streamable-http')
