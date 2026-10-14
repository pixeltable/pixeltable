import os
from typing import Any

from mcp.server.mcpserver import MCPServer

# Variant and port come from the environment so alternate servers can run:
# - 'changed' changes 'pixelmultiple's signature
# - 'gone' omits it entirely
# - the port lets several run at once.
_VARIANT = os.environ.get('PIXELTABLE_MCP_VARIANT', 'full')
_PORT = int(os.environ.get('PIXELTABLE_MCP_PORT', '8000'))

mcp = MCPServer('PixeltableDemo', debug=True, log_level='DEBUG')


if _VARIANT == 'changed':

    @mcp.tool(name='pixelmultiple')
    def pixelmultiple_changed(a: int) -> int:
        """Computes the Pixelmultiple of an integer."""
        return a + 22

elif _VARIANT != 'gone':

    @mcp.tool(name='pixelmultiple')
    def pixelmultiple(a: int, b: int) -> int:
        """Computes the Pixelmultiple of two integers."""
        return (a + 22) * b


@mcp.tool()
def pixeldict(d: dict[str, Any] | None) -> dict[str, Any]:
    """Returns the Pixeldict of a dictionary."""
    if d is None:
        d = {}
    else:
        d = d.copy()
    d['pixelkey'] = 'pixeltable'
    return d


if __name__ == '__main__':
    # port and stateless_http are transport settings in MCP 2; they are no longer constructor arguments
    mcp.run(transport='streamable-http', stateless_http=True, port=_PORT)
