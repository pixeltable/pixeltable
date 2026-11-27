import os
from typing import Any

from mcp.server import ServerRequestContext
from mcp.server.mcpserver import MCPServer
from mcp.types import ListToolsResult, PaginatedRequestParams

# Variant and port come from the environment so alternate servers can run:
# - 'changed' changes 'pixelmultiple's signature
# - 'gone' omits it entirely
# - 'paged' lists one tool per tools/list page, with 'pixelmultiple' on the last one
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


if _VARIANT == 'paged':

    async def list_tools_page(ctx: ServerRequestContext, params: PaginatedRequestParams | None) -> ListToolsResult:
        tools = sorted(await mcp.list_tools(), key=lambda tool: tool.name)
        page = 0 if params is None or params.cursor is None else int(params.cursor)
        next_cursor = str(page + 1) if page + 1 < len(tools) else None
        return ListToolsResult(tools=tools[page : page + 1], next_cursor=next_cursor)

    # MCPServer has no pagination option, so replace its tools/list handler
    mcp._lowlevel_server.add_request_handler('tools/list', PaginatedRequestParams, list_tools_page)


if __name__ == '__main__':
    mcp.run(transport='streamable-http', stateless_http=True, port=_PORT)
