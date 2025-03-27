# This is a MCP-based server that provides an interface to query JFK files using SSE (Server-Sent Events)
# It demonstrates how to set up a real-time communication channel between the client and server

import argparse

import uvicorn
from initialize import mcp
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can server the provied mcp server with SSE."""
    # Set up SSE endpoint for real-time messaging
    sse = SseServerTransport('/messages/')

    async def handle_sse(request: Request) -> None:
        # Establish SSE connection and handle bi-directional communication
        async with sse.connect_sse(request.scope, request.receive, request._send) as (read_stream, write_stream):
            await mcp_server.run(read_stream, write_stream, mcp_server.create_initialization_options())

    # Define routes: /sse for establishing connections, /messages/ for handling message posts
    return Starlette(
        debug=debug, routes=[Route('/sse', endpoint=handle_sse), Mount('/messages/', app=sse.handle_post_message)]
    )


if __name__ == '__main__':
    # Initialize the MCP server for handling queries
    mcp_server = mcp._mcp_server

    # Set up command-line arguments for server configuration
    parser = argparse.ArgumentParser(description='Run MCP SSE-based server')

    # Add custom Host and Port arguments (e.g. python server.py --host 0.0.0.0 --port 8083)
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8083, help='Port to listen on')
    args = parser.parse_args()

    # Start the server with the specified configuration
    starlette_app = create_starlette_app(mcp_server, debug=True)
    uvicorn.run(starlette_app, host=args.host, port=args.port)
