#!/usr/bin/env python3
"""
Start the Pixeltable Remote Server locally for testing.

Usage:
    python start_remote_server.py [--port PORT] [--host HOST]

Default: http://localhost:8000
"""

import argparse
import sys
from pathlib import Path

# Add pixeltable to path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn

from pixeltable.share.remote import app


def main() -> None:
    parser = argparse.ArgumentParser(description='Start Pixeltable Remote Server')
    parser.add_argument('--port', type=int, default=8000, help='Port to run on (default: 8000)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    args = parser.parse_args()

    print(f'Starting Pixeltable Remote Server on http://{args.host}:{args.port}')

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level='info',
            reload=False,  # Disable reload for stability
        )
    except KeyboardInterrupt:
        print('\nServer stopped by user')
    except Exception as e:
        print(f'\nServer error: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
