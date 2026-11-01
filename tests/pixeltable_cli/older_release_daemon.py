"""Serve /api/health like a daemon from a release that predates packages_fingerprint.

The test environment cannot run an older release, so this script stands in for one. Usage:
older_release_daemon.py PORT PROJECT_ROOT
"""

import http.server
import json
import os
import sys
from typing import Any

from pixeltable_cli.utils import identity

body: dict[str, Any] = identity()
del body['packages_fingerprint']
body |= {'ok': True, 'service': 'pxt', 'pid': os.getpid(), 'started_at': '', 'project_root': sys.argv[2]}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


server = http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler)
print('ready', flush=True)
server.serve_forever()
