"""ModalServer, which serves one UDF of the pixeltable.udf setting in the containers of a Modal Server.

- A Modal App has one Server per UDF. The pixeltable.udf setting gives the resources of each Server's containers.
- Each container runs a ModalServer, which serves the UDF's calls over HTTP.
- ModalServerClient (pixeltable.exec.expr_eval.modal_client) POSTs each call to Modal's proxy. The proxy checks the
  request's proxy token and forwards the request over HTTP/2 to a container of the UDF's Server.

Why the App uses Servers instead of Modal functions (.remote()):
- A Server call takes one round trip through Modal's proxy. A .remote() call takes two RPCs and moves an argument
  above 2 MB through Modal's blob storage.
- With the Servers in us-east, an echo call took 74 ms at p50, against 153-165 ms with .remote(), and a call with
  a 1 MB argument took 223 ms, against 824 ms.
- Server calls need only a proxy token, which can call Servers but can't deploy Apps; .remote() calls need a Modal
  API token.
"""

import os

import modal

from .udf_server import UdfServer

# Modal's proxy forwards requests to this port of a Server's containers
PORT = 8000
# Servers take no parameters, so the container gets the UDF path from this environment variable
UDF_PATH_VAR = 'PXT_UDF_PATH'


class ModalServer(UdfServer):
    """Runs one UDF's calls in a container of the UDF's Server."""

    @modal.enter()
    def enter(self) -> None:
        self.start(os.environ[UDF_PATH_VAR], PORT, os.environ.get('MODAL_TASK_ID'))
