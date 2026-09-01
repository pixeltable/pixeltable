# This module intentionally omits from __future__ import annotations: the served application's route
# handlers carry annotations that FastAPI resolves at import time.

import argparse
import logging
from pathlib import Path

from pixeltable.config import Config
from pixeltable.service.db import report_instance_fingerprint, unpack_project_archive
from pixeltable.serving._app import create_app, init_instrumentation, instrument_app


def _serve(
    db_uri: str | None,
    app_file: str,
    service_name: str,
    base_path: str,
    project_dir: Path,
    digest: str | None,
    host: str,
    port: int,
    otel: bool,
) -> None:
    """Pod entry point: serve one of the project's services out of project_dir, and report what loaded.

    Without db_uri the project is already in project_dir, put there by the image the pod runs.
    TODO: fetch unconditionally, once every database holds an archive apart from its image.
    """
    import uvicorn

    if db_uri is not None:
        unpack_project_archive(db_uri, project_dir, expected_digest=digest)
    # the project is this process's project root, so its modules and its database entry resolve
    Config.init(reinit=True, project_root=project_dir)

    if otel:
        # before the first Pixeltable operation, so that loading the file is traced too
        init_instrumentation()
    app, _ = create_app(str(project_dir / app_file), service_name, base_path)
    if otel:
        instrument_app(app)
    if db_uri is not None:
        # after the file has loaded, so the fingerprint names the serving files
        # TODO: report unconditionally, once the control plane records what an instance loaded
        report_instance_fingerprint(db_uri, service_name, base_path)

    log_level = logging.getLogger('pixeltable').getEffectiveLevel()
    # log_config=None keeps uvicorn from replacing the logging Env has already set up
    uvicorn.run(app, host=host, port=port, log_level=log_level, log_config=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='pixeltable.serving.pod_runner')
    parser.add_argument('--db', help="pxt://org:db; when given, the pod fetches that database's project")
    parser.add_argument('--app-file', required=True, help='path to the application file, from the project root')
    parser.add_argument('--name', required=True, help='the service to serve')
    parser.add_argument('--base-path', default='')
    parser.add_argument('--project-dir', type=Path, required=True, help='unpack the project here')
    parser.add_argument('--digest', help='refuse a project other than this one')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--otel', action='store_true')
    args = parser.parse_args()
    _serve(
        args.db,
        args.app_file,
        args.name,
        args.base_path,
        args.project_dir,
        args.digest,
        args.host,
        args.port,
        args.otel,
    )
