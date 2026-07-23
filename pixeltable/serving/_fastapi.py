from __future__ import annotations

import copy
import inspect
import logging
import mimetypes
import os
import shutil
import threading
import urllib.parse
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from typing import _GenericAlias  # type: ignore[attr-defined]  # isort: skip

import fastapi
import numpy as np
import PIL.Image
import pydantic
import sqlalchemy as sql
from fastapi import Body, File, Form, HTTPException, Query as QueryParam, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic.fields import FieldInfo

import pixeltable as pxt
from pixeltable import catalog, exceptions as excs, exprs, func, type_system as ts
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.exec.globals import INLINED_OBJECT_MD_KEY
from pixeltable.serving import SqlExport
from pixeltable.serving.globals import SqlExporter
from pixeltable.utils import image as image_utils
from pixeltable.utils.http import fetch_url
from pixeltable.utils.local_store import LocalStore, TempStore

_logger = logging.getLogger(__name__)


class BackgroundJobResponse(pydantic.BaseModel):
    id: str
    job_url: str


class DeleteResponse(pydantic.BaseModel):
    """Response from a delete endpoint."""

    num_rows: int


class JobStatusResponse(pydantic.BaseModel):
    """Status of a background job."""

    status: Literal['pending', 'done', 'error']

    # only set for status == 'error'
    error: Optional[str] = None

    # the per-route response_model produced by add_insert_route(); typed as Any since it varies by route
    # only set for status == 'done'
    result: Optional[Any] = None


# Name used to register the /media/{path:path} route. Insert routes use this name with
# request.url_for(_MEDIA_ROUTE_NAME, path=...) to build absolute media URLs at request time.
_MEDIA_ROUTE_NAME = 'pxt_serve_media'
_JOB_STATUS_ROUTE_NAME = 'pxt_serve_job_status'

_EMBEDDED_OBJECT_TYPES: tuple[type, ...] = (np.ndarray, np.generic, PIL.Image.Image, bytes)


def _validate_registered_schema(t: pxt.Table, schema_version: int) -> None:
    """Raise 409 if the table's schema changed since the route was registered.

    The route holds the table handle captured at registration (handles are thread-safe). A schema bump shows up
    as a different schema_version; a drop (or drop-and-recreate at the same path under a new id) makes the
    captured handle's metadata lookup raise TABLE_NOT_FOUND. Both mean the frozen request/response contract is
    stale, so the caller should restart the service.
    """
    try:
        changed = t.get_metadata()['schema_version'] != schema_version
    except pxt.Error as exc:
        if exc.error_code is not pxt.ErrorCode.TABLE_NOT_FOUND:
            raise
        changed = True
    if changed:
        raise HTTPException(
            status_code=409, detail='table schema changed since route was registered; please restart the service'
        )


def _check_route_output_schema(output_cols: list[catalog.ColumnVersionMd], error_prefix: str) -> None:
    """Reject output column types whose values can't be served by FastAPI today.

    Array and Binary cells materialize into coalesced side files that the /media route can't serve as single units,
    or come through compute as raw ndarray/bytes that can't be serialized to json.
    """
    for col in output_cols:
        if col.col_type.is_array_type() or col.col_type.is_binary_type():
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                (
                    f'{error_prefix}: output column {col.name!r} has type {col.col_type._to_base_str()}, '
                    'which is not supported'
                ),
            )


def _check_json_value_servable(val: Any, col_name: str) -> None:
    """
    Raise HTTPException if a json cell value contains content the FastAPI layer can't serve
    (embedded references to side files, embedded objects that can't be serialized to json).
    """
    if isinstance(val, dict):
        if INLINED_OBJECT_MD_KEY in val:
            raise HTTPException(
                status_code=500,
                detail=(
                    f'output column {col_name!r}: JSON value contains an embedded array/binary/image, '
                    'which is not supported.'
                ),
            )
        for v in val.values():
            _check_json_value_servable(v, col_name)
    elif isinstance(val, list):
        for v in val:
            _check_json_value_servable(v, col_name)
    elif isinstance(val, _EMBEDDED_OBJECT_TYPES):
        # Report the base PIL type ('Image'), not the concrete decoder subclass (e.g. PngImageFile), whose
        # name depends on how the image was decoded.
        type_name = 'Image' if isinstance(val, PIL.Image.Image) else type(val).__name__
        raise HTTPException(
            status_code=500,
            detail=(
                f'output column {col_name!r}: JSON value contains an embedded {type_name}, which is not supported.'
            ),
        )


def _resolve_type_hints(user_fn: Callable, fn_name: str, error_prefix: str) -> dict[str, Any]:
    """Resolve PEP-563 string annotations (from __future__ import annotations) and forward refs."""
    try:
        return get_type_hints(user_fn)
    except NameError as e:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT, f'{error_prefix}: {fn_name!r}: cannot resolve type annotations: {e}'
        ) from e


def _validate_type_hint(
    annot: type | _GenericAlias, col_type: ts.ColumnType, *, subject: str, error_prefix: str
) -> None:
    """Check that a type annotation is interpretable and compatible with a ColumnType."""
    annot_col_type = ts.ColumnType.from_python_type(annot)
    if annot_col_type is None:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_TYPE,
            f'{error_prefix}: {subject}: cannot interpret annotation {annot!r} as a Pixeltable type',
        )

    # media values are served as /media URL strings
    expected_col_type = ts.StringType(nullable=col_type.nullable) if col_type.is_media_type() else col_type
    if not annot_col_type.is_supertype_of(expected_col_type):
        raise pxt.RequestError(
            pxt.ErrorCode.TYPE_MISMATCH,
            f'{error_prefix}: {subject} has annotation {annot!r} '
            f'({annot_col_type}), which is incompatible with column type {expected_col_type}',
        )


def _validated_response_model(hints: dict[str, Any], fn_name: str, error_prefix: str) -> type[pydantic.BaseModel]:
    """The return annotation of a decorated route function, which must be a pydantic model."""
    return_annot = hints.get('return')
    if not (isinstance(return_annot, type) and issubclass(return_annot, pydantic.BaseModel)):
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT,
            f'{error_prefix}: {fn_name!r} must have a return annotation that is a pydantic.BaseModel subclass; '
            f'got {return_annot!r}',
        )
    return return_annot


def _flush_image(val: Any) -> Any:
    """Flush a PIL image to a temp file and return its local file:// uri; pass other values through."""
    if not isinstance(val, PIL.Image.Image):
        return val
    fmt = image_utils.default_format(val)
    dest = TempStore.create_path(extension=f'.{fmt}')
    val.save(dest, format=fmt)
    return dest.as_uri()


def _exactly_one_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The single row of an insert/update result; 500 on any other count."""
    if len(rows) != 1:
        raise HTTPException(status_code=500, detail=f'operation returned unexpected row count ({len(rows)})')
    return rows[0]


def _single_computed_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The single row of a compute() result, for response shapes that require exactly one row.

    404 on an empty result (the input row was dropped by a view's filter); 500 if the target view fans the
    input row out to multiple rows.
    """
    if len(rows) == 0:
        raise HTTPException(status_code=404, detail='compute produced no output row (dropped by a view filter)')
    if len(rows) > 1:
        raise HTTPException(status_code=500, detail=f'expected a single output row, got {len(rows)}')
    return rows[0]


# ColumnType.Type -> json-Schema contentMediaType
# .../*: tell OpenAPI tooling to render the URL as a media link without committing to a specific subtype
_MEDIA_CONTENT_TYPES: dict[ts.ColumnType.Type, str] = {
    ts.ColumnType.Type.IMAGE: 'image/*',
    ts.ColumnType.Type.VIDEO: 'video/*',
    ts.ColumnType.Type.AUDIO: 'audio/*',
    ts.ColumnType.Type.DOCUMENT: 'application/octet-stream',
}


T = TypeVar('T')


def _run_endpoint_op(
    endpoint_op: Callable[..., T], kwargs: dict[str, Any], tmp_paths: list[Path], url_for_media: Callable[[str], str]
) -> T:
    try:
        return endpoint_op(kwargs, url_for_media)
    except Exception as e:
        for p in tmp_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                _logger.warning(f'failed to delete tmp file: {p}', exc_info=True)
        if isinstance(e, pxt.Error):
            raise HTTPException(status_code=e.http_status, detail=e.to_dict()) from e
        raise


class PxtEndpoint:
    """
    Wrapper for an endpoint `Callable` that carries additional metadata about the endpoint operation.
    """

    router: 'FastAPIRouter'
    uploadfile_inputs: list[str]
    background: bool
    endpoint_op: Callable[..., Any]
    tbl: pxt.Table | None
    route_type: Literal['insert', 'update', 'delete', 'query', 'compute']

    def __init__(
        self,
        router: 'FastAPIRouter',
        name: str,
        signature: inspect.Signature,
        uploadfile_inputs: list[str],
        background: bool,
        endpoint_op: Callable[..., Any],
        tbl: pxt.Table | None,
        route_type: Literal['insert', 'update', 'delete', 'query', 'compute'],
    ) -> None:
        self.router = router
        self.uploadfile_inputs = uploadfile_inputs
        self.background = background
        self.endpoint_op = endpoint_op
        self.tbl = tbl
        self.route_type = route_type

        # FastAPI needs the correct signature and a name
        self.__signature__ = signature
        self.__name__ = name

    def __call__(self, request: Request, **kwargs: Any) -> Any:
        sample_url = str(request.url_for(_MEDIA_ROUTE_NAME, path='_'))
        media_url_base = sample_url[:-1]

        def url_for_media(rel_path: str) -> str:
            return f'{media_url_base}{urllib.parse.quote(rel_path, safe="/")}'

        # write out uploads while the request is still alive
        tmp_paths: list[Path] = []
        if len(self.uploadfile_inputs) > 0:
            # list(...): make sure that the sequence of name/val pairs can't change underneath us
            for input_name, val in list(kwargs.items()):
                if input_name in self.uploadfile_inputs:
                    path = self.router._write_to_temp(val)
                    tmp_paths.append(path)
                    kwargs[input_name] = str(path)

        if self.background:
            job_id = uuid.uuid4().hex
            fut = self.router._executor.submit(_run_endpoint_op, self.endpoint_op, kwargs, tmp_paths, url_for_media)
            with self.router._jobs_lock:
                self.router._jobs[job_id] = fut
            return BackgroundJobResponse(id=job_id, job_url=str(request.url_for(_JOB_STATUS_ROUTE_NAME, job_id=job_id)))
        else:
            return _run_endpoint_op(self.endpoint_op, kwargs, tmp_paths, url_for_media)


class FastAPIRouter(fastapi.APIRouter):
    """
    A FastAPI `APIRouter` that exposes Pixeltable table operations as HTTP endpoints.

    `FastAPIRouter` is for apps that already have a FastAPI server. If you do
    not have one, use `pxt serve` from the CLI; Pixeltable creates and runs the
    FastAPI app for you. Learn more here: [HTTP Serving](https://docs.pixeltable.com/howto/deployment/serving).
    """

    _executor: ThreadPoolExecutor
    _jobs: dict[str, Future]  # holds background requests; key: job id (uuid4().hex)
    _jobs_lock: threading.Lock
    _home_dir: Path
    _allowed_media_dirs: list[Path]
    _engine_cache: dict[str, sql.Engine]  # keyed by SqlExport.db_connect; shared across routes

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix='pxt-serve-background')
        self._jobs = {}
        self._jobs_lock = threading.Lock()
        self._home_dir = Config.get().home.resolve()
        self._allowed_media_dirs = [
            Env.get().media_dir.resolve(),
            Env.get().tmp_dir.resolve(),
            Env.get().file_cache_dir.resolve(),
        ]
        self._engine_cache = {}
        self._register_media_route()
        self._register_jobs_route()
        # Shut down the worker pool when the parent app's lifespan ends. include_router()
        # merges this handler into the app's on_shutdown list, so it fires on app shutdown.
        self.add_event_handler('shutdown', self.__shutdown)

    def add_api_route(self, path: str, *args: Any, **kwargs: Any) -> None:
        """Wrap FastAPI's add_api_route with a duplicate (path, method) check."""
        # FastAPI's APIRoute normalizes methods to uppercase; match its contract.
        new_methods = {m.upper() for m in (kwargs.get('methods') or ['GET'])}
        # FastAPI stores routes under self.prefix + path; compare against the prefixed form
        prefixed_path = self.prefix + path
        for route in self.routes:
            if not isinstance(route, fastapi.routing.APIRoute) or route.path != prefixed_path:
                continue
            if len(overlap := route.methods & new_methods) == 0:
                continue
            conflict = ', '.join(sorted(overlap))
            raise excs.AlreadyExistsError(
                excs.ErrorCode.PATH_ALREADY_EXISTS, f'route already registered: {conflict} {prefixed_path!r}'
            )
        super().add_api_route(path, *args, **kwargs)

    def __shutdown(self) -> None:
        # wait until in-flight requests are done and won't access _engine_cache
        self._executor.shutdown(wait=True, cancel_futures=True)
        for eng in self._engine_cache.values():
            eng.dispose()
        self._engine_cache.clear()

    def _get_sql_engine(self, db_connect: str) -> sql.Engine:
        eng = self._engine_cache.get(db_connect)
        if eng is None:
            eng = sql.create_engine(db_connect)
            self._engine_cache[db_connect] = eng
        return eng

    def add_compute_route(
        self,
        t: pxt.Table,
        *,
        path: str,
        inputs: list[str] | None = None,
        uploadfile_inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        return_fileresponse: bool = False,
        export_sql: SqlExport | None = None,
        background: bool = False,
    ) -> None:
        """
        Add a POST endpoint that materializes the computed columns of `t` and returns the resulting rows.

        The endpoint runs [`Table.compute()`][pixeltable.Table.compute] on the request body
        and returns the materialized rows as a JSON array, without persisting them. This is
        identical to [`add_insert_route()`][pixeltable.serving.FastAPIRouter.add_insert_route],
        except that no data is actually inserted into the table.

        `t` may be a view; the request body then carries a row for the view's base table, and the
        response contains the view's output rows for that input row:

        - an input row that doesn't satisfy the view's filter produces an empty array
        - an iterator view can produce multiple output rows per input row

        Args:
            t: The table or view to compute rows over.
            path: The URL path for the endpoint.
            inputs: Columns to accept as request fields. Defaults to all non-computed columns
                (of the base table, if `t` is a view).
            uploadfile_inputs: Columns to accept as
                [`UploadFile`](https://fastapi.tiangolo.com/tutorial/request-files/) fields
                (must be media-typed). These are sent as multipart form data; all other inputs
                become [`Form`](https://fastapi.tiangolo.com/tutorial/request-forms/) fields.
            outputs: Columns to include in the response. Defaults to all columns (including inputs).
            return_fileresponse: If True, return the single media-typed output column as a
                [`FileResponse`](https://fastapi.tiangolo.com/advanced/custom-response/#fileresponse).
                Requires exactly one media-typed output column, and the computation must produce
                exactly one row: the endpoint returns 404 if the input row is dropped by a view's
                filter, and an error if it fans out to multiple rows.
            export_sql: If set, export each computed row into an external RDBMS table after the
                computation succeeds. See [`SqlExport`][pixeltable.serving.SqlExport] for the
                target specification and supported `method` values. The exported rows are the
                response body. See [`add_insert_route()`][pixeltable.serving.FastAPIRouter.add_insert_route]
                for the schema-compatibility rules.
            background: If True, return immediately with `{"id": ..., "job_url": ...}` and run
                the computation in a background thread. Poll `job_url` for the result.

        Examples:
            JSON request/response (no row stored):

            >>> router.add_compute_route(t, path='/preview', inputs=['prompt'], outputs=['result'])

            ```bash
            curl -X POST http://localhost:8000/preview \\
              -H 'Content-Type: application/json' \\
              -d '{"prompt": "a sunset over the ocean"}'
            # [{"prompt": "a sunset over the ocean", "result": "..."}]
            ```

            Compute the rows of a view `frames` (defined over `t` with a `FrameIterator`) for a
            given base table row:

            >>> router.add_compute_route(frames, path='/frames', outputs=['pos', 'frame'])

            ```bash
            curl -X POST http://localhost:8000/frames \\
              -H 'Content-Type: application/json' \\
              -d '{"video": "http://example.com/clip.mp4"}'
            # [{"pos": 0, "frame": "http://.../media/..."}, {"pos": 1, "frame": "..."}, ...]
            ```

            File upload with `FileResponse`:

            >>> router.add_compute_route(
            ...     t, path='/resize', inputs=['width', 'height'],
            ...     uploadfile_inputs=['image'], outputs=['resized'], return_fileresponse=True,
            ... )

            ```bash
            curl -X POST http://localhost:8000/resize \\
              -F image=@photo.jpg -F width=640 -F height=480 \\
              --output resized.jpg
            # saves the resized image to resized.jpg; no row is inserted into `t`
            ```

            Export each computed row into an external RDBMS table:

            >>> router.add_compute_route(
            ...     t,
            ...     path='/preview',
            ...     inputs=['prompt'],
            ...     outputs=['prompt', 'result'],
            ...     export_sql=SqlExport(
            ...         db_connect='postgresql+psycopg://user:pw@host/analytics',
            ...         table='previews',
            ...     ),
            ... )

            Each successful POST computes the rows and appends them (columns: `prompt`, `result`)
            to `public.previews`. The Pixeltable table is not modified.

            Background processing:

            >>> router.add_compute_route(t, path='/slow', background=True)

            ```bash
            # submit
            curl -X POST http://localhost:8000/slow -d '{"prompt": "hello"}'
            # {"id": "abc123", "job_url": "http://localhost:8000/jobs/abc123"}
            ```

            ```bash
            # poll
            curl http://localhost:8000/jobs/abc123
            # {"status": "done", "result": {...}}
            ```
        """
        self._add_insert_compute_route(
            t,
            path=path,
            inputs=inputs,
            uploadfile_inputs=uploadfile_inputs,
            outputs=outputs,
            return_fileresponse=return_fileresponse,
            export_sql=export_sql,
            background=background,
            route_type='compute',
        )

    def add_insert_route(
        self,
        t: pxt.Table,
        *,
        path: str,
        inputs: list[str] | None = None,
        uploadfile_inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        return_fileresponse: bool = False,
        export_sql: SqlExport | None = None,
        background: bool = False,
    ) -> None:
        """
        Add a POST endpoint that inserts a single row into `t` and returns the resulting row.

        The request body contains the input column values as JSON fields (or as
        [multipart form data](https://fastapi.tiangolo.com/tutorial/request-files/) when
        `uploadfile_inputs` is used). The response is a JSON object with the output column values,
        or a [`FileResponse`](https://fastapi.tiangolo.com/advanced/custom-response/#fileresponse)
        when `return_fileresponse=True`.

        Args:
            t: The table to insert into.
            path: The URL path for the endpoint.
            inputs: Columns to accept as request fields. Defaults to all non-computed columns.
            uploadfile_inputs: Columns to accept as
                [`UploadFile`](https://fastapi.tiangolo.com/tutorial/request-files/) fields
                (must be media-typed). These are sent as multipart form data; all other inputs
                become [`Form`](https://fastapi.tiangolo.com/tutorial/request-forms/) fields.
            outputs: Columns to include in the response. Defaults to all columns (including inputs).
            return_fileresponse: If True, return the single media-typed output column as a
                [`FileResponse`](https://fastapi.tiangolo.com/advanced/custom-response/#fileresponse).
                Requires exactly one media-typed output column.
            export_sql: If set, export each inserted row into an external RDBMS table after the
                Pixeltable insert succeeds. See [`SqlExport`][pixeltable.serving.SqlExport] for
                the target specification and supported `method` values.

                The row written is the response body: same columns as `outputs`, with media-typed
                columns rendered as URL strings (so the corresponding target columns must be
                string-typed).

                Schema compatibility against the response columns is validated once at
                registration time; the target table must already exist or registration fails.
                Mutually exclusive with `return_fileresponse`. Compatible with `background=True`
                (the SQL write runs in the worker thread).

                Note: when paired with `method='update'`, a Pixeltable insert triggers a
                target-side update -- this is intentional, supporting the append-only-source /
                current-state-view pattern.

                If the external write fails after the Pixeltable insert has already succeeded,
                the request fails with HTTP 500; no rollback is performed.
            background: If True, return immediately with `{"id": ..., "job_url": ...}` and run
                the insert in a background thread. Poll `job_url` for the result. Mutually
                exclusive with `return_fileresponse`.

        Examples:
            JSON request/response:

            >>> router.add_insert_route(t, path='/generate', inputs=['prompt'], outputs=['result'])

            ```bash
            curl -X POST http://localhost:8000/generate \\
              -H 'Content-Type: application/json' \\
              -d '{"prompt": "a sunset over the ocean"}'
            # {"prompt": "a sunset over the ocean", "result": "..."}
            ```

            File upload with `FileResponse`:

            >>> router.add_insert_route(
            ...     t, path='/resize', inputs=['width', 'height'],
            ...     uploadfile_inputs=['image'], outputs=['resized'], return_fileresponse=True,
            ... )

            ```bash
            curl -X POST http://localhost:8000/resize \\
              -F image=@photo.jpg -F width=640 -F height=480 \\
              --output resized.jpg
            # saves the resized image to resized.jpg
            ```

            Export each inserted row into an external RDBMS table:

            >>> router.add_insert_route(
            ...     t,
            ...     path='/generate',
            ...     inputs=['prompt'],
            ...     outputs=['prompt', 'result'],
            ...     export_sql=SqlExport(
            ...         db_connect='postgresql+psycopg://user:pw@host/analytics',
            ...         table='generations',
            ...         db_schema='public',
            ...     ),
            ... )

            Each successful POST inserts a row into the Pixeltable table and then inserts the
            same row (columns: `prompt`, `result`) to `public.generations` in the target database.

            Background processing:

            >>> router.add_insert_route(t, path='/slow', background=True)

            ```bash
            # submit
            curl -X POST http://localhost:8000/slow -d '{"prompt": "hello"}'
            # {"id": "abc123", "job_url": "http://localhost:8000/jobs/abc123"}
            ```

            ```bash
            # poll
            curl http://localhost:8000/jobs/abc123
            # {"status": "done", "result": {...}}
            ```
        """
        self._add_insert_compute_route(
            t,
            path=path,
            inputs=inputs,
            uploadfile_inputs=uploadfile_inputs,
            outputs=outputs,
            return_fileresponse=return_fileresponse,
            export_sql=export_sql,
            background=background,
            route_type='insert',
        )

    def _add_insert_compute_route(
        self,
        t: pxt.Table,
        *,
        path: str,
        inputs: list[str] | None,
        uploadfile_inputs: list[str] | None,
        outputs: list[str] | None,
        return_fileresponse: bool,
        export_sql: SqlExport | None,
        background: bool,
        route_type: Literal['insert', 'compute'],
    ) -> None:
        """Shared implementation of add_insert_route()/add_compute_route()."""
        error_prefix = f'add_{route_type}_route()'
        _, input_col_names, output_col_names, cols_by_name = self._validate_dml_args(
            t,
            inputs=inputs,
            uploadfile_inputs=uploadfile_inputs,
            outputs=outputs,
            return_fileresponse=return_fileresponse,
            background=background,
            error_prefix=error_prefix,
            route_type=route_type,
        )
        uploadfile_inputs = uploadfile_inputs or []
        output_cols = [cols_by_name[name] for name in output_col_names]
        _check_route_output_schema(output_cols, error_prefix)
        json_output_col_names = [n for n in output_col_names if cols_by_name[n].col_type.is_json_type()]

        sql_exporter = self._make_schema_sql_exporter(
            export_sql,
            return_fileresponse=return_fileresponse,
            schema={name: cols_by_name[name].col_type for name in output_col_names},
            error_prefix=error_prefix,
        )

        path_str = ''.join(el.capitalize() for el in path.split('/') if len(el) > 0)
        response_model = self._create_model(f'{path_str}Response', output_cols=output_cols)

        def rows_processor(rows: Sequence[Mapping[str, Any]], url_for_media: Callable[[str], str]) -> Any:
            if route_type == 'insert':
                rows = [_exactly_one_row(rows)]
            elif return_fileresponse:
                # a FileResponse is a single file; a view can drop the input row or fan it out
                rows = [_single_computed_row(rows)]
            for row in rows:
                for name in json_output_col_names:
                    _check_json_value_servable(row.get(name), name)
            output = self._create_output(rows, output_col_names, response_model, return_fileresponse, url_for_media)
            if return_fileresponse:
                return output
            assert isinstance(output, list)
            if sql_exporter is not None:
                for result in output:
                    assert isinstance(result, pydantic.BaseModel)
                    sql_exporter.export_row(result)
            # a compute response is an array: a view can produce zero or multiple output rows per input row
            return output if route_type == 'compute' else output[0]

        self._add_dml_route(
            t,
            path=path,
            pk_col_names=[],
            input_col_names=input_col_names,
            uploadfile_inputs=uploadfile_inputs,
            return_fileresponse=return_fileresponse,
            background=background,
            endpoint_name=f'{route_type}_{path.strip("/").replace("/", "_") or "root"}',
            rows_processor=rows_processor,
            response_model=list[response_model] if route_type == 'compute' else response_model,  # type: ignore[valid-type]
            route_type=route_type,
        )

    def compute_route(
        self,
        t: pxt.Table,
        *,
        path: str,
        inputs: list[str] | None = None,
        uploadfile_inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        export_sql: SqlExport | None = None,
        background: bool = False,
    ) -> Callable[[Callable[..., pydantic.BaseModel]], Callable[..., pydantic.BaseModel]]:
        """
        Decorator that registers a POST endpoint computing the workflow given by `t` and post-processing the result.

        The endpoint runs [`Table.compute()`][pixeltable.Table.compute] on the request body
        and passes the materialized rows to the decorated function, whose return value (a pydantic
        model) is the HTTP response. No data is persisted; use this for
        "what would an insert produce?" workflows.

        `t` may be a view; the request body then carries a row for the view's base table, and the
        decorated function receives the view's output rows for that input row (zero if the view's
        filter drops it, several if an iterator view fans it out).

        The decorated function can take one of two forms:

        - Per-column: one keyword-only parameter per output column (`def fn(*, caption: str, ...)`).
          The computation must produce exactly one row; the endpoint returns 404 if the input row
          is dropped by a view's filter, and an error if it fans out to multiple rows.
        - Batch: a single parameter annotated `list[M]`, with `M` a pydantic model that has one
          field per output column (`def fn(rows: list[FrameRow])`). The function is called with all
          output rows -- including an empty list -- and can aggregate them into a single response.

        In both forms, media-typed outputs (image, video, audio, document) are delivered as `/media/`
        URL strings -- annotate them as `str` (or `str | None` if the column is nullable).

        Args:
            t: The table or view to compute rows over.
            path: The URL path for the endpoint.
            inputs: Columns to accept as request fields. Defaults to all non-computed columns
                (of the base table, if `t` is a view).
            uploadfile_inputs: Columns to accept as
                [`UploadFile`](https://fastapi.tiangolo.com/tutorial/request-files/) fields
                (must be media-typed). These are sent as multipart form data; all other inputs
                become [`Form`](https://fastapi.tiangolo.com/tutorial/request-forms/) fields.
            outputs: Columns from the computed rows to pass to the decorated function.
                Defaults to all columns.
            export_sql: If set, export the decorated function's return value into an external
                RDBMS table after the computation succeeds. See
                [`SqlExport`][pixeltable.serving.SqlExport] for the target specification and
                supported `method` values. See
                [`@insert_route`][pixeltable.serving.FastAPIRouter.insert_route] for the schema-compatibility rules.
            background: If True, return immediately with `{"id": ..., "job_url": ...}` and run
                the computation plus post-processing in a background thread. Poll `job_url` for the
                result; the decorated function's return value is delivered as the job result.

        Examples:
            Per-column form:

            >>> class PreviewResponse(pydantic.BaseModel):
            ...     caption: str
            ...     score: float
            ...
            ... @router.compute_route(
            ...     t, path='/preview', inputs=['prompt'], outputs=['caption', 'score'],
            ... )
            ... def format_response(*, caption: str, score: float) -> PreviewResponse:
            ...     return PreviewResponse(caption=caption.strip(), score=round(score, 3))

            ```bash
            curl -X POST http://localhost:8000/preview \\
              -H 'Content-Type: application/json' \\
              -d '{"prompt": "a sunset over the ocean"}'
            # {"caption": "orange sky above calm water", "score": 0.932}
            ```

            Nothing is inserted into `t`; the computed row is passed only to `format_response`.

            Batch form, aggregating the rows of a view `frames` (defined over `t` with a
            `FrameIterator`) for a given base table row:

            >>> class FrameRow(pydantic.BaseModel):
            ...     pos: int
            ...     frame: str
            ...
            ... class FramesResponse(pydantic.BaseModel):
            ...     n_frames: int
            ...     frame_urls: list[str]
            ...
            ... @router.compute_route(frames, path='/frames', outputs=['pos', 'frame'])
            ... def summarize(rows: list[FrameRow]) -> FramesResponse:
            ...     return FramesResponse(n_frames=len(rows), frame_urls=[r.frame for r in rows])

            Export the post-processed response into an external RDBMS table:

            >>> @router.compute_route(
            ...     t, path='/preview', inputs=['prompt'], outputs=['caption', 'score'],
            ...     export_sql=SqlExport(
            ...         db_connect='postgresql+psycopg://user:pw@host/analytics',
            ...         table='preview_captions',
            ...     ),
            ... )
            ... def format_response(*, caption: str, score: float) -> PreviewResponse:
            ...     return PreviewResponse(caption=caption.strip(), score=round(score, 3))

            Each successful POST computes the row and appends the response (fields `caption`,
            `score`) to `preview_captions`. The Pixeltable table is not modified.
        """
        return self._insert_compute_route(
            t,
            path=path,
            inputs=inputs,
            uploadfile_inputs=uploadfile_inputs,
            outputs=outputs,
            export_sql=export_sql,
            background=background,
            route_type='compute',
        )

    def insert_route(
        self,
        t: pxt.Table,
        *,
        path: str,
        inputs: list[str] | None = None,
        uploadfile_inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        export_sql: SqlExport | None = None,
        background: bool = False,
    ) -> Callable[[Callable[..., pydantic.BaseModel]], Callable[..., pydantic.BaseModel]]:
        """
        Decorator that registers a POST endpoint performing a `Table.insert()` followed by user-defined post-processing.

        The request body carries the input column values (JSON, or multipart form data when `uploadfile_inputs` is
        used). After inserting the row, the decorated function is called with the requested output columns as
        keyword arguments (parameter names and Pixeltable types must match `outputs`). Its return value must be a
        Pydantic model and is returned as the HTTP response body.

        Media-typed outputs (image, video, audio, document) are delivered to the function as `/media/` URL
        strings -- annotate those parameters as `str` (or `str | None` if the column is nullable), not as
        `pxt.Image` / `pxt.Video` / etc.

        Args:
            t: The table to insert into.
            path: The URL path for the endpoint.
            inputs: Columns to accept as request fields. Defaults to all non-computed columns.
            uploadfile_inputs: Columns to accept as
                [`UploadFile`](https://fastapi.tiangolo.com/tutorial/request-files/) fields
                (must be media-typed). These are sent as multipart form data; all other inputs
                become [`Form`](https://fastapi.tiangolo.com/tutorial/request-forms/) fields.
            outputs: Columns from the inserted row to pass to the decorated function as keyword
                arguments. Defaults to all columns.
            export_sql: If set, export the decorated function's return value into an external
                RDBMS table after the Pixeltable insert succeeds. See
                [`SqlExport`][pixeltable.serving.SqlExport] for the target specification and
                supported `method` values.

                The row written is the user function's pydantic return value (its fields, not the
                source columns), so the target table schema must match those fields. Media-typed
                fields are modeled as strings (URL form).

                Schema compatibility is validated once at registration time; the target table
                must already exist or registration fails. Compatible with `background=True` (the
                SQL write runs in the worker thread).

                If the external write fails after the Pixeltable insert has already succeeded,
                the request fails with HTTP 500; no rollback is performed.
            background: If True, return immediately with `{"id": ..., "job_url": ...}` and run
                the insert plus post-processing in a background thread. Poll `job_url` for the
                result; the decorated function's return value is delivered as the job result.

        Examples:
            >>> class GenerateResponse(pydantic.BaseModel):
            ...     caption: str
            ...     score: float
            ...
            ... @router.insert_route(
            ...     t, path='/generate', inputs=['prompt'], outputs=['caption', 'score'], background=False
            ... )
            ... def format_response(*, caption: str, score: float) -> GenerateResponse:
            ...     return GenerateResponse(caption=caption.strip(), score=round(score, 3))

            ```bash
            curl -X POST http://localhost:8000/generate \\
              -H 'Content-Type: application/json' \\
              -d '{"prompt": "a sunset over the ocean"}'
            # {"caption": "orange sky above calm water", "score": 0.932}
            ```

            Export the post-processed response into an external RDBMS table:

            >>> @router.insert_route(
            ...     t, path='/generate', inputs=['prompt'], outputs=['caption', 'score'],
            ...     export_sql=SqlExport(
            ...         db_connect='postgresql+psycopg://user:pw@host/analytics',
            ...         table='captions',
            ...     ),
            ... )
            ... def format_response(*, caption: str, score: float) -> GenerateResponse:
            ...     return GenerateResponse(caption=caption.strip(), score=round(score, 3))

            Each successful POST inserts a row into the Pixeltable table and then appends a row
            with columns `caption`, `score` (the response model fields) to `captions`.
        """
        return self._insert_compute_route(
            t,
            path=path,
            inputs=inputs,
            uploadfile_inputs=uploadfile_inputs,
            outputs=outputs,
            export_sql=export_sql,
            background=background,
            route_type='insert',
        )

    def _insert_compute_route(
        self,
        t: pxt.Table,
        *,
        path: str,
        inputs: list[str] | None,
        uploadfile_inputs: list[str] | None,
        outputs: list[str] | None,
        export_sql: SqlExport | None,
        background: bool,
        route_type: Literal['insert', 'compute'],
    ) -> Callable[[Callable[..., pydantic.BaseModel]], Callable[..., pydantic.BaseModel]]:
        """Shared implementation of insert_route()/compute_route()."""
        error_prefix = f'{route_type}_route()'
        _, input_col_names, output_col_names, cols_by_name = self._validate_dml_args(
            t,
            inputs=inputs,
            uploadfile_inputs=uploadfile_inputs,
            outputs=outputs,
            return_fileresponse=False,
            background=background,
            error_prefix=error_prefix,
            route_type=route_type,
        )
        uploadfile_inputs = uploadfile_inputs or []

        def decorator(user_fn: Callable[..., pydantic.BaseModel]) -> Callable[..., pydantic.BaseModel]:
            output_schema = {col_name: cols_by_name[col_name].col_type for col_name in output_col_names}
            # a compute_route() function can take the entire batch of computed rows instead of per-column
            # values of a single row; the two forms are distinguished by the function signature
            batch_row_model = (
                self._batch_row_model(user_fn, error_prefix=error_prefix) if route_type == 'compute' else None
            )
            batch_param_name: str | None = None
            if batch_row_model is not None:
                batch_param_name, response_model = self._validate_batch_fn(
                    user_fn, batch_row_model, output_schema=output_schema, error_prefix=error_prefix
                )
            else:
                response_model = self._validate_per_column_fn(
                    user_fn, output_schema=output_schema, error_prefix=error_prefix
                )

            sql_exporter = self._make_model_sql_exporter(
                export_sql, response_model=response_model, error_prefix=error_prefix
            )

            def rows_processor(rows: Sequence[Mapping[str, Any]], url_for_media: Callable[[str], str]) -> Any:
                def output_vals(row: Mapping[str, Any]) -> dict[str, Any]:
                    # compute() returns image outputs in-memory; flush them so they convert to /media URLs
                    return {
                        name: self._convert_media_val(_flush_image(row[name]), url_for_media)
                        for name in output_col_names
                    }

                result: Any
                if batch_row_model is not None:
                    assert batch_param_name is not None  # set together with batch_row_model above
                    result = user_fn(**{batch_param_name: [batch_row_model(**output_vals(row)) for row in rows]})
                else:
                    # the per-column form consumes a single row
                    row = _single_computed_row(rows) if route_type == 'compute' else _exactly_one_row(rows)
                    result = user_fn(**output_vals(row))
                if sql_exporter is not None:
                    sql_exporter.export_row(result)
                return result

            self._add_dml_route(
                t,
                path=path,
                pk_col_names=[],
                input_col_names=input_col_names,
                uploadfile_inputs=uploadfile_inputs,
                return_fileresponse=False,
                background=background,
                endpoint_name=f'{route_type}_{path.strip("/").replace("/", "_") or "root"}',
                rows_processor=rows_processor,
                response_model=response_model,
                route_type=route_type,
            )
            return user_fn

        return decorator

    def add_update_route(
        self,
        t: pxt.Table,
        *,
        path: str,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        return_fileresponse: bool = False,
        export_sql: SqlExport | None = None,
        background: bool = False,
    ) -> None:
        """
        Add a POST endpoint that updates a single row in `t` and returns the updated row.
        The row to update is identified by its primary key, which must be included in the
        request body alongside the input column values. The update is performed via a
        single-row [`batch_update()`][pixeltable.Table.batch_update] call, using the primary
        key columns to identify the row and the columns referenced in `inputs` as the values
        to set.

        The request body contains values for the primary key columns plus the input columns
        as JSON fields. The response is a JSON object with the output column values, or a
        [`FileResponse`](https://fastapi.tiangolo.com/advanced/custom-response/#fileresponse)
        when `return_fileresponse=True`.

        Note: media-typed columns (image, video, audio, document) are excluded from `inputs`
        and from the default input set.

        Args:
            t: The table to update.
            path: The URL path for the endpoint.
            inputs: Columns to accept as request fields, excluding primary key and media-typed
                columns (which cannot be updated). Defaults to all non-computed, non-primary-key,
                non-media columns.
            outputs: Columns to include in the response. Defaults to all columns (including
                inputs).
            return_fileresponse: If True, return the single media-typed output column as a
                [`FileResponse`](https://fastapi.tiangolo.com/advanced/custom-response/#fileresponse).
                Requires exactly one media-typed output column.
            export_sql: If set, export each updated row into an external RDBMS table after the
                Pixeltable update succeeds. See [`SqlExport`][pixeltable.serving.SqlExport] for
                the target specification and supported `method` values.

                The row written is the response body: same columns as `outputs`, with media-typed
                columns rendered as URL strings (so the corresponding target columns must be
                string-typed).

                Schema compatibility is validated once at registration time; the target table
                must already exist or registration fails. Mutually exclusive with
                `return_fileresponse`. Compatible with `background=True`.

                Note: with `method='insert'` (the default), every update appends a new row to the
                target table -- the target acts as an audit log, not a current-state view. Use
                `method='update'` to keep the target as a current-state view keyed on the
                target's primary key.

                If the external write fails after the Pixeltable update has already succeeded,
                the request fails with HTTP 500; no rollback is performed.
            background: If True, return immediately with `{"id": ..., "job_url": ...}` and run
                the update in a background thread. Poll `job_url` for the result. Mutually
                exclusive with `return_fileresponse`.

        Examples:
            JSON request/response (table has primary key `id`):

            >>> router.add_update_route(t, path='/update', inputs=['prompt'], outputs=['prompt', 'result'])

            ```bash
            curl -X POST http://localhost:8000/update \\
              -H 'Content-Type: application/json' \\
              -d '{"id": 1, "prompt": "a sunset over the ocean"}'
            # {"prompt": "a sunset over the ocean", "result": "..."}
            ```

            Append every update to an external audit table:

            >>> router.add_update_route(
            ...     t, path='/update', inputs=['prompt'], outputs=['id', 'prompt', 'result'],
            ...     export_sql=SqlExport(
            ...         db_connect='postgresql+psycopg://user:pw@host/analytics',
            ...         table='update_log',
            ...     ),
            ... )

            Background processing:

            >>> router.add_update_route(t, path='/slow-update', background=True)

            ```bash
            # submit
            curl -X POST http://localhost:8000/slow-update \\
              -H 'Content-Type: application/json' \\
              -d '{"id": 1, "prompt": "hello"}'
            # {"id": "abc123", "job_url": "http://localhost:8000/jobs/abc123"}

            # poll
            curl http://localhost:8000/jobs/abc123
            # {"status": "done", "result": {...}}
            ```
        """
        pk_col_names, input_col_names, output_col_names, cols_by_name = self._validate_dml_args(
            t,
            inputs=inputs,
            uploadfile_inputs=None,
            outputs=outputs,
            return_fileresponse=return_fileresponse,
            background=background,
            error_prefix='add_update_route()',
            route_type='update',
        )
        output_cols = [cols_by_name[name] for name in output_col_names]
        _check_route_output_schema(output_cols, 'add_update_route()')
        json_output_col_names = [n for n in output_col_names if cols_by_name[n].col_type.is_json_type()]

        sql_exporter = self._make_schema_sql_exporter(
            export_sql,
            return_fileresponse=return_fileresponse,
            schema={name: cols_by_name[name].col_type for name in output_col_names},
            error_prefix='add_update_route()',
        )

        path_str = ''.join(el.capitalize() for el in path.split('/') if len(el) > 0)
        update_response_model = self._create_model(f'{path_str}Response', output_cols=output_cols)

        def rows_processor(rows: Sequence[Mapping[str, Any]], url_for_media: Callable[[str], str]) -> Any:
            row = _exactly_one_row(rows)
            for name in json_output_col_names:
                _check_json_value_servable(row.get(name), name)
            output = self._create_output(
                [row], output_col_names, update_response_model, return_fileresponse, url_for_media
            )
            result = output[0] if isinstance(output, list) else output
            if sql_exporter is not None:
                assert isinstance(result, pydantic.BaseModel)
                sql_exporter.export_row(result)
            return result

        self._add_dml_route(
            t,
            path=path,
            pk_col_names=pk_col_names,
            input_col_names=input_col_names,
            uploadfile_inputs=[],
            return_fileresponse=return_fileresponse,
            background=background,
            endpoint_name=f'update_{path.strip("/").replace("/", "_") or "root"}',
            rows_processor=rows_processor,
            response_model=update_response_model,
            route_type='update',
        )

    def update_route(
        self,
        t: pxt.Table,
        *,
        path: str,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        export_sql: SqlExport | None = None,
        background: bool = False,
    ) -> Callable[[Callable[..., pydantic.BaseModel]], Callable[..., pydantic.BaseModel]]:
        """
        Decorator that registers a POST endpoint performing a `Table.batch_update()` followed by
        user-defined post-processing.

        The request body carries values for the primary key columns (to identify the row) plus the
        input column values as JSON fields. After updating the row, the decorated function is called
        with the requested output columns as keyword arguments (parameter names and Pixeltable types
        must match `outputs`). Its return value must be a Pydantic model and is returned as the HTTP
        response body.

        Media-typed outputs (image, video, audio, document) are delivered to the function as `/media/` URL
        strings -- annotate those parameters as `str` (or `str | None` if the column is nullable), not as
        `pxt.Image` / `pxt.Video` / etc.

        If the row does not exist, the endpoint returns HTTP 404 without calling the decorated
        function.

        Note: media-typed columns (image, video, audio, document) and primary key columns cannot be
        used as `inputs`. Primary key columns are always part of the request body for row
        identification.

        Args:
            t: The table to update.
            path: The URL path for the endpoint.
            inputs: Columns to accept as update fields. Defaults to all non-computed, non-primary-key,
                non-media columns.
            outputs: Columns from the updated row to pass to the decorated function as keyword
                arguments. Defaults to all columns.
            export_sql: If set, export the decorated function's return value into an external
                RDBMS table after the Pixeltable update succeeds. See
                [`SqlExport`][pixeltable.serving.SqlExport] for the target specification and
                supported `method` values.

                The row written is the user function's pydantic return value (its fields, not the
                source columns), so the target table schema must match those fields. Media-typed
                fields are modeled as strings (URL form).

                Schema compatibility is validated once at registration time; the target table
                must already exist or registration fails. Compatible with `background=True`.

                Note: with `method='insert'` (the default), every update appends a new row to the
                target table -- the target acts as an audit log, not a current-state view. Use
                `method='update'` to keep the target as a current-state view keyed on the
                target's primary key.

                If the external write fails after the Pixeltable update has already succeeded,
                the request fails with HTTP 500; no rollback is performed.
            background: If True, return immediately with `{"id": ..., "job_url": ...}` and run the
                update plus post-processing in a background thread. Poll `job_url` for the result;
                the decorated function's return value is delivered as the job result.

        Examples:
            >>> class ItemResponse(pydantic.BaseModel):
            ...     id: int
            ...     summary: str
            ...     score: float
            ...
            ... @router.update_route(
            ...     t, path='/update', inputs=['text'], outputs=['id', 'text', 'score']
            ... )
            ... def format_response(*, id: int, text: str, score: float) -> ItemResponse:
            ...     return ItemResponse(id=id, summary=text[:100], score=round(score, 3))

            ```bash
            curl -X POST http://localhost:8000/update \\
              -H 'Content-Type: application/json' \\
              -d '{"id": 42, "text": "new content"}'
            # {"id": 42, "summary": "new content", "score": 0.871}
            ```

            Append every post-processed update into an external audit table:

            >>> @router.update_route(
            ...     t, path='/update', inputs=['text'], outputs=['id', 'text', 'score'],
            ...     export_sql=SqlExport(
            ...         db_connect='postgresql+psycopg://user:pw@host/analytics',
            ...         table='item_log',
            ...     ),
            ... )
            ... def format_response(*, id: int, text: str, score: float) -> ItemResponse:
            ...     return ItemResponse(id=id, summary=text[:100], score=round(score, 3))

            Background processing:

            >>> @router.update_route(t, path='/slow-update', background=True)
            ... def format_response(*, id: int, result: str) -> MyResponse:
            ...     return MyResponse(id=id, result=result.strip())

            ```bash
            # submit
            curl -X POST http://localhost:8000/slow-update \\
              -H 'Content-Type: application/json' \\
              -d '{"id": 1, "text": "hello"}'
            # {"id": "abc123", "job_url": "http://localhost:8000/jobs/abc123"}

            # poll
            curl http://localhost:8000/jobs/abc123
            # {"status": "done", "result": {"id": 1, "result": "hello"}}
            ```
        """
        pk_col_names, input_col_names, output_col_names, cols_by_name = self._validate_dml_args(
            t,
            inputs=inputs,
            uploadfile_inputs=None,
            outputs=outputs,
            return_fileresponse=False,
            background=background,
            error_prefix='update_route()',
            route_type='update',
        )

        def decorator(user_fn: Callable[..., pydantic.BaseModel]) -> Callable[..., pydantic.BaseModel]:
            response_model = self._validate_per_column_fn(
                user_fn,
                output_schema={col_name: cols_by_name[col_name].col_type for col_name in output_col_names},
                error_prefix='update_route()',
            )

            sql_exporter = self._make_model_sql_exporter(
                export_sql, response_model=response_model, error_prefix='update_route()'
            )

            def rows_processor(rows: Sequence[Mapping[str, Any]], url_for_media: Callable[[str], str]) -> Any:
                row = _exactly_one_row(rows)
                kwargs = {name: self._convert_media_val(row[name], url_for_media) for name in output_col_names}
                result = user_fn(**kwargs)
                if sql_exporter is not None:
                    sql_exporter.export_row(result)
                return result

            self._add_dml_route(
                t,
                path=path,
                pk_col_names=pk_col_names,
                input_col_names=input_col_names,
                uploadfile_inputs=[],
                return_fileresponse=False,
                background=background,
                endpoint_name=f'update_{path.strip("/").replace("/", "_") or "root"}',
                rows_processor=rows_processor,
                response_model=response_model,
                route_type='update',
            )
            return user_fn

        return decorator

    def add_delete_route(
        self, t: pxt.Table, *, path: str, match_columns: list[str] | None = None, background: bool = False
    ) -> None:
        """
        Add a POST endpoint that deletes rows from `t` matching the given match column values.

        The request body contains the match column values as JSON fields. The endpoint deletes every row
        where each match column equals the provided value, and returns the number of rows affected.

        Args:
            t: The table to delete from.
            path: The URL path for the endpoint.
            match_columns: Columns to match on (AND-ed equality). Defaults to the table's primary key.
                Must be non-empty.
            background: If True, return immediately with `{"id": ..., "job_url": ...}` and run the
                operation in a background thread. Poll `job_url` for the result.

        Examples:
            >>> router.add_delete_route(t, path='/delete')

            ```bash
            curl -X POST http://localhost:8000/delete \\
              -H 'Content-Type: application/json' \\
              -d '{"id": 42}'
            # {"num_rows": 1}
            ```
        """
        md = t.get_metadata()
        if md['kind'] != 'table':
            raise pxt.RequestError(
                pxt.ErrorCode.UNSUPPORTED_OPERATION,
                f'add_delete_route(): cannot delete from {md["kind"]} {md["name"]!r}',
            )

        schema_version = md['schema_version']
        col_md = md['columns']

        match_columns = copy.copy(match_columns)  # insulate ourselves from external changes
        if match_columns is None:
            pk = [name for name, c in col_md.items() if c['is_primary_key']]
            if not pk:
                raise pxt.RequestError(
                    pxt.ErrorCode.UNSUPPORTED_OPERATION,
                    'add_delete_route(): table has no primary key; specify `match_columns` explicitly',
                )
            match_columns = pk
        if len(match_columns) == 0:
            raise pxt.RequestError(
                pxt.ErrorCode.MISSING_REQUIRED, 'add_delete_route(): `match_columns` must be non-empty'
            )
        for name in match_columns:
            if name not in col_md:
                raise pxt.NotFoundError(pxt.ErrorCode.COLUMN_NOT_FOUND, f'add_delete_route(): unknown column {name!r}')

        endpoint_model: type[pydantic.BaseModel] = BackgroundJobResponse if background else DeleteResponse

        def run_delete(
            row_kwargs: dict[str, Any],
            url_for_media: Callable[[str], str],  # unused; part of the endpoint_op contract
        ) -> DeleteResponse:
            # the table handle is thread-safe, so reuse the captured one; re-validate the frozen contract
            _validate_registered_schema(t, schema_version)

            where_expr: exprs.Expr | None = None
            for name in match_columns:
                predicate = t[name] == row_kwargs[name]
                where_expr = predicate if where_expr is None else (where_expr & predicate)
            status = t.delete(where=where_expr)
            return DeleteResponse(num_rows=status.num_rows)

        # use the metadata path (works for both local and hosted tables); skip system columns (name is None)
        cols_by_name = {col.name: col for col in t._tbl_path.column_md() if col.name is not None}
        match_cols = [cols_by_name[name] for name in match_columns]
        sig = self._create_endpoint_signature(input_cols=match_cols)
        endpoint = PxtEndpoint(
            self,
            f'delete_{path.strip("/").replace("/", "_") or "root"}',
            sig,
            uploadfile_inputs=[],
            background=background,
            endpoint_op=run_delete,
            tbl=t,
            route_type='delete',
        )
        self.add_api_route(path, endpoint, methods=['POST'], response_model=endpoint_model)

    def add_query_route(
        self,
        *,
        path: str,
        query: pxt.Function,
        inputs: list[str] | None = None,
        uploadfile_inputs: list[str] | None = None,
        one_row: bool = False,
        return_fileresponse: bool = False,
        background: bool = False,
        method: Literal['get', 'post'] = 'post',
    ) -> None:
        """
        Add an endpoint that executes a `@pxt.query` or `pxt.retrieval_udf` and returns the results.

        By default the endpoint accepts POST requests with a JSON
        [`Body`](https://fastapi.tiangolo.com/tutorial/body/) and returns `{"rows": [{...}, ...]}`.
        Use `method='get'` for
        [`Query`](https://fastapi.tiangolo.com/tutorial/query-params/) parameters instead.

        Args:
            path: The URL path for the endpoint.
            query: The query to execute, created with `@pxt.query` or `pxt.retrieval_udf()`.
            inputs: Parameters to accept as request fields. Defaults to all query parameters.
            uploadfile_inputs: Parameters to accept as
                [`UploadFile`](https://fastapi.tiangolo.com/tutorial/request-files/) fields
                (must be media-typed).
            one_row: If True, expect exactly one result row and return it as a plain JSON object
                (not wrapped in `{"rows": [...]}`). 0 rows produces a 404, >1 rows a 409.
            return_fileresponse: If True, return the single media-typed result column as a
                [`FileResponse`](https://fastapi.tiangolo.com/advanced/custom-response/#fileresponse).
                Requires `one_row` semantics (0 rows -> 404, >1 rows -> 409).
                Mutually exclusive with `background`.
            background: If True, return immediately with `{"id": ..., "job_url": ...}` and run
                the query in a background thread. Poll `job_url` for the result. Mutually
                exclusive with `return_fileresponse`.
            method: HTTP method for the endpoint (`'get'` or `'post'`).

        Examples:
            Multi-row JSON response:

            >>> router.add_query_route(path='/search', query=search_docs)

            ```bash
            curl -X POST http://localhost:8000/search \\
              -H 'Content-Type: application/json' \\
              -d '{"query_text": "hello"}'
            # {"rows": [{"id": 1, "text": "hello world", "score": 0.95}, ...]}
            ```

            Single-row lookup:

            >>> router.add_query_route(path='/lookup', query=lookup_by_id, one_row=True)

            ```bash
            curl -X POST http://localhost:8000/lookup -d '{"id": 42}'
            # {"id": 42, "name": "Alice", "email": "alice@example.com"}
            ```

            GET with query-string parameters:

            >>> router.add_query_route(path='/lookup', query=lookup_by_id, method='get')

            ```bash
            curl 'http://localhost:8000/lookup?id=42'
            # {"id": 42, "name": "Alice", "email": "alice@example.com"}
            ```

            `FileResponse`:

            >>> router.add_query_route(
            ...     path='/thumbnail', query=get_thumbnail, return_fileresponse=True,
            ... )

            ```bash
            curl -X POST http://localhost:8000/thumbnail -d '{"id": 1}' --output thumb.jpg
            # saves the thumbnail image to thumb.jpg
            ```
        """
        if not isinstance(query, func.QueryTemplateFunction):
            raise pxt.RequestError(
                pxt.ErrorCode.TYPE_MISMATCH,
                f'add_query_route(): `query` must be a @pxt.query or retrieval_udf, not {type(query).__name__}',
            )
        if return_fileresponse and background:
            raise pxt.RequestError(
                pxt.ErrorCode.INVALID_ARGUMENT,
                'add_query_route(): return_fileresponse and background are mutually exclusive',
            )
        uploadfile_inputs = uploadfile_inputs or []

        if method == 'get' and len(uploadfile_inputs) > 0:
            raise pxt.RequestError(
                pxt.ErrorCode.INVALID_ARGUMENT,
                f'add_query_route(): GET endpoints cannot have uploadfile_inputs (got {uploadfile_inputs})',
            )

        # Freeze the response schema at registration time by resolving SELECT * against the
        # current columns. Subsequent requests use this materialized query, so adding or
        # dropping columns on the underlying table doesn't silently change the API contract.
        template_query = query.template_query
        effective_select_list = list(template_query._effective_select_list)
        template_query = pxt.Query(
            select_list=[(e, n) for e, n in effective_select_list],
            from_clause=template_query._from_clause,
            where_clause=template_query.where_clause,
            group_by_clause=template_query.group_by_clause,
            grouping_tbl_key=template_query.grouping_tbl_key,
            order_by_clause=template_query.order_by_clause,
            limit=template_query.limit_val,
            offset=template_query.offset_val,
            sample_clause=template_query.sample_clause,
        )

        query_params = dict(query.signature.parameters)
        query_schema = {p.name: p.col_type for p in query_params.values()}
        result_schema = dict(template_query.schema)

        input_param_names, _ = self._validate_args(
            input_schema=query_schema,
            output_schema=result_schema,
            inputs=inputs,
            uploadfile_inputs=uploadfile_inputs,
            outputs=None,
            return_fileresponse=return_fileresponse,
            error_prefix='add_query_route()',
            input_item_str='parameter',
            output_item_str='column',
        )
        # restrict the schema to validated inputs so the endpoint only accepts the declared parameters
        input_schema = {name: query_schema[name] for name in input_param_names}
        # extract Python-level defaults so they appear in the endpoint signature and OpenAPI spec
        input_defaults = {
            name: query_params[name].default.val for name in input_param_names if query_params[name].has_default()
        }

        return_scalar = query.return_scalar
        scalar_col_name = next(iter(result_schema)) if return_scalar else None

        # build pydantic models for the response
        output_model: type[pydantic.BaseModel] | None  # what's returned by run_query(); None for scalars
        query_result_model: type[pydantic.BaseModel] | None = None
        if return_scalar:
            output_model = None
        else:
            path_elements = path.split('/')
            path_str = ''.join(el.capitalize() for el in path_elements if len(el) > 0)
            query_result_model = self._create_model(name=f'{path_str}RowResponse', output_schema=result_schema)
            if one_row:
                output_model = query_result_model
            else:
                # Multi-row: wrap per-row models in a response with a rows field.
                output_model = pydantic.create_model(
                    f'{path_str}Response',
                    rows=(list[query_result_model], pydantic.Field(description='Query result rows')),  # type: ignore[valid-type]
                )

        # endpoint_model: what FastAPI advertises as the response type
        endpoint_model: Any
        if background:
            endpoint_model = BackgroundJobResponse
        elif return_fileresponse:
            endpoint_model = None
        elif return_scalar:
            scalar_py_type = next(iter(result_schema.values())).to_pydantic_type()
            endpoint_model = list[scalar_py_type] if not one_row else scalar_py_type  # type: ignore[valid-type]
        else:
            endpoint_model = output_model

        # Apply per-expr rewrites for media/image columns. Done after validation so that
        # _validate_args sees the original media-typed schema; the run_query closure below
        # captures the rewritten query.
        has_img_col_refs = any(
            e.col_type.is_image_type() and isinstance(e, exprs.ColumnRef) for e, _ in effective_select_list
        )
        if has_img_col_refs or return_fileresponse:
            rewritten_select_list: list[tuple[exprs.Expr, str]] = []
            for e, name in effective_select_list:
                if return_fileresponse and isinstance(e, exprs.ColumnRef) and e.col_type.is_media_type():
                    # serve from a local path even if the media file is stored externally
                    rewritten_select_list.append(
                        (exprs.ColumnPropertyRef(e, exprs.ColumnPropertyRef.Property.LOCALPATH), name)
                    )
                elif isinstance(e, exprs.ColumnRef) and e.col_type.is_image_type():
                    # avoid materializing PIL.Image in the response payload
                    rewritten_select_list.append(
                        (exprs.ColumnPropertyRef(e, exprs.ColumnPropertyRef.Property.FILEURL), name)
                    )
                else:
                    rewritten_select_list.append((e, name))
            template_query = pxt.Query(
                select_list=rewritten_select_list,
                from_clause=template_query._from_clause,
                where_clause=template_query.where_clause,
                group_by_clause=template_query.group_by_clause,
                grouping_tbl_key=template_query.grouping_tbl_key,
                order_by_clause=template_query.order_by_clause,
                limit=template_query.limit_val,
                offset=template_query.offset_val,
                sample_clause=template_query.sample_clause,
            )

        def run_query(call_kwargs: dict[str, Any], url_for_media: Callable[[str], str]) -> Any:
            result_set = template_query._collect(args=call_kwargs)
            rows = list(result_set)

            # do error checking now, before converting data
            if return_fileresponse or one_row:
                if len(rows) == 0:
                    raise HTTPException(status_code=404, detail='query returned no rows')
                if len(rows) > 1:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f'Query returned {len(rows)} rows; expected exactly 1 for one_row=True. '
                            'Change the query to return exactly one row.'
                        ),
                    )

            output = self._create_output(
                rows,
                output_names=list(result_schema.keys()),
                output_model=None if return_scalar else query_result_model,
                return_fileresponse=return_fileresponse,
                url_for_media=url_for_media,
            )
            if return_fileresponse:
                return output
            elif return_scalar:
                # output is list[dict]; extract the single column value from each row
                assert isinstance(output, list)
                scalars = [row[scalar_col_name] for row in output]  # type: ignore[index]
                return scalars[0] if one_row else scalars
            elif one_row:
                assert isinstance(output, list)
                return output[0]
            else:
                return output_model(rows=output)

        sig = self._create_endpoint_signature(
            input_schema=input_schema,
            upload_col_names=uploadfile_inputs,
            is_post=(method == 'post'),
            defaults=input_defaults,
        )
        endpoint = PxtEndpoint(
            self,
            f'query_{path.strip("/").replace("/", "_") or "root"}',
            sig,
            uploadfile_inputs=uploadfile_inputs,
            background=background,
            endpoint_op=run_query,
            tbl=None,
            route_type='query',
        )

        api_kwargs: dict[str, Any] = {'methods': [method.upper()]}
        if endpoint_model is not None:
            api_kwargs['response_model'] = endpoint_model
        if return_fileresponse:
            api_kwargs['response_class'] = FileResponse
        self.add_api_route(path, endpoint, **api_kwargs)

    def _make_schema_sql_exporter(
        self,
        export_sql: SqlExport | None,
        *,
        return_fileresponse: bool,
        schema: dict[str, ts.ColumnType],
        error_prefix: str,
    ) -> SqlExporter | None:
        if export_sql is None:
            return None
        if return_fileresponse:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                f'{error_prefix}: export_sql and return_fileresponse are mutually exclusive',
            )
        return SqlExporter(
            export_sql,
            engine=self._get_sql_engine(export_sql.db_connect),
            output_schema=schema,
            error_prefix=error_prefix,
        )

    def _make_model_sql_exporter(
        self, export_sql: SqlExport | None, *, response_model: type[pydantic.BaseModel], error_prefix: str
    ) -> SqlExporter | None:
        if export_sql is None:
            return None
        sql_output_schema: dict[str, ts.ColumnType] = {}
        for fname, finfo in response_model.model_fields.items():
            ct = ts.ColumnType.from_python_type(finfo.annotation, infer_pydantic_json=True)
            if ct is None:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_TYPE,
                    f'{error_prefix}: cannot interpret response field {fname!r} annotation '
                    f'{finfo.annotation!r} as a Pixeltable type for export_sql',
                )
            sql_output_schema[fname] = ct
        return SqlExporter(
            export_sql,
            engine=self._get_sql_engine(export_sql.db_connect),
            output_schema=sql_output_schema,
            error_prefix=error_prefix,
        )

    def _add_dml_route(
        self,
        t: pxt.Table,
        *,
        path: str,
        pk_col_names: list[str],
        input_col_names: list[str],
        uploadfile_inputs: list[str],
        return_fileresponse: bool,
        background: bool,
        endpoint_name: str,
        rows_processor: Callable[[Sequence[Mapping[str, Any]], Callable[[str], str]], Any],
        response_model: Any | None,
        route_type: Literal['insert', 'update', 'compute'],
    ) -> None:
        """Create endpoint for insert/compute/update.

        The endpoint signature is the PK columns (for row identification, update only) followed by
        the input columns. rows_processor is called with the operation's output rows and url_for_media
        and produces the response body. For insert routes, pk_col_names must be []; for update routes,
        uploadfile_inputs must be [].
        """
        md = t.get_metadata()
        schema_version = md['schema_version']

        # use the metadata path (works for both local and hosted tables); skip system columns (name is None)
        cols_by_name = {col.name: col for col in t._tbl_path.column_md() if col.name is not None}
        pk_cols = [cols_by_name[name] for name in pk_col_names]
        # compute-route inputs conform to the insertable base's schema (see _validate_dml_args)
        input_cols_by_name = (
            {col.name: col for col in t._tbl_path.root.column_md() if col.name is not None}
            if route_type == 'compute'
            else cols_by_name
        )
        input_cols = [input_cols_by_name[name] for name in input_col_names]

        def run_dml(row_kwargs: dict[str, Any], url_for_media: Callable[[str], str]) -> Any:
            # the table handle is thread-safe, so reuse the captured one; re-validate the frozen contract
            _validate_registered_schema(t, schema_version)
            rows: Sequence[Mapping[str, Any]]
            if route_type == 'update':
                status = t.batch_update([row_kwargs], if_not_exists='ignore', return_rows=True)
                if status.num_rows == 0:
                    raise HTTPException(status_code=404, detail='row not found')
                rows = status.rows or []
            elif route_type == 'compute':
                rows = t.compute([row_kwargs])
            else:  # 'insert'
                status = t.insert([row_kwargs], return_rows=True)
                rows = status.rows or []
            return rows_processor(rows, url_for_media)

        sig = self._create_endpoint_signature(input_cols=pk_cols + input_cols, upload_col_names=uploadfile_inputs)
        endpoint = PxtEndpoint(
            self,
            endpoint_name,
            sig,
            uploadfile_inputs=uploadfile_inputs,
            background=background,
            endpoint_op=run_dml,
            tbl=t,
            route_type=route_type,
        )
        api_kwargs: dict[str, Any] = {'methods': ['POST']}
        if background:
            api_kwargs['response_model'] = BackgroundJobResponse
        elif not return_fileresponse:
            assert response_model is not None
            api_kwargs['response_model'] = response_model
        if return_fileresponse:
            api_kwargs['response_class'] = FileResponse

        self.add_api_route(path, endpoint, **api_kwargs)

    def _validate_per_column_fn(
        self, user_fn: Callable, *, output_schema: dict[str, ts.ColumnType], error_prefix: str
    ) -> type[pydantic.BaseModel]:
        """Validate the per-column form of a decorated route function; return the resolved response model."""
        sig = inspect.signature(user_fn)
        fn_name = getattr(user_fn, '__name__', repr(user_fn))
        hints = _resolve_type_hints(user_fn, fn_name, error_prefix)

        param_names: set[str] = set()
        output_col_names = list(output_schema.keys())
        for p in sig.parameters.values():
            if p.kind != inspect.Parameter.KEYWORD_ONLY:
                raise pxt.RequestError(
                    pxt.ErrorCode.UNSUPPORTED_OPERATION,
                    f'{error_prefix}: {fn_name!r} parameter {p.name!r} must be keyword-only '
                    '(place parameters after `*` in the signature)',
                )

            if p.name not in output_col_names:
                raise pxt.RequestError(
                    pxt.ErrorCode.UNSUPPORTED_OPERATION,
                    f'{error_prefix}: {fn_name!r} parameter {p.name!r} is not in the declared outputs '
                    f'{output_col_names}',
                )

            if p.name not in hints:
                raise pxt.RequestError(
                    pxt.ErrorCode.INVALID_ARGUMENT,
                    f'{error_prefix}: {fn_name!r} parameter {p.name!r} has no type annotation',
                )

            _validate_type_hint(
                hints[p.name],
                output_schema[p.name],
                subject=f'{fn_name!r} parameter {p.name!r}',
                error_prefix=error_prefix,
            )
            param_names.add(p.name)

        missing = [n for n in output_col_names if n not in param_names]
        if missing:
            raise pxt.RequestError(
                pxt.ErrorCode.MISSING_REQUIRED,
                f'{error_prefix}: {fn_name!r} is missing parameters for outputs {missing}; every declared '
                'output must appear as a keyword-only parameter',
            )

        return _validated_response_model(hints, fn_name, error_prefix)

    def _batch_row_model(self, user_fn: Callable, *, error_prefix: str) -> type[pydantic.BaseModel] | None:
        """The row model M if user_fn has the batch form (a single parameter annotated list[M], with M a pydantic
        model), otherwise None. Detection only: it does not check M's fields against the declared outputs.
        """
        sig = inspect.signature(user_fn)
        params = list(sig.parameters.values())
        if len(params) != 1 or params[0].kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return None
        fn_name = getattr(user_fn, '__name__', repr(user_fn))
        annot = _resolve_type_hints(user_fn, fn_name, error_prefix).get(params[0].name)
        if get_origin(annot) is not list:
            return None
        annot_args = get_args(annot)
        if len(annot_args) != 1 or not (
            isinstance(annot_args[0], type) and issubclass(annot_args[0], pydantic.BaseModel)
        ):
            return None
        return annot_args[0]

    def _validate_batch_fn(
        self,
        user_fn: Callable,
        row_model: type[pydantic.BaseModel],
        *,
        output_schema: dict[str, ts.ColumnType],
        error_prefix: str,
    ) -> tuple[str, type[pydantic.BaseModel]]:
        """Validate the batch form of a compute_route() function and return (parameter name, response model).

        The sole parameter's name must not collide with an output column, and row_model's fields must cover the
        declared outputs with compatible types.
        """
        sig = inspect.signature(user_fn)
        fn_name = getattr(user_fn, '__name__', repr(user_fn))
        hints = _resolve_type_hints(user_fn, fn_name, error_prefix)
        param_name = next(iter(sig.parameters))

        if param_name in output_schema:
            raise pxt.RequestError(
                pxt.ErrorCode.INVALID_ARGUMENT,
                f'{error_prefix}: {fn_name!r} parameter {param_name!r} matches an output column name, which is '
                'ambiguous; rename the parameter or use one keyword-only parameter per output column',
            )

        # validate the row model's fields against the output columns, like per-column parameters
        for field_name, field in row_model.model_fields.items():
            if field_name not in output_schema:
                raise pxt.RequestError(
                    pxt.ErrorCode.UNSUPPORTED_OPERATION,
                    f'{error_prefix}: {fn_name!r} row model field {field_name!r} is not in the declared outputs '
                    f'{list(output_schema.keys())}',
                )
            if field.annotation is None:
                raise pxt.RequestError(
                    pxt.ErrorCode.INVALID_ARGUMENT,
                    f'{error_prefix}: {fn_name!r} row model field {field_name!r} has no type annotation',
                )
            _validate_type_hint(
                field.annotation,
                output_schema[field_name],
                subject=f'{fn_name!r} row model field {field_name!r}',
                error_prefix=error_prefix,
            )
        missing = [n for n in output_schema if n not in row_model.model_fields]
        if missing:
            raise pxt.RequestError(
                pxt.ErrorCode.MISSING_REQUIRED,
                f'{error_prefix}: {fn_name!r} row model is missing fields for outputs {missing}; every declared '
                'output must appear as a row model field',
            )

        return param_name, _validated_response_model(hints, fn_name, error_prefix)

    class DmlArgsValidationResult(NamedTuple):
        pk_col_names: list[str]
        input_col_names: list[str]
        output_col_names: list[str]
        cols_by_name: dict[str, catalog.ColumnVersionMd]

    def _validate_dml_args(
        self,
        t: pxt.Table,
        *,
        inputs: list[str] | None,
        uploadfile_inputs: list[str] | None,
        outputs: list[str] | None,
        return_fileresponse: bool,
        background: bool,
        error_prefix: str,
        route_type: Literal['insert', 'update', 'compute'],
    ) -> DmlArgsValidationResult:
        """
        Validate insert-/update-route args. Returns (pk_col_names, input_col_names, output_col_names, cols_by_name).
        """
        verb = 'insert into' if route_type == 'insert' else route_type
        md = t.get_metadata()
        allowed_kinds = ('table', 'view') if route_type == 'compute' else ('table',)
        if md['kind'] not in allowed_kinds:
            raise pxt.RequestError(
                pxt.ErrorCode.UNSUPPORTED_OPERATION, f'{error_prefix}: cannot {verb} {md["kind"]} {md["name"]!r}'
            )
        if route_type == 'compute':
            # mirrors the restrictions of Table.compute()
            if t._tbl_path.has_snapshot():
                raise pxt.RequestError(
                    pxt.ErrorCode.UNSUPPORTED_OPERATION,
                    f'{error_prefix}: cannot compute view {md["name"]!r}: its base hierarchy contains a snapshot',
                )
            if t._tbl_path.has_sample_clause():
                raise pxt.RequestError(
                    pxt.ErrorCode.UNSUPPORTED_OPERATION,
                    f'{error_prefix}: cannot compute view {md["name"]!r}: it is defined with a sample clause',
                )
        if return_fileresponse and background:
            raise pxt.RequestError(
                pxt.ErrorCode.INVALID_ARGUMENT,
                f'{error_prefix}: return_fileresponse and background are mutually exclusive',
            )

        col_md = md['columns']
        pk_col_names = [name for name, c in col_md.items() if c['is_primary_key']]
        if route_type == 'update' and not pk_col_names:
            raise pxt.RequestError(pxt.ErrorCode.UNSUPPORTED_OPERATION, f'{error_prefix}: table has no primary key')

        # skip system columns (name is None)
        cols_by_name = {col.name: col for col in t._tbl_path.column_md() if col.name is not None}
        # compute() takes rows conforming to the insertable base table's schema, instead of the view schema
        input_cols_by_name = (
            {col.name: col for col in t._tbl_path.root.column_md() if col.name is not None}
            if route_type == 'compute'
            else cols_by_name
        )
        pk_set = set(pk_col_names)

        for name in [*(inputs or []), *(uploadfile_inputs or [])]:
            # computed columns cannot be inputs
            if name in input_cols_by_name and input_cols_by_name[name].is_computed:
                raise pxt.RequestError(
                    pxt.ErrorCode.INVALID_ARGUMENT,
                    f'{error_prefix}: {name!r} is a computed column and cannot be used as input',
                )

            # PK columns cannot be inputs for an update route
            if route_type == 'update' and name in pk_set:
                raise pxt.RequestError(
                    pxt.ErrorCode.INVALID_ARGUMENT,
                    f'{error_prefix}: {name!r} is a primary key column and cannot be used as input',
                )

            # media columns cannot be updated
            if route_type == 'update' and name in cols_by_name and cols_by_name[name].col_type.is_media_type():
                raise pxt.RequestError(
                    pxt.ErrorCode.UNSUPPORTED_OPERATION,
                    f'{error_prefix}: {name!r} is a media column and cannot be updated',
                )

        # input_schema: non-computed columns; for updates, also exclude PK and media columns
        input_schema = {
            c.name: c.col_type
            for c in input_cols_by_name.values()
            if not c.is_computed and not (route_type == 'update' and (c.name in pk_set or c.col_type.is_media_type()))
        }
        input_col_names, output_col_names = self._validate_args(
            input_schema=input_schema,
            output_schema={c.name: c.col_type for c in cols_by_name.values()},
            inputs=inputs,
            uploadfile_inputs=uploadfile_inputs,
            outputs=outputs,
            return_fileresponse=return_fileresponse,
            error_prefix=error_prefix,
            input_item_str='column',
            output_item_str='column',
        )
        return self.DmlArgsValidationResult(pk_col_names, input_col_names, output_col_names, cols_by_name)

    def _validate_args(
        self,
        input_schema: dict[str, ts.ColumnType],
        output_schema: dict[str, ts.ColumnType],
        inputs: list[str] | None,
        uploadfile_inputs: list[str] | None,
        outputs: list[str] | None,
        return_fileresponse: bool,
        error_prefix: str,
        input_item_str: str,
        output_item_str: str,
    ) -> tuple[list[str], list[str]]:
        """
        Validates inputs/uploadfile_inputs/outputs against input_/output_schema.
        input_schema and output_schema are allowed to overlap.
        Returns (complete list of input column names, complete list of output column names).
        """
        # validate inputs
        input_cols: list[str]
        if inputs is None:
            # default: all input columns
            input_cols = list(input_schema.keys())
        else:
            input_cols = []
            for name in inputs:
                if name not in input_schema:
                    raise pxt.NotFoundError(
                        pxt.ErrorCode.COLUMN_NOT_FOUND, f'{error_prefix}: unknown input {input_item_str} {name!r}'
                    )
                input_cols.append(name)

        # validate uploadfile_inputs
        if uploadfile_inputs is not None:
            for name in uploadfile_inputs:
                if name not in input_schema:
                    raise pxt.NotFoundError(
                        pxt.ErrorCode.COLUMN_NOT_FOUND,
                        f'{error_prefix}: unknown uploadfile input {input_item_str} {name!r}',
                    )
                col_type = input_schema[name]
                if not col_type.is_media_type():
                    raise pxt.RequestError(
                        pxt.ErrorCode.UNSUPPORTED_OPERATION,
                        f'{error_prefix}: uploadfile input {input_item_str} {name!r} is not a media {input_item_str}',
                    )
                if inputs is not None and name in inputs:
                    raise pxt.RequestError(
                        pxt.ErrorCode.INVALID_ARGUMENT,
                        f'{error_prefix}: {name!r} appears in both `inputs` and `uploadfile_inputs`',
                    )
                if name not in input_cols:
                    input_cols.append(name)

        # validate outputs
        output_cols: list[str]
        if outputs is None:
            output_cols = list(output_schema.keys())
        else:
            output_cols = []
            for name in outputs:
                if name not in output_schema:
                    raise pxt.NotFoundError(
                        pxt.ErrorCode.COLUMN_NOT_FOUND, f'{error_prefix}: unknown output {output_item_str} {name!r}'
                    )
                output_cols.append(name)

        if return_fileresponse and (len(output_cols) != 1 or not output_schema[output_cols[0]].is_media_type()):
            raise pxt.RequestError(
                pxt.ErrorCode.INVALID_ARGUMENT,
                f'{error_prefix}: return_fileresponse=True requires exactly one media-typed output {output_item_str}',
            )

        return input_cols, output_cols

    def _create_model(
        self,
        name: str,
        output_schema: dict[str, ts.ColumnType] | None = None,
        output_cols: list[catalog.ColumnVersionMd] | None = None,
    ) -> type[pydantic.BaseModel]:
        assert (output_schema is None) != (output_cols is None)
        fields: dict[str, tuple[Any, FieldInfo]]
        if output_cols is not None:
            fields = {col.name: self._build_response_field(col.col_type, comment=col.comment) for col in output_cols}
        else:
            fields = {col_name: self._build_response_field(col_type) for col_name, col_type in output_schema.items()}
        return pydantic.create_model(name, **fields)  # type: ignore[call-overload]

    def _create_endpoint_signature(
        self,
        input_cols: list[catalog.ColumnVersionMd] | None = None,
        input_schema: dict[str, ts.ColumnType] | None = None,
        defaults: dict[str, Any] | None = None,  # input name -> default value
        upload_col_names: list[str] | None = None,
        is_post: bool = True,
    ) -> inspect.Signature:
        assert (input_cols is None) != (input_schema is None)

        # the first parameter is request, which FastAPI auto-injects (it doesn't appear in the request body / OpenAPI
        # inputs); we need it to resolve media URLs via request.url_for().
        params: list[inspect.Parameter] = [
            inspect.Parameter('request', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Request)
        ]

        # if any upload is present, we have to use multipart/form-data (every non-UploadFile parameter needs Form(...))
        upload_col_names = upload_col_names or []
        has_uploads = len(upload_col_names) > 0
        assert is_post or not has_uploads
        defaults = defaults or {}

        input_cols_by_name = {c.name: c for c in input_cols} if input_cols is not None else {}
        schema: dict[str, ts.ColumnType]
        if input_schema is not None:
            schema = input_schema
        else:
            schema = {name: col.col_type for name, col in input_cols_by_name.items()}

        for name, col_type in schema.items():
            annotation: Any
            # propagate the column comment (if any) as the OpenAPI field description
            desc = input_cols_by_name[name].comment if name in input_cols_by_name else None
            if name in upload_col_names:
                annotation = Annotated[UploadFile, File(..., description=desc)]
            elif not is_post:
                py_type = col_type.to_pydantic_type()
                annotation = Annotated[py_type, QueryParam(description=desc)]
            else:
                py_type = col_type.to_pydantic_type()
                if has_uploads:
                    annotation = Annotated[py_type, Form(description=desc)]
                else:
                    annotation = Annotated[py_type, Body(embed=True, description=desc)]

            if name in defaults:
                param_default = defaults[name]
            elif col_type.nullable:
                param_default = None
            else:
                param_default = inspect.Parameter.empty
            params.append(
                inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, annotation=annotation, default=param_default)
            )

        return inspect.Signature(parameters=params)

    def _create_output(
        self,
        rows: Sequence[Mapping[str, Any]],
        output_names: list[str],
        output_model: type[pydantic.BaseModel] | None,
        return_fileresponse: bool,
        url_for_media: Callable[[str], str],
    ) -> FileResponse | list[pydantic.BaseModel] | list[dict[str, Any]]:
        """
        Converts rows by turning PIL images and local media uris into fetchable urls.

        Returns:
        - a FileResponse, if return_fileresponse is True
        - a list of pydantic model instances, if output_model is not None
        - otherwise a list of row dicts
        """

        rows = [{col_name: _flush_image(val) for col_name, val in row.items()} for row in rows]

        if return_fileresponse:
            assert len(output_names) == 1 and len(rows) == 1
            output_name = output_names[0]
            val = rows[0][output_name]
            if val is None:
                raise HTTPException(status_code=500, detail=f'output column {output_name!r} is null')
            local_path: Path
            if val.startswith('file:'):
                local_path = LocalStore.file_url_to_path(val) or Path(val)
            elif os.path.isabs(val):
                local_path = Path(val)
            else:
                # a remote reference (the proxy daemon's media, or an external s3/http url): fetch it to a local
                # file so it can be returned as a FileResponse
                local_path = fetch_url(val)
            if not local_path.exists() or not local_path.is_file():
                raise HTTPException(status_code=500, detail=f'output file not found: {output_name!r}')
            media_type, _ = mimetypes.guess_type(local_path)
            return FileResponse(local_path, media_type=media_type or 'application/octet-stream')

        converted = [{name: self._convert_media_val(row[name], url_for_media) for name in output_names} for row in rows]
        if output_model is not None:
            return [output_model(**row) for row in converted]
        else:
            return converted

    def _is_allowed_media_path(self, resolved: Path) -> bool:
        return any(resolved == d or d in resolved.parents for d in self._allowed_media_dirs)

    def _register_media_route(self) -> None:
        """Register a GET /media/{path:path} route that serves Pixeltable media and tmp files"""

        def serve_media(path: str) -> FileResponse:
            resolved = (self._home_dir / path).resolve()
            if not self._is_allowed_media_path(resolved):
                raise HTTPException(status_code=404, detail='not found')
            if not resolved.is_file():
                raise HTTPException(status_code=404, detail='not found')
            media_type, _ = mimetypes.guess_type(str(resolved))
            return FileResponse(resolved, media_type=media_type or 'application/octet-stream')

        # name=...: we need to be able to refer to this route in Request.url_for()
        self.add_api_route(
            '/media/{path:path}', serve_media, methods=['GET'], response_class=FileResponse, name=_MEDIA_ROUTE_NAME
        )

    def _register_jobs_route(self) -> None:
        def get_job_status(job_id: str) -> JobStatusResponse:
            with self._jobs_lock:
                fut = self._jobs.get(job_id)
            if fut is None:
                raise HTTPException(status_code=404, detail='unknown job id')
            if not fut.done():
                return JobStatusResponse(status='pending')
            exc = fut.exception()
            if exc is not None:
                return JobStatusResponse(status='error', error=str(exc))
            # FileResponse cannot be JSON-serialized; only response_model instances make it here
            # in practice (background+return_fileresponse is rejected at registration).
            return JobStatusResponse(status='done', result=fut.result())

        self.add_api_route(
            '/jobs/{job_id}',
            get_job_status,
            methods=['GET'],
            response_model=JobStatusResponse,
            name=_JOB_STATUS_ROUTE_NAME,
        )

    def _build_response_field(self, col_type: ts.ColumnType, *, comment: str | None = None) -> tuple[Any, FieldInfo]:
        """Build a (annotation, FieldInfo) pair for pydantic.create_model() from a ColumnType."""
        py_type = col_type.to_pydantic_type()
        field_kwargs: dict[str, Any] = {}
        if comment is not None:
            field_kwargs['description'] = comment
        if col_type.is_media_type():
            json_extra: dict[str, Any] = {'format': 'uri'}
            content_type = _MEDIA_CONTENT_TYPES.get(col_type._type)
            if content_type is not None:
                json_extra['contentMediaType'] = content_type
            field_kwargs['json_schema_extra'] = json_extra
        if col_type.nullable:
            return py_type, pydantic.Field(default=None, **field_kwargs)
        else:
            return py_type, pydantic.Field(**field_kwargs)

    def _convert_media_val(self, val: Any, url_for_media: Callable[[str], str]) -> Any:
        """
        If val is a local media file (a file:// uri or a bare absolute path) under an allowed media directory,
        converts it to a fetchable url of the /media endpoint. Otherwise returns val unchanged.

        Media values reach here in either form: a file:// uri (e.g. a column's fileurl) or a bare local path
        (e.g. a ResultSet's localpath, or a proxy-fetched file in the FileCache).
        """
        if not isinstance(val, str):
            return val
        if val.startswith('file:'):
            file_path = LocalStore.file_url_to_path(val)
        elif os.path.isabs(val):
            file_path = Path(val)
        else:
            return val  # a relative path or a remote (http/s3/...) url; leave for the client to fetch
        if file_path is None:
            return val
        resolved = file_path.resolve()
        if not self._is_allowed_media_path(resolved):
            return val
        # ensure forward slashes in the URL regardless of OS
        rel_path = resolved.relative_to(self._home_dir).as_posix()
        return url_for_media(rel_path)

    def _write_to_temp(self, upload: UploadFile) -> Path:
        """Write an UploadFile into TempStore"""
        ext = ''
        if upload.filename is not None:
            ext = os.path.splitext(upload.filename)[1]
        if len(ext) == 0 and upload.content_type is not None:
            ext = mimetypes.guess_extension(upload.content_type) or ''
        dest = TempStore.create_path(extension=ext)
        with open(dest, 'wb') as out:
            upload.file.seek(0)
            shutil.copyfileobj(upload.file, out)
        return dest
