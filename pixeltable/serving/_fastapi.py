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
from typing import Annotated, Any, Callable, Literal, Optional, TypeVar

import fastapi
import PIL.Image
import pydantic
from fastapi import Body, File, Form, HTTPException, Query as QueryParam, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic.fields import FieldInfo

import pixeltable as pxt
import pixeltable.catalog as catalog
import pixeltable.exprs as exprs
import pixeltable.func as func
import pixeltable.type_system as ts
import pixeltable.utils.image as image_utils
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.utils.local_store import LocalStore, TempStore

_logger = logging.getLogger('pixeltable')


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


# Name used to register the `/media/{path:path}` route. Insert routes use this name with
# `request.url_for(_MEDIA_ROUTE_NAME, path=...)` to build absolute media URLs at request time.
_MEDIA_ROUTE_NAME = 'pxt_serve_media'
_JOB_STATUS_ROUTE_NAME = 'pxt_serve_job_status'


# ColumnType.Type -> JSON-Schema contentMediaType
# `.../*`: tell OpenAPI tooling to render the URL as a media link without committing to a specific subtype
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


class FastAPIRouter(fastapi.APIRouter):
    """
    A FastAPI `APIRouter` that exposes Pixeltable table operations as HTTP endpoints.
    """

    _executor: ThreadPoolExecutor
    _jobs: dict[str, Future]  # holds background requests; key: job id (uuid4().hex)
    _jobs_lock: threading.Lock
    _home_dir: Path
    _allowed_media_dirs: list[Path]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix='pxt-serve-background')
        self._jobs = {}
        self._jobs_lock = threading.Lock()
        self._home_dir = Config.get().home.resolve()
        self._allowed_media_dirs = [Env.get().media_dir.resolve(), Env.get().tmp_dir.resolve()]
        self._register_media_route()
        self._register_jobs_route()
        # Shut down the worker pool when the parent app's lifespan ends. include_router()
        # merges this handler into the app's on_shutdown list, so it fires on app shutdown.
        self.add_event_handler('shutdown', self._shutdown)

    def _shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _add_insert_route(
        self,
        t: pxt.Table,
        *,
        path: str,
        uploadfile_inputs: list[str],
        input_col_names: list[str],
        background: bool,
        endpoint_name: str,
        endpoint_model: type[pydantic.BaseModel] | None,
        response_class: type | None,
        row_processor: Callable[[dict[str, Any], Callable[[str], str]], Any],
    ) -> None:
        """Shared wiring for `add_insert_route()` / `insert_route()`.

        `build_response` is called with the single inserted row (as a dict) plus
        `url_for_media`, and returns the HTTP response body.
        """
        md = t.get_metadata()
        tbl_path, tbl_id, schema_version = md['path'], md['id'], md['schema_version']

        cols_by_name = {col.name: col for col in t._tbl_version_path.columns()}
        input_cols = [cols_by_name[name] for name in input_col_names]

        def run_insert(row_kwargs: dict[str, Any], url_for_media: Callable[[str], str]) -> Any:
            # handles aren't thread-portable, so fetch and re-validate against the registered schema
            tbl = pxt.get_table(tbl_path)
            if tbl._id != tbl_id or tbl._tbl_version.get().schema_version != schema_version:
                raise HTTPException(
                    status_code=409,
                    detail='table schema changed since route was registered; please restart the service',
                )

            status = tbl.insert([row_kwargs], return_rows=True)
            if status.rows is None or len(status.rows) != 1:
                n = 0 if status.rows is None else len(status.rows)
                raise HTTPException(status_code=500, detail=f'insert returned unexpected row count ({n})')
            return row_processor(status.rows[0], url_for_media)

        sig = self._create_endpoint_signature(input_cols=input_cols, upload_col_names=uploadfile_inputs)
        endpoint = self._create_endpoint(
            endpoint_name, sig, uploadfile_inputs=uploadfile_inputs, background=background, endpoint_op=run_insert
        )
        api_kwargs: dict[str, Any] = {'methods': ['POST']}
        if endpoint_model is not None:
            api_kwargs['response_model'] = endpoint_model
        if response_class is not None:
            api_kwargs['response_class'] = response_class
        self.add_api_route(path, endpoint, **api_kwargs)

    def add_insert_route(
        self,
        t: pxt.Table,
        *,
        path: str,
        inputs: list[str] | None = None,
        uploadfile_inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        return_fileresponse: bool = False,
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
            background: If True, return immediately with `{"id": ..., "job_url": ...}` and run
                the insert in a background thread. Poll `job_url` for the result. Mutually
                exclusive with `return_fileresponse`.

        Examples:
            JSON request/response:

            ```python
            router.add_insert_route(t, path='/generate', inputs=['prompt'], outputs=['result'])
            ```

            ```bash
            curl -X POST http://localhost:8000/generate \
              -H 'Content-Type: application/json' \
              -d '{"prompt": "a sunset over the ocean"}'
            # {"prompt": "a sunset over the ocean", "result": "..."}
            ```

            File upload with `FileResponse`:

            ```python
            router.add_insert_route(
                t, path='/resize', inputs=['width', 'height'],
                uploadfile_inputs=['image'], outputs=['resized'], return_fileresponse=True,
            )
            ```

            ```bash
            curl -X POST http://localhost:8000/resize \
              -F image=@photo.jpg -F width=640 -F height=480 \
              --output resized.jpg
            # saves the resized image to resized.jpg
            ```

            Background processing:

            ```python
            router.add_insert_route(t, path='/slow', background=True)
            ```

            ```bash
            # submit
            curl -X POST http://localhost:8000/slow -d '{"prompt": "hello"}'
            # {"id": "abc123", "job_url": "http://localhost:8000/jobs/abc123"}

            # poll
            curl http://localhost:8000/jobs/abc123
            # {"status": "done", "result": {...}}
            ```
        """
        input_col_names, output_col_names, cols_by_name = self._validate_insert_args(
            t,
            inputs=inputs,
            uploadfile_inputs=uploadfile_inputs,
            outputs=outputs,
            return_fileresponse=return_fileresponse,
            background=background,
            error_prefix='add_insert_route()',
        )
        uploadfile_inputs = uploadfile_inputs or []
        output_cols = [cols_by_name[name] for name in output_col_names]

        # response model derived from output columns, named after the path
        path_str = ''.join(el.capitalize() for el in path.split('/') if len(el) > 0)
        insert_response_model = self._create_model(f'{path_str}Response', output_cols=output_cols)

        endpoint_model: type[pydantic.BaseModel] | None
        if background:
            endpoint_model = BackgroundJobResponse
        elif return_fileresponse:
            endpoint_model = None
        else:
            endpoint_model = insert_response_model

        def row_processor(row: dict[str, Any], url_for_media: Callable[[str], str]) -> Any:
            output = self._create_output(
                [row], output_col_names, insert_response_model, return_fileresponse, url_for_media
            )
            return output[0] if isinstance(output, list) else output

        self._add_insert_route(
            t,
            path=path,
            uploadfile_inputs=uploadfile_inputs,
            input_col_names=input_col_names,
            background=background,
            endpoint_name=f'insert_{path.strip("/").replace("/", "_") or "root"}',
            endpoint_model=endpoint_model,
            response_class=FileResponse if return_fileresponse else None,
            row_processor=row_processor,
        )

    def _validate_insert_args(
        self,
        t: pxt.Table,
        *,
        inputs: list[str] | None,
        uploadfile_inputs: list[str] | None,
        outputs: list[str] | None,
        return_fileresponse: bool,
        background: bool,
        error_prefix: str,
    ) -> tuple[list[str], list[str], dict[str, catalog.Column]]:
        """Validate insert-route args and return (input_col_names, output_col_names, cols_by_name)."""
        md = t.get_metadata()
        if md['kind'] != 'table':
            raise pxt.RequestError(
                pxt.ErrorCode.UNSUPPORTED_OPERATION, f'{error_prefix}: cannot insert into {md["kind"]} {md["name"]!r}'
            )
        if return_fileresponse and background:
            raise pxt.RequestError(
                pxt.ErrorCode.INVALID_ARGUMENT,
                f'{error_prefix}: return_fileresponse and background are mutually exclusive',
            )

        col_md = md['columns']
        cols_by_name = {col.name: col for col in t._tbl_version_path.columns()}

        # computed columns cannot be input
        for name in [*(inputs or []), *(uploadfile_inputs or [])]:
            if name in col_md and col_md[name]['is_computed']:
                raise pxt.RequestError(
                    pxt.ErrorCode.INVALID_ARGUMENT,
                    f'{error_prefix}: {name!r} is a computed column and cannot be used as input',
                )

        input_col_names, output_col_names = self._validate_args(
            input_schema={c.name: c.col_type for c in cols_by_name.values() if not c.is_computed},
            output_schema={c.name: c.col_type for c in cols_by_name.values()},
            inputs=inputs,
            uploadfile_inputs=uploadfile_inputs,
            outputs=outputs,
            return_fileresponse=return_fileresponse,
            error_prefix=error_prefix,
            input_item_str='column',
            output_item_str='column',
        )
        return input_col_names, output_col_names, cols_by_name

    def insert_route(
        self,
        t: pxt.Table,
        *,
        path: str,
        inputs: list[str] | None = None,
        uploadfile_inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        background: bool = False,
    ) -> Callable[[Callable[..., pydantic.BaseModel]], Callable[..., pydantic.BaseModel]]:
        """
        Decorator that registers a POST endpoint performing a `Table.insert()` followed by user-defined post-processing.

        The request body carries the input column values (JSON, or multipart form data when `uploadfile_inputs` is
        used). After inserting the row, the decorated function is called with the requested output columns as
        keyword arguments (parameter names and Pixeltable types must match `outputs`). Its return value must be a
        Pydantic model and is returned as the HTTP response body.

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
            background: If True, return immediately with `{"id": ..., "job_url": ...}` and run
                the insert plus post-processing in a background thread. Poll `job_url` for the
                result; the decorated function's return value is delivered as the job result.

        Examples:
            ```python
            class GenerateResponse(pydantic.BaseModel):
                caption: str
                score: float

            @router.insert_route(
                t, path='/generate', inputs=['prompt'], outputs=['caption', 'score'], background=False
            )
            def format_response(*, caption: str, score: float) -> GenerateResponse:
                return GenerateResponse(caption=caption.strip(), score=round(score, 3))
            ```

            ```bash
            curl -X POST http://localhost:8000/generate \\
              -H 'Content-Type: application/json' \\
              -d '{"prompt": "a sunset over the ocean"}'
            # {"caption": "orange sky above calm water", "score": 0.932}
            ```
        """
        input_col_names, output_col_names, _ = self._validate_insert_args(
            t,
            inputs=inputs,
            uploadfile_inputs=uploadfile_inputs,
            outputs=outputs,
            return_fileresponse=False,
            background=background,
            error_prefix='insert_route()',
        )
        uploadfile_inputs = uploadfile_inputs or []

        def decorator(user_fn: Callable[..., pydantic.BaseModel]) -> Callable[..., pydantic.BaseModel]:
            self._validate_insert_route_fn(user_fn, output_col_names=output_col_names)
            response_model = user_fn.__annotations__['return']

            def row_processor(row: dict[str, Any], url_for_media: Callable[[str], str]) -> pydantic.BaseModel:
                kwargs = {name: self._convert_media_val(row[name], url_for_media) for name in output_col_names}
                return user_fn(**kwargs)

            self._add_insert_route(
                t,
                path=path,
                uploadfile_inputs=uploadfile_inputs,
                input_col_names=input_col_names,
                background=background,
                endpoint_name=f'insert_{path.strip("/").replace("/", "_") or "root"}',
                endpoint_model=BackgroundJobResponse if background else response_model,
                response_class=None,
                row_processor=row_processor,
            )
            return user_fn

        return decorator

    @staticmethod
    def _validate_insert_route_fn(user_fn: Callable, *, output_col_names: list[str]) -> None:
        """Validate the shape of the user's decorated function for `insert_route()`."""
        sig = inspect.signature(user_fn)
        fn_name = getattr(user_fn, '__name__', repr(user_fn))
        param_names: set[str] = set()
        for p in sig.parameters.values():
            if p.kind != inspect.Parameter.KEYWORD_ONLY:
                raise pxt.RequestError(
                    pxt.ErrorCode.UNSUPPORTED_OPERATION,
                    f'insert_route(): {fn_name!r} parameter {p.name!r} must be keyword-only '
                    '(place parameters after `*` in the signature)',
                )
            if p.name not in output_col_names:
                raise pxt.RequestError(
                    pxt.ErrorCode.UNSUPPORTED_OPERATION,
                    f'insert_route(): {fn_name!r} parameter {p.name!r} is not among the declared outputs '
                    f'{output_col_names}',
                )
            param_names.add(p.name)
        missing = [n for n in output_col_names if n not in param_names]
        if missing:
            raise pxt.RequestError(
                pxt.ErrorCode.MISSING_REQUIRED,
                f'insert_route(): {fn_name!r} is missing parameters for outputs {missing}; every declared '
                'output must appear as a keyword-only parameter',
            )
        return_annot = user_fn.__annotations__.get('return')
        if not (isinstance(return_annot, type) and issubclass(return_annot, pydantic.BaseModel)):
            raise pxt.RequestError(
                pxt.ErrorCode.INVALID_ARGUMENT,
                f'insert_route(): {fn_name!r} must have a return annotation that is a pydantic.BaseModel subclass; '
                f'got {return_annot!r}',
            )

    def add_update_route(
            self,
            t: pxt.Table,
            *,
            path: str,
            inputs: list[str] | None = None,
            uploadfile_inputs: list[str] | None = None,
            outputs: list[str] | None = None,
            return_fileresponse: bool = False,
            background: bool = False,
    ) -> None:
        """
        Add a POST endpoint that updates a single row in `t`, identified by its primary key, and returns the
        newly-updated row. The update is performed as a `Table.batch_update()` with the primary key columns and the
        columns referenced in `inputs`.

        The request body contains values for the primary key columns plus the input columns as JSON fields (or as
        [multipart form data](https://fastapi.tiangolo.com/tutorial/request-files/) when
        `uploadfile_inputs` is used). The response is a JSON object with the output column values,
        or a [`FileResponse`](https://fastapi.tiangolo.com/advanced/custom-response/#fileresponse)
        when `return_fileresponse=True`.

        Args:
            t: The table to insert into.
            path: The URL path for the endpoint.
            inputs: Column to be updated. Defaults to all non-computed, non-primary key columns.
            uploadfile_inputs: Columns to accept as
                [`UploadFile`](https://fastapi.tiangolo.com/tutorial/request-files/) fields
                (must be media-typed). These are sent as multipart form data; all other inputs
                become [`Form`](https://fastapi.tiangolo.com/tutorial/request-forms/) fields.
            outputs: Columns to include in the response. Defaults to all columns (including inputs).
            return_fileresponse: If True, return the single media-typed output column as a
                [`FileResponse`](https://fastapi.tiangolo.com/advanced/custom-response/#fileresponse).
                Requires exactly one media-typed output column.
            background: If True, return immediately with `{"id": ..., "job_url": ...}` and run
                the insert in a background thread. Poll `job_url` for the result. Mutually
                exclusive with `return_fileresponse`.

        Examples:
        """

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
            ```python
            router.add_delete_route(t, path='/delete')
            ```

            ```bash
            curl -X POST http://localhost:8000/delete -H 'Content-Type: application/json' \\
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

        tbl_path = md['path']
        tbl_id = md['id']
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
            tbl = pxt.get_table(tbl_path)
            if tbl._id != tbl_id or tbl._tbl_version.get().schema_version != schema_version:
                raise HTTPException(
                    status_code=409,
                    detail='table schema changed since route was registered; please restart the service',
                )

            where_expr: exprs.Expr | None = None
            for name in match_columns:
                predicate = tbl[name] == row_kwargs[name]
                where_expr = predicate if where_expr is None else (where_expr & predicate)
            status = tbl.delete(where=where_expr)
            return DeleteResponse(num_rows=status.num_rows)

        cols_by_name = {col.name: col for col in t._tbl_version_path.columns()}
        match_cols = [cols_by_name[name] for name in match_columns]
        sig = self._create_endpoint_signature(input_cols=match_cols)
        endpoint = self._create_endpoint(
            f'delete_{path.strip("/").replace("/", "_") or "root"}',
            sig,
            uploadfile_inputs=[],
            background=background,
            endpoint_op=run_delete,
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

            ```python
            router.add_query_route(path='/search', query=search_docs)
            ```

            ```bash
            curl -X POST http://localhost:8000/search \
              -H 'Content-Type: application/json' \
              -d '{"query_text": "hello"}'
            # {"rows": [{"id": 1, "text": "hello world", "score": 0.95}, ...]}
            ```

            Single-row lookup:

            ```python
            router.add_query_route(path='/lookup', query=lookup_by_id, one_row=True)
            ```

            ```bash
            curl -X POST http://localhost:8000/lookup -d '{"id": 42}'
            # {"id": 42, "name": "Alice", "email": "alice@example.com"}
            ```

            GET with query-string parameters:

            ```python
            router.add_query_route(path='/lookup', query=lookup_by_id, method='get')
            ```

            ```bash
            curl 'http://localhost:8000/lookup?id=42'
            # {"id": 42, "name": "Alice", "email": "alice@example.com"}
            ```

            `FileResponse`:

            ```python
            router.add_query_route(
                path='/thumbnail', query=get_thumbnail, return_fileresponse=True,
            )
            ```

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

        query_params = dict(query.signature.parameters)
        query_schema = {p.name: p.col_type for p in query_params.values()}
        result_schema = dict(query.template_query.schema)

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
                # Multi-row: wrap per-row models in a response with a `rows` field.
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

        template_query = query.template_query
        has_img_col_refs = any(
            item.col_type.is_image_type() and isinstance(item, exprs.ColumnRef)
            for item, _ in template_query._effective_select_list
        )
        if has_img_col_refs or return_fileresponse:
            # query rewrite:
            # - return_fileresponse: we want a local path when referencing media columns, even if the media file is
            #   stored externally
            # - otherwise: we want the fileurl property when referencing image columns (to avoid getting a PIL.Image)
            select_list: list[tuple[exprs.Expr, str]] = []
            for e, name in query.template_query._effective_select_list:
                if return_fileresponse and isinstance(e, exprs.ColumnRef) and e.col_type.is_media_type():
                    select_list.append((exprs.ColumnPropertyRef(e, exprs.ColumnPropertyRef.Property.LOCALPATH), name))
                elif isinstance(e, exprs.ColumnRef) and e.col_type.is_image_type():
                    select_list.append((exprs.ColumnPropertyRef(e, exprs.ColumnPropertyRef.Property.FILEURL), name))
                else:
                    select_list.append((e, name))
            template_query = pxt.Query(
                select_list=select_list,
                from_clause=template_query._from_clause,
                where_clause=template_query.where_clause,
                group_by_clause=template_query.group_by_clause,
                grouping_tbl=template_query.grouping_tbl,
                order_by_clause=template_query.order_by_clause,
                limit=query.template_query.limit_val,
                offset=query.template_query.offset_val,
                sample_clause=query.template_query.sample_clause,
            )

        def run_query(call_kwargs: dict[str, Any], url_for_media: Callable[[str], str]) -> Any:
            bound_df = template_query.bind(call_kwargs)
            result_set = bound_df.collect()
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
        endpoint = self._create_endpoint(
            f'query_{path.strip("/").replace("/", "_") or "root"}',
            sig,
            uploadfile_inputs=uploadfile_inputs,
            background=background,
            endpoint_op=run_query,
        )

        api_kwargs: dict[str, Any] = {'methods': [method.upper()]}
        if endpoint_model is not None:
            api_kwargs['response_model'] = endpoint_model
        if return_fileresponse:
            api_kwargs['response_class'] = FileResponse
        self.add_api_route(path, endpoint, **api_kwargs)

    def _create_endpoint(
        self,
        name: str,
        signature: inspect.Signature,
        uploadfile_inputs: list[str],
        background: bool,
        endpoint_op: Callable,
    ) -> Callable[..., Any]:
        def endpoint(request: Request, **kwargs: Any) -> Any:
            sample_url = str(request.url_for(_MEDIA_ROUTE_NAME, path='_'))
            media_url_base = sample_url[:-1]

            def url_for_media(rel_path: str) -> str:
                return f'{media_url_base}{urllib.parse.quote(rel_path, safe="/")}'

            # write out uploads while the request is still alive
            tmp_paths: list[Path] = []
            if len(uploadfile_inputs) > 0:
                # list(...): make sure that the sequence of name/val pairs can't change underneath us
                for input_name, val in list(kwargs.items()):
                    if input_name in uploadfile_inputs:
                        path = self._write_to_temp(val)
                        tmp_paths.append(path)
                        kwargs[input_name] = str(path)

            if background:
                job_id = uuid.uuid4().hex
                fut = self._executor.submit(_run_endpoint_op, endpoint_op, kwargs, tmp_paths, url_for_media)
                with self._jobs_lock:
                    self._jobs[job_id] = fut
                return BackgroundJobResponse(
                    id=job_id, job_url=str(request.url_for(_JOB_STATUS_ROUTE_NAME, job_id=job_id))
                )
            else:
                return _run_endpoint_op(endpoint_op, kwargs, tmp_paths, url_for_media)

        # FastAPI needs the correct signature and a name
        endpoint.__signature__ = signature  # type: ignore[attr-defined]
        endpoint.__name__ = name
        return endpoint

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
        output_cols: list[catalog.Column] | None = None,
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
        input_cols: list[catalog.Column] | None = None,
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
        rows: list[dict[str, Any]],
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
        # flush PIL images to temp files, store the local file:// uri
        for row in rows:
            for col_name, val in row.items():
                if isinstance(val, PIL.Image.Image):
                    fmt = image_utils.default_format(val)
                    ext = f'.{fmt}'
                    dest = TempStore.create_path(extension=ext)
                    val.save(dest, format=fmt)
                    row[col_name] = dest.as_uri()

        if return_fileresponse:
            assert len(output_names) == 1 and len(rows) == 1
            output_name = output_names[0]
            val = rows[0][output_name]
            if val is None:
                raise HTTPException(status_code=500, detail=f'output column {output_name!r} is null')
            local_path: Path
            if val.startswith('file:'):
                local_path = LocalStore.file_url_to_path(val) or Path(val)
            else:
                local_path = Path(val)
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
        """Register a `GET /media/{path:path}` route that serves Pixeltable media and tmp files"""

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
        If val is a file:// uri under the Pixeltable media or tmp directory, converts that to a fetchable url of
        the /media endpoint. Otherwise returns val unchanged.
        """
        if not isinstance(val, str) or not val.startswith('file:'):
            return val
        file_path = LocalStore.file_url_to_path(val)
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
