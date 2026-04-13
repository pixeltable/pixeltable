from __future__ import annotations

import inspect
import mimetypes
import os
import shutil
import threading
import urllib.parse
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Optional

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
from pixeltable.utils.local_store import TempStore


class BackgroundJobResponse(pydantic.BaseModel):
    id: str
    job_url: str


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


class PxtFastAPIRouter(fastapi.APIRouter):
    """A FastAPI `APIRouter` that exposes Pixeltable table operations as HTTP endpoints."""

    _executor: ThreadPoolExecutor
    _jobs: dict[str, Future]
    _jobs_lock: threading.Lock

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix='pxt-serve-background')
        self._jobs = {}
        self._jobs_lock = threading.Lock()
        self._register_media_route()
        self._register_jobs_route()
        # Shut down the worker pool when the parent app's lifespan ends. include_router()
        # merges this handler into the app's on_shutdown list, so it fires on app shutdown.
        self.add_event_handler('shutdown', self._shutdown)

    def _shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

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
        Create endpoint for a `Table.insert()` operation.

        Adds an endpoint to the router that inserts a row into `t`. The endpoint expects a POST request with a
        JSON payload that contains the values for the columns specified in `inputs`. It returns a JSON object
        that contains the values of the columns specified in `outputs`.

        Args:
            t: The table to insert into.
            path: The path to the endpoint.
            inputs: The columns to use as request fields. If not specified, all non-computed columns are used.
            outputs: The columns to return in the response. If not specified, all columns are returned (including those
                provided in the request).
            uploadfile_inputs: The input columns to use as request fields of type `UploadFile`.
                These can only reference media columns (`Image`, `Video`, `Audio`, `Document`).
            return_fileresponse: If True, the endpoint returns a `FileResponse` for the single (required) media-typed
                output column. If False (default), it returns a JSON object with one field per output column; local
                media values (`file://` URIs) are translated into servable URLs.
            background: if True, runs the endpoint logic in a background thread and immediately returns a JSON
                object with fields `id` and `job_url`.

        Examples
        - call for table with image and text input and video output
        - include how to send request via curl
        """
        if not isinstance(t, pxt.InsertableTable):
            raise pxt.Error(f'add_insert_route(): cannot insert into {t._display_name()} {t._name}')
        if return_fileresponse and background:
            raise pxt.Error('add_insert_route(): return_fileresponse and background are mutually exclusive')

        tbl_path = t._path()
        tbl_id = t._id
        schema_version = t._tbl_version.get().schema_version

        cols_by_name = {col.name: col for col in t._tbl_version_path.columns()}
        cols_by_id = {col.id: col for col in cols_by_name.values()}

        # extra validation: computed columns cannot be input
        if inputs is not None or uploadfile_inputs is not None:
            for name in [*(inputs or []), *(uploadfile_inputs or [])]:
                if name in cols_by_name and cols_by_name[name].is_computed:
                    raise pxt.Error(f'add_insert_route(): {name!r} is a computed column and cannot be used as input')

        input_col_names, output_col_names = self._validate_args(
            input_schema={c.name: c.col_type for c in cols_by_name.values() if not c.is_computed},
            output_schema={c.name: c.col_type for c in cols_by_name.values()},
            inputs=inputs,
            uploadfile_inputs=uploadfile_inputs,
            outputs=outputs,
            return_fileresponse=return_fileresponse,
            error_prefix='add_insert_route()',
            input_item_str='column',
            output_item_str='column',
        )

        input_col_ids = [cols_by_name[name].id for name in input_col_names]
        input_cols = [cols_by_id[i] for i in input_col_ids]
        upload_col_ids = [cols_by_name[name].id for name in uploadfile_inputs] if uploadfile_inputs is not None else []
        output_col_ids = [cols_by_name[name].id for name in output_col_names]

        insert_response_model, endpoint_model = self._create_route_models(
            path=path,
            output_cols=[cols_by_id[col_id] for col_id in output_col_ids],
            return_fileresponse=return_fileresponse,
            background=background,
        )

        def run_insert(
            tbl: pxt.Table, row_kwargs: dict[str, Any], tmp_paths: list[Path], url_for_media: Callable[[str], str]
        ) -> Any:
            try:
                status = tbl.insert([row_kwargs], return_rows=True)
            except BaseException:
                # insert() did not consume the staged files; remove any leftovers from TempStore
                for p in tmp_paths:
                    if p.exists():
                        TempStore.delete_media_file(p)
                raise
            assert status.rows is not None and len(status.rows) == 1, 'expected exactly one inserted row'
            output = self._create_output(
                status.rows, output_col_names, insert_response_model, return_fileresponse, url_for_media
            )
            if isinstance(output, list):
                return output[0]
            else:
                return output

        def endpoint(request: Request, **kwargs: Any) -> Any:
            # get a fresh table handle and validate it against the one we saw when the route was registered
            tbl = pxt.get_table(tbl_path)
            assert tbl._id == tbl_id, f'{tbl._id} != {tbl_id}'
            assert tbl._tbl_version.get().schema_version == schema_version

            # Resolve the media URL prefix now, in the request thread. We can't safely call
            # request.url_for() from the background worker: the ASGI scope's lifetime is the
            # request/response cycle, and touching it after the response has been sent relies on
            # undocumented behavior. Instead, ask Starlette to build a URL with '_' as the path
            # param ('_' is URL-unreserved so it passes through percent-encoding unchanged),
            # strip it, and close `url_for_media` over the resulting plain string.
            sample_url = str(request.url_for(_MEDIA_ROUTE_NAME, path='_'))
            media_url_base = sample_url[:-1]

            def url_for_media(rel_path: str) -> str:
                return f'{media_url_base}{urllib.parse.quote(rel_path, safe="/")}'

            # write out uploads while the request is still alive
            tmp_paths: list[Path] = []
            if len(upload_col_ids) > 0:
                for col in input_cols:
                    if col.id in upload_col_ids:
                        path = self._write_to_temp(kwargs[col.name])
                        tmp_paths.append(path)
                        kwargs[col.name] = str(path)

            if background:
                job_id = uuid.uuid4().hex
                fut = self._executor.submit(run_insert, tbl, kwargs, tmp_paths, url_for_media)
                with self._jobs_lock:
                    self._jobs[job_id] = fut
                return BackgroundJobResponse(
                    id=job_id, job_url=str(request.url_for(_JOB_STATUS_ROUTE_NAME, job_id=job_id))
                )
            else:
                try:
                    return run_insert(tbl, kwargs, tmp_paths, url_for_media)
                except pxt.Error as e:
                    raise HTTPException(status_code=400, detail=str(e)) from e

        sig = self._create_endpoint_signature(
            input_cols=input_cols, upload_col_names=[cols_by_id[id].name for id in upload_col_ids]
        )
        endpoint.__signature__ = sig  # type: ignore[attr-defined]
        endpoint.__name__ = f'insert_{path.strip("/").replace("/", "_") or "root"}'

        api_kwargs: dict[str, Any] = {'methods': ['POST']}
        if endpoint_model is not None:
            api_kwargs['response_model'] = endpoint_model
        if return_fileresponse:
            api_kwargs['response_class'] = FileResponse
        self.add_api_route(path, endpoint, **api_kwargs)

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
        Create endpoint for a parameterized query.

        Adds an endpoint to the router that invokes a query udf. The endpoint expects a POST request with a
        JSON body that contains the values for the parameters of the query udf. It returns a JSON array
        whose elements are objects, one per result row, containing the select-list items of the query.

        Args:
            path: The path to the endpoint.
            query: The query to execute, created with the `@pxt.query` decorator or `pxt.retrieval_udf()`.
            inputs: The parameters to use as request fields. If not specified, all parameters are used.
                In v1, query parameters with default values are not supported and must be excluded.
            uploadfile_inputs: The parameters to use as request fields of type `UploadFile`.
                These parameters must be media-typed (`Image`, `Video`, `Audio`, `Document`).
            one_row: If True, the query is expected to return exactly one row and the endpoint sends that single row as
                the JSON object response. Otherwise the endpoint sends a JSON object with a field `rows` which contains
                the query result as a list of dicts.
                If True and the query returns no rows, the endpoint responds with a 404 error.
            return_fileresponse: If True, the endpoint returns a `FileResponse` for the single (required) media-typed
                result column. The query must return exactly one row at request time: 0 rows produces a 404,
                more than 1 row produces a 500. If False (default), it returns a JSON array with one element per
                result row; local media values (`file://` URIs) are translated into servable URLs.
            background: If True, runs the endpoint logic in a background thread and immediately returns a JSON
                object with fields `id` and `job_url`.
            method: The HTTP method to use for the endpoint.
        """
        if not isinstance(query, func.QueryTemplateFunction):
            raise pxt.Error(
                f'add_query_route(): `query` must be a @pxt.query or retrieval_udf, not {type(query).__name__}'
            )
        if return_fileresponse and background:
            raise pxt.Error('add_query_route(): return_fileresponse and background are mutually exclusive')
        assert query.template_query is not None
        uploadfile_inputs = uploadfile_inputs or []

        query_params = dict(query.signature.parameters)
        query_schema = {p.name: p.col_type for p in query_params.values()}
        result_schema = dict(query.template_query.schema)

        _, _ = self._validate_args(
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

        return_scalar = query.return_scalar
        scalar_col_name = next(iter(result_schema)) if return_scalar else None

        # build pydantic models for the response
        output_model: type[pydantic.BaseModel] | None  # what's returned by run_query(); None for scalars
        query_result_model: type[pydantic.BaseModel] | None = None
        if return_scalar:
            output_model = None
        else:
            query_result_model, _ = self._create_route_models(
                path=path, output_schema=result_schema, return_fileresponse=return_fileresponse, background=background
            )
            if one_row:
                output_model = query_result_model
            else:
                # Multi-row: wrap per-row models in a response with a `rows` field.
                path_elements = path.split('/')
                path_str = ''.join([el.capitalize() for el in path_elements if len(el) > 0])
                output_model = pydantic.create_model(
                    f'{path_str}RowResponse',
                    rows=(list[query_result_model], pydantic.Field(description='Query result rows')),  # type: ignore[valid-type]
                )

        # endpoint_model: what FastAPI advertises as the response type
        endpoint_model: Any
        if background:
            endpoint_model = BackgroundJobResponse
        elif return_fileresponse:
            endpoint_model = None
        elif return_scalar:
            scalar_py_type = next(iter(result_schema.values())).to_python_type()
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

        def run_query(call_kwargs: dict[str, Any], tmp_paths: list[Path], url_for_media: Callable[[str], str]) -> Any:
            try:
                bound_df = template_query.bind(call_kwargs)
                result_set = bound_df.collect()
            except BaseException:
                for p in tmp_paths:
                    if p.exists():
                        TempStore.delete_media_file(p)
                raise
            rows = list(result_set)

            # do error checking now, before converting data
            if return_fileresponse or one_row:
                if len(rows) == 0:
                    raise HTTPException(status_code=404, detail='query returned no rows')
                if len(rows) > 1:
                    raise HTTPException(status_code=500, detail=f'query returned {len(rows)} rows; expected exactly 1')

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

        def endpoint(request: Request, **kwargs: Any) -> Any:
            sample_url = str(request.url_for(_MEDIA_ROUTE_NAME, path='_'))
            media_url_base = sample_url[:-1]

            def url_for_media(rel_path: str) -> str:
                return f'{media_url_base}{urllib.parse.quote(rel_path, safe="/")}'

            # write out uploads while the request is still alive
            tmp_paths: list[Path] = []
            if len(uploadfile_inputs) > 0:
                for name in kwargs:  # noqa: PLC0206
                    if name in uploadfile_inputs:
                        path = self._write_to_temp(kwargs[name])
                        tmp_paths.append(path)
                        kwargs[name] = str(path)

            if background:
                job_id = uuid.uuid4().hex
                fut = self._executor.submit(run_query, kwargs, tmp_paths, url_for_media)
                with self._jobs_lock:
                    self._jobs[job_id] = fut
                return BackgroundJobResponse(
                    id=job_id, job_url=str(request.url_for(_JOB_STATUS_ROUTE_NAME, job_id=job_id))
                )
            else:
                try:
                    return run_query(kwargs, tmp_paths, url_for_media)
                except pxt.Error as e:
                    raise HTTPException(status_code=400, detail=str(e)) from e

        sig = self._create_endpoint_signature(
            input_schema=query_schema, upload_col_names=uploadfile_inputs, is_post=(method == 'post')
        )
        endpoint.__signature__ = sig  # type: ignore[attr-defined]
        endpoint.__name__ = f'query_{path.strip("/").replace("/", "_") or "root"}'

        api_kwargs: dict[str, Any] = {'methods': [method.upper()]}
        if endpoint_model is not None:
            api_kwargs['response_model'] = endpoint_model
        if return_fileresponse:
            api_kwargs['response_class'] = FileResponse
        self.add_api_route(path, endpoint, **api_kwargs)

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
                    raise pxt.Error(f'{error_prefix}: unknown input {input_item_str} {name!r}')
                input_cols.append(name)

        # validate uploadfile_inputs
        if uploadfile_inputs is not None:
            for name in uploadfile_inputs:
                if name not in input_schema:
                    raise pxt.Error(f'{error_prefix}: unknown uploadfile input {input_item_str} {name!r}')
                col_type = input_schema[name]
                if not col_type.is_media_type():
                    raise pxt.Error(
                        f'{error_prefix}: uploadfile input {input_item_str} {name!r} is not a media {input_item_str}'
                    )
                if inputs is not None and name in inputs:
                    raise pxt.Error(f'{error_prefix}: {name!r} appears in both `inputs` and `uploadfile_inputs`')
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
                    raise pxt.Error(f'{error_prefix}: unknown output {output_item_str} {name!r}')
                output_cols.append(name)

        if return_fileresponse and (len(output_cols) != 1 or not output_schema[output_cols[0]].is_media_type()):
            raise pxt.Error(
                f'{error_prefix}: return_fileresponse=True requires exactly one media-typed output {output_item_str}'
            )

        return input_cols, output_cols

    def _create_route_models(
        self,
        path: str,
        output_schema: dict[str, ts.ColumnType] | None = None,
        output_cols: list[catalog.Column] | None = None,
        return_fileresponse: bool = False,
        background: bool = False,
    ) -> tuple[type[pydantic.BaseModel] | None, type[pydantic.BaseModel] | None]:
        """Returns: (output model, endpoint model)"""
        assert (output_schema is None) != (output_cols is None)

        output_model: type[pydantic.BaseModel] | None
        if return_fileresponse:
            output_model = None
        else:
            fields: dict[str, tuple[Any, FieldInfo]]
            if output_cols is not None:
                fields = {
                    col.name: self._build_response_field(col.col_type, comment=col.comment) for col in output_cols
                }
            else:
                fields = {name: self._build_response_field(col_type) for name, col_type in output_schema.items()}
            # we name the response model after the path
            path_elements = path.split('/')
            path_str = ''.join([el.capitalize() for el in path_elements if len(el) > 0])
            model_name = f'{path_str}Response'
            output_model = pydantic.create_model(model_name, **fields)  # type: ignore[call-overload]

        endpoint_model: type[pydantic.BaseModel] | None
        if background:
            endpoint_model = BackgroundJobResponse
        elif return_fileresponse:
            endpoint_model = None
        else:
            endpoint_model = output_model

        return output_model, endpoint_model

    def _create_endpoint_signature(
        self,
        input_cols: list[catalog.Column] | None = None,
        input_schema: dict[str, ts.ColumnType] | None = None,
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

        input_cols_by_name = {c.name: c for c in input_cols} if input_cols is not None else {}
        input_col_names = list(input_schema.keys()) if input_cols is None else list(input_cols_by_name.keys())
        input_col_types = list(input_schema.values()) if input_cols is None else [c.col_type for c in input_cols]
        for name, col_type in zip(input_col_names, input_col_types):
            annotation: Any
            # propagate the column comment (if any) as the OpenAPI field description
            desc = input_cols_by_name[name].comment if name in input_cols_by_name else None
            if name in upload_col_names:
                annotation = Annotated[UploadFile, File(..., description=desc)]
            elif not is_post:
                # GET: parameters come from the query string
                py_type = col_type.to_python_type()
                annotation = Annotated[py_type, QueryParam(..., description=desc)]
            else:
                py_type = col_type.to_python_type()
                if has_uploads:
                    annotation = Annotated[py_type, Form(..., description=desc)]
                else:
                    annotation = Annotated[py_type, Body(..., embed=True, description=desc)]
            params.append(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, annotation=annotation))

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
            assert len(output_names) == 1
            assert len(rows) == 1
            output_name = output_names[0]
            val = rows[0][output_name]
            if val is None:
                raise HTTPException(status_code=500, detail=f'output column {output_name!r} is null')
            assert isinstance(val, str)
            local_path: Path
            if val.startswith('file://'):
                local_path = Path(urllib.parse.unquote(val[len('file://') :]))
            else:
                local_path = Path(val)
            assert local_path.exists() and local_path.is_file()
            media_type, _ = mimetypes.guess_type(local_path)
            return FileResponse(local_path, media_type=media_type or 'application/octet-stream')

        converted = [{name: self._convert_media_val(row[name], url_for_media) for name in output_names} for row in rows]
        if output_model is not None:
            return [output_model(**row) for row in converted]
        else:
            return converted

    def _register_media_route(self) -> None:
        """Register a `GET /media/{path:path}` route that serves Pixeltable-owned files"""
        root_dir = os.path.realpath(str(Config.get().home))

        def serve_media(path: str) -> FileResponse:
            abs_path = os.path.realpath(os.path.join(root_dir, path))
            if not abs_path.startswith(root_dir + os.sep) and abs_path != root_dir:
                raise HTTPException(status_code=404, detail='not found')
            if not os.path.isfile(abs_path):
                raise HTTPException(status_code=404, detail='not found')
            media_type, _ = mimetypes.guess_type(abs_path)
            return FileResponse(abs_path, media_type=media_type or 'application/octet-stream')

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
        py_type = col_type.to_python_type()
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
            return Optional[py_type], pydantic.Field(default=None, **field_kwargs)
        else:
            return py_type, pydantic.Field(**field_kwargs)

    def _convert_media_val(self, val: Any, url_for_media: Callable[[str], str]) -> Any:
        """
        If val is a file:// uri under the Pixeltable root directory, converts that to a fetchable url of
        the /media endpoint. Otherwise returns val unchanged.
        """
        if not isinstance(val, str) or not val.startswith('file://'):
            return val
        local_path = urllib.parse.unquote(val[len('file://') :])
        root_dir = Config.get().home
        rel_path = os.path.relpath(local_path, root_dir)
        if rel_path.startswith('..'):
            return val
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
