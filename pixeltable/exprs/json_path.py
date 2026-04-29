from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import jmespath
import sqlalchemy as sql

from pixeltable import catalog, exceptions as excs, type_system as ts

from .column_ref import ColumnRef
from .data_row import DataRow
from .expr import Expr
from .globals import print_slice
from .json_mapper import JsonMapperDispatch
from .object_ref import ObjectRef
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache


class JsonPath(Expr):
    """
    anchor can be None, in which case this is a relative JsonPath and the anchor is set later via set_anchor().
    scope_idx: for relative paths, index of referenced JsonMapper
    (0: indicates the immediately preceding JsonMapper, -1: the parent of the immediately preceding mapper, ...)
    """

    path_elements: list[str | int | slice]
    compiled_path: jmespath.parser.ParsedResult | None
    scope_idx: int
    file_handles: dict[Path, io.BufferedReader]  # key: file path

    def __init__(
        self, anchor: Expr | None, path_elements: list[str | int | slice] | None = None, scope_idx: int = 0
    ) -> None:
        if path_elements is None:
            path_elements = []

        super().__init__(
            self.__resolve_type(anchor.col_type, path_elements) if anchor is not None else ts.JsonType(nullable=True)
        )
        self.path_elements = path_elements
        self.compiled_path = jmespath.compile(self._json_path()) if len(path_elements) > 0 else None
        if anchor is not None:
            self.components = [anchor]
        self.scope_idx = scope_idx
        # NOTE: the _create_id() result will change if set_anchor() gets called;
        # this is not a problem, because _create_id() shouldn't be called after init()
        self.id = self._create_id()
        self.file_handles = {}

    @classmethod
    def __errstr(cls, el: str | int | slice) -> str:
        if isinstance(el, str):
            return repr(el)
        elif isinstance(el, int):
            return f'[{el}]'
        elif isinstance(el, slice):
            start_str = '' if el.start is None else str(el.start)
            stop_str = '' if el.stop is None else str(el.stop)
            step_str = '' if el.step is None else f':{el.step}'
            return f'[{start_str}:{stop_str}{step_str}]'
        else:
            raise AssertionError(f'Invalid JsonPath element: {el}')

    @classmethod
    def __resolve_type(cls, col_type: ts.ColumnType, path_elements: list[str | int | slice]) -> ts.ColumnType:
        if len(path_elements) == 0:
            # JsonPath expressions always have `nullable=True`, regardless of the schema. This is because
            # schema validation is optional in some runtime contexts, so it's possible to encounter data
            # at runtime that doesn't match the schema.
            return col_type.copy(nullable=True)

        el = path_elements[0]

        if not isinstance(col_type, ts.JsonType):
            # There are more path elements, but we've arrived at something other than JsonType;
            # fall back on general JsonType.
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'Invalid JsonPath: cannot resolve {cls.__errstr(el)} on primitive type `{col_type}`',
            )

        schema = col_type.type_schema
        if schema is None:
            # There are more path elements, but the JsonType has no schema; fall back on general JsonType.
            return ts.JsonType(nullable=True)

        # We're still a JsonType, and we have a schema, and there are more path elements to resolve.
        # Try various ways of resolving the next path element based on the schema.

        if el == '*' and isinstance(schema.type_spec, list):
            # '*' is a no-op for type resolution purposes; it simply confirms that the type schema is a list and
            # selects all elements.
            return cls.__resolve_type(col_type, path_elements[1:])

        if isinstance(el, str) and isinstance(schema.type_spec, dict) and el in schema.type_spec:
            # Dict key resolution.
            return cls.__resolve_type(schema.type_spec[el], path_elements[1:])

        if isinstance(el, str) and isinstance(schema.type_spec, list):
            # Key resolution on a list: acts transitively on all elements of the list. This corresponds to
            # expressions like `col.f1[0:3].f2` where f1 is a list of dicts, extracting `f2` from each dict in the
            # list.
            type_spec = [cls.__resolve_type(t, path_elements) for t in schema.type_spec]
            variadic_type = (
                cls.__resolve_type(schema.variadic_type, path_elements) if schema.variadic_type is not None else None
            )
            return ts.JsonType(ts.JsonType.TypeSchema(type_spec, variadic_type=variadic_type), nullable=True)

        if isinstance(el, int) and isinstance(schema.type_spec, list):
            if el >= 0:
                # Positive index on tuple
                if el < len(schema.type_spec):
                    return cls.__resolve_type(schema.type_spec[el], path_elements[1:])
                elif schema.variadic_type is not None:
                    return cls.__resolve_type(schema.variadic_type, path_elements[1:])
            elif schema.variadic_type is None:
                # Negative index on fixed-length tuple
                if -el <= len(schema.type_spec):
                    return cls.__resolve_type(schema.type_spec[el], path_elements[1:])
            else:
                # Negative index on variadic tuple: we don't know which element it will reference, so we need to
                # find the supertype of all possible types it could reference
                relevant_types = [*schema.type_spec[el:], schema.variadic_type]
                supertype = ts.ColumnType.common_supertype(relevant_types)
                if supertype is not None:
                    return cls.__resolve_type(supertype, path_elements[1:])

        if isinstance(el, slice) and isinstance(schema.type_spec, list):
            if schema.variadic_type is None:
                # Slice on fixed-length tuple
                new_type = ts.JsonType(ts.JsonType.TypeSchema(type_spec=schema.type_spec[el]), nullable=True)
                return cls.__resolve_type(new_type, path_elements[1:])
            elif (el.start is None or el.start >= 0) and (el.stop is None or el.stop >= 0):
                # Slice with positive indices on variadic tuple.
                type_spec = schema.type_spec[el]
                variadic_type = (
                    None if el.stop is not None and el.stop <= len(schema.type_spec) else schema.variadic_type
                )
                new_type = ts.JsonType(
                    ts.JsonType.TypeSchema(type_spec=type_spec, variadic_type=variadic_type), nullable=True
                )
                return cls.__resolve_type(new_type, path_elements[1:])
            else:
                # Slice with negative indices on variadic tuple: just make this a variadic tuple of the supertype.
                # We could try to do something more clever in certain cases, but it's such an edge case that it's
                # not clear the added complexity is worth it. This simple logic will handle the vast majority of
                # common cases correctly (including all lists / pure varidic tuples).
                supertype = ts.ColumnType.common_supertype([*schema.type_spec, schema.variadic_type])
                if supertype is not None:
                    new_type = ts.JsonType(ts.JsonType.TypeSchema([], variadic_type=supertype), nullable=True)
                    return cls.__resolve_type(new_type, path_elements[1:])

        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION,
            f'Invalid JsonPath: cannot resolve {cls.__errstr(el)}, '
            f'because it does not match the expected type schema:\n{col_type}',
        )

    def release(self) -> None:
        for fh in self.file_handles.values():
            fh.close()
        self.file_handles.clear()

    def __repr__(self) -> str:
        # else 'R': the anchor is RELATIVE_PATH_ROOT
        anchor_str = str(self.anchor) if self.anchor is not None else 'R'
        if len(self.path_elements) == 0:
            return anchor_str
        return f'{anchor_str}{"." if isinstance(self.path_elements[0], str) else ""}{self._json_path()}'

    def _as_dict(self) -> dict:
        assert len(self.components) <= 1
        components_dict: dict[str, Any]
        if len(self.components) == 0 or isinstance(self.components[0], ObjectRef):
            # If the anchor is an ObjectRef, it means this JsonPath is a bound relative path. We store it as a relative
            # path, *not* a bound path (which has no meaning in the dict).
            components_dict = {}
        else:
            components_dict = super()._as_dict()
        path_elements = [[el.start, el.stop, el.step] if isinstance(el, slice) else el for el in self.path_elements]
        return {'path_elements': path_elements, 'scope_idx': self.scope_idx, **components_dict}

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> JsonPath:
        assert 'path_elements' in d
        assert 'scope_idx' in d
        assert len(components) <= 1
        anchor = components[0] if len(components) == 1 else None
        path_elements = [slice(el[0], el[1], el[2]) if isinstance(el, list) else el for el in d['path_elements']]
        return cls(anchor, path_elements, d['scope_idx'])

    @property
    def anchor(self) -> Expr | None:
        return None if len(self.components) == 0 else self.components[0]

    def set_anchor(self, anchor: Expr) -> None:
        assert len(self.components) == 0
        self.components = [anchor]

    def is_relative_path(self) -> bool:
        return self.anchor is None

    def _has_relative_path(self) -> bool:
        return self.is_relative_path() or super()._has_relative_path()

    def _bind_rel_paths(self, mapper: 'JsonMapperDispatch' | None = None) -> None:
        if self.is_relative_path():
            # TODO: take scope_idx into account
            self.set_anchor(mapper.scope_anchor)
        else:
            self.anchor._bind_rel_paths(mapper)

    def __call__(self, *args: object, **kwargs: object) -> 'JsonPath':
        """
        Construct a relative path that references an ancestor of the immediately enclosing JsonMapper.
        """
        if not self.is_relative_path():
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, '() for an absolute path is invalid')
        if len(args) != 1 or not isinstance(args[0], int) or args[0] >= 0:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'R() requires a negative index')
        return JsonPath(None, [], args[0])

    def __getattr__(self, name: str) -> 'JsonPath':
        assert isinstance(name, str)
        return JsonPath(self.anchor, [*self.path_elements, name])

    def __getitem__(self, index: object) -> 'JsonPath':
        if isinstance(index, (int, slice, str)):
            return JsonPath(self.anchor, [*self.path_elements, index])
        raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, f'Invalid json list index: {index}')

    def default_column_name(self) -> str | None:
        anchor_name = self.anchor.default_column_name() if self.anchor is not None else ''
        ret_name = f'{anchor_name}.{self._json_path()}'

        def cleanup_char(s: str) -> str:
            if s == '.':
                return '_'
            elif s == '*':
                return 'star'
            elif s.isalnum():
                return s
            else:
                return ''

        clean_name = ''.join(map(cleanup_char, ret_name))
        clean_name = clean_name.lstrip('_')  # remove leading underscore
        if not clean_name:  # Replace '' with None
            clean_name = None

        assert clean_name is None or catalog.is_valid_identifier(clean_name)
        return clean_name

    def _equals(self, other: JsonPath) -> bool:
        return self.path_elements == other.path_elements

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('path_elements', self.path_elements)]

    def sql_expr(self, _: SqlElementCache) -> sql.ColumnElement | None:
        """
        Postgres appears to have a bug: jsonb_path_query('{a: [{b: 0}, {b: 1}]}', '$.a.b') returns
        *two* rows (each containing col val 0), not a single row with [0, 0].
        We need to use a workaround: retrieve the entire dict, then use jmespath to extract the path correctly.
        """
        # path_str = '$.' + '.'.join(self.path_elements)
        # assert isinstance(self._anchor(), ColumnRef)
        # return sql.func.jsonb_path_query(self._anchor().col.sa_col, path_str)
        return None

    def _json_path(self) -> str:
        assert len(self.path_elements) > 0
        result: list[str] = []
        for element in self.path_elements:
            if element == '*':
                result.append('[*]')
            elif isinstance(element, str):
                result.append(f'{"." if len(result) > 0 else ""}{element}')
            elif isinstance(element, int):
                result.append(f'[{element}]')
            elif isinstance(element, slice):
                result.append(f'[{print_slice(element)}]')
        return ''.join(result)

    def eval(self, row: DataRow, row_builder: RowBuilder) -> None:
        assert self.anchor is not None, self
        val = row[self.anchor.slot_idx]
        if self.compiled_path is not None:
            val = self.compiled_path.search(val)
        row[self.slot_idx] = val
        if val is None or self.anchor is None or not isinstance(self.anchor, ColumnRef):
            return

        # the origin of val is a json-typed column, which might stored inlined objects
        if self.anchor.slot_idx not in row.slot_md:
            # we can infer that there aren't any inlined objects because our execution plan doesn't include
            # materializing the cellmd (eg, insert plans)
            # TODO: have the planner pass that fact into ExprEvalNode explicitly to streamline this path a bit more
            return

        # defer import until it's needed
        from pixeltable.exec.cell_reconstruction_node import json_has_inlined_objs, reconstruct_json

        cell_md = row.slot_md[self.anchor.slot_idx]
        if cell_md is None or cell_md.file_urls is None or not json_has_inlined_objs(val):
            # val doesn't contain inlined objects
            return

        row.vals[self.slot_idx] = reconstruct_json(val, cell_md.file_urls, self.file_handles)


RELATIVE_PATH_ROOT = JsonPath(None)
