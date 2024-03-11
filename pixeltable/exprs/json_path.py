from __future__ import annotations
from typing import Optional, List, Any, Dict, Tuple, Union

import jmespath
import sqlalchemy as sql

from .globals import print_slice
from .expr import Expr
from .json_mapper import JsonMapper
from .data_row import DataRow
from .row_builder import RowBuilder
import pixeltable
import pixeltable.exceptions as excs
import pixeltable.catalog as catalog
import pixeltable.type_system as ts


class JsonPath(Expr):
    def __init__(self, anchor: Optional['pixeltable.exprs.ColumnRef'], path_elements: Optional[List[str]] = None, scope_idx: int = 0):
        """
        anchor can be None, in which case this is a relative JsonPath and the anchor is set later via set_anchor().
        scope_idx: for relative paths, index of referenced JsonMapper
        (0: indicates the immediately preceding JsonMapper, -1: the parent of the immediately preceding mapper, ...)
        """
        if path_elements is None:
            path_elements = []
        super().__init__(ts.JsonType())
        if anchor is not None:
            self.components = [anchor]
        self.path_elements: List[Union[str, int]] = path_elements
        self.compiled_path = jmespath.compile(self._json_path()) if len(path_elements) > 0 else None
        self.scope_idx = scope_idx
        # NOTE: the _create_id() result will change if set_anchor() gets called;
        # this is not a problem, because _create_id() shouldn't be called after init()
        self.id = self._create_id()

    def __str__(self) -> str:
        # else "R": the anchor is RELATIVE_PATH_ROOT
        return (f'{str(self._anchor) if self._anchor is not None else "R"}'
            f'{"." if isinstance(self.path_elements[0], str) else ""}{self._json_path()}')

    def _as_dict(self) -> Dict:
        return {'path_elements': self.path_elements, 'scope_idx': self.scope_idx, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr]) -> Expr:
        assert 'path_elements' in d
        assert 'scope_idx' in d
        assert len(components) <= 1
        anchor = components[0] if len(components) == 1 else None
        return cls(anchor, d['path_elements'], d['scope_idx'])

    @property
    def _anchor(self) -> Optional[Expr]:
        return None if len(self.components) == 0 else self.components[0]

    def set_anchor(self, anchor: Expr) -> None:
        assert len(self.components) == 0
        self.components = [anchor]

    def is_relative_path(self) -> bool:
        return self._anchor is None

    def bind_rel_paths(self, mapper: Optional['JsonMapper'] = None) -> None:
        if not self.is_relative_path():
            return
        # TODO: take scope_idx into account
        self.set_anchor(mapper.scope_anchor)

    def __call__(self, *args: object, **kwargs: object) -> 'JsonPath':
        """
        Construct a relative path that references an ancestor of the immediately enclosing JsonMapper.
        """
        if not self.is_relative_path():
            raise excs.Error(f'() for an absolute path is invalid')
        if len(args) != 1 or not isinstance(args[0], int) or args[0] >= 0:
            raise excs.Error(f'R() requires a negative index')
        return JsonPath(None, [], args[0])

    def __getattr__(self, name: str) -> 'JsonPath':
        assert isinstance(name, str)
        return JsonPath(self._anchor, self.path_elements + [name])

    def __getitem__(self, index: object) -> 'JsonPath':
        if isinstance(index, str):
            if index != '*':
                raise excs.Error(f'Invalid json list index: {index}')
        else:
            if not isinstance(index, slice) and not isinstance(index, int):
                raise excs.Error(f'Invalid json list index: {index}')
        return JsonPath(self._anchor, self.path_elements + [index])

    def __rshift__(self, other: object) -> 'JsonMapper':
        rhs_expr = Expr.from_object(other)
        if rhs_expr is None:
            raise excs.Error(f'>> requires an expression on the right-hand side, found {type(other)}')
        return JsonMapper(self, rhs_expr)

    def default_column_name(self) -> Optional[str]:
        anchor_name = self._anchor.default_column_name() if self._anchor is not None else ''
        ret_name = f'{anchor_name}.{self._json_path()}'
        
        def cleanup_char(s : str) -> str:
            if s == '.':
                return '_'
            elif s == '*':
                return 'star'
            elif s.isalnum():
                return s
            else:
                return ''
            
        clean_name = ''.join(map(cleanup_char, ret_name))
        clean_name = clean_name.lstrip('_') # remove leading underscore
        if clean_name == '':
            clean_name = None
        
        assert clean_name is None or catalog.is_valid_identifier(clean_name)
        return clean_name

    def _equals(self, other: JsonPath) -> bool:
        return self.path_elements == other.path_elements

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('path_elements', self.path_elements)]

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        """
        Postgres appears to have a bug: jsonb_path_query('{a: [{b: 0}, {b: 1}]}', '$.a.b') returns
        *two* rows (each containing col val 0), not a single row with [0, 0].
        We need to use a workaround: retrieve the entire dict, then use jmespath to extract the path correctly.
        """
        #path_str = '$.' + '.'.join(self.path_elements)
        #assert isinstance(self._anchor(), ColumnRef)
        #return sql.func.jsonb_path_query(self._anchor().col.sa_col, path_str)
        return None

    def _json_path(self) -> str:
        assert len(self.path_elements) > 0
        result: List[str] = []
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

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        val = data_row[self._anchor.slot_idx]
        if self.compiled_path is not None:
            val = self.compiled_path.search(val)
        data_row[self.slot_idx] = val


RELATIVE_PATH_ROOT = JsonPath(None)
