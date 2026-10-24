"""`pxt.URI` and `pxt.ConfigVar` (configuration references)."""

from __future__ import annotations

import re
from typing import Any, ClassVar, Generic, Self, TypeVar

from pixeltable import exceptions as excs

ConfVarT = TypeVar('ConfVarT', bound=str)

_CONFIG_VAR_NAME_RE = r'[a-z_][a-z0-9_]*'


class URI(str):
    """A storage destination: an object-store URI, or a local filesystem path."""

    def __new__(cls, value: str) -> Self:
        if value.strip() == '':
            raise ValueError('a destination cannot be empty')
        return super().__new__(cls, value)


class ConfigVar(Generic[ConfVarT]):
    """A reference to a database variable, defined at module scope."""

    TAG = '$confvar'
    _CONFVAR_TYPES: ClassVar[dict[str, type[str]]] = {'str': str, 'URI': URI}

    name: str
    type_: type[ConfVarT]

    def __init__(self, name: str, type_: type[ConfVarT]) -> None:
        if re.fullmatch(_CONFIG_VAR_NAME_RE, name) is None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                f'Invalid config var name {name!r}: lowercase letters, digits and underscores only, '
                'not starting with a digit.',
            )
        if type_ not in self._CONFVAR_TYPES.values():
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                f'Invalid config var type {type_.__name__!r}: must be one of {", ".join(self._CONFVAR_TYPES)}.',
            )
        self.name = name
        self.type_ = type_

    def value(self) -> ConfVarT:
        raise NotImplementedError('ConfigVar.value() is not implemented yet in pixeltable-rs')

    def _as_dict(self) -> dict[str, Any]:
        return {self.TAG: self.name, 'type': self.type_.__name__}

    def __repr__(self) -> str:
        return f'ConfigVar({self.name!r}, {self.type_.__name__})'

    def __str__(self) -> str:
        return f'${self.name}'
