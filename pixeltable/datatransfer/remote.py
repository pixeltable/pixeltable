from __future__ import annotations

import abc
from typing import Any

import pixeltable.type_system as ts
from pixeltable import Table


class Remote:

    @abc.abstractmethod
    def get_push_columns(self) -> dict[str, ts.ColumnType]: ...

    @abc.abstractmethod
    def get_pull_columns(self) -> dict[str, ts.ColumnType]: ...

    @abc.abstractmethod
    def sync(self, t: Table, col_mapping: dict[str, str], push: bool, pull: bool) -> None: ...

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, md: dict[str, Any]) -> Remote: ...
