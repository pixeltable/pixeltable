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


# A remote that cannot be synced, used mainly for testing.
class MockRemote(Remote):

    def __init__(self, push_cols: dict[str, ts.ColumnType], pull_cols: dict[str, ts.ColumnType]):
        self.push_cols = push_cols
        self.pull_cols = pull_cols

    def get_push_columns(self) -> dict[str, ts.ColumnType]:
        return self.push_cols

    def get_pull_columns(self) -> dict[str, ts.ColumnType]:
        return self.pull_cols

    def sync(self, t: Table, col_mapping: dict[str, str], push: bool, pull: bool) -> NotImplemented:
        raise NotImplementedError()

    def to_dict(self) -> dict[str, Any]:
        return {
            'push_cols': {k: v.as_dict() for k, v in self.push_cols.items()},
            'pull_cols': {k: v.as_dict() for k, v in self.pull_cols.items()}
        }

    @classmethod
    def from_dict(cls, md: dict[str, Any]) -> Remote:
        return cls(
            {k: ts.ColumnType.from_dict(v) for k, v in md['push_cols'].items()},
            {k: ts.ColumnType.from_dict(v) for k, v in md['pull_cols'].items()}
        )
