from __future__ import annotations

from .table_proxy import TableProxy


class InsertableTableProxy(TableProxy):
    """A proxy for a hosted InsertableTable handle."""

    def _display_name(self) -> str:
        return 'table'
