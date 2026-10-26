from __future__ import annotations

import html

from pixeltable.types import FunctionInfo


class FunctionList(list[FunctionInfo]):
    """A list of functions, as returned by [`pxt.list_functions()`][pixeltable.list_functions].

    Printed, it shows one function per line; in a notebook, it renders as a table.
    """

    def __repr__(self) -> str:
        return '\n'.join(self._line(info) for info in self)

    def _line(self, info: FunctionInfo) -> str:
        line = f'{info["kind"]:<9} {info["call"]}'
        if len(info['signatures']) > 1:
            line += f'  [{len(info["signatures"])} signatures]'
        if info['summary'] is not None:
            line += f'  # {info["summary"]}'
        return line

    def _repr_html_(self) -> str:
        rows = ''.join(
            f'<tr><td>{info["kind"]}</td><td><code>{html.escape(info["call"])}</code></td>'
            f'<td>{"" if info["summary"] is None else html.escape(info["summary"])}</td></tr>'
            for info in self
        )
        return f'<table><tr><th>kind</th><th>call</th><th>summary</th></tr>{rows}</table>'
