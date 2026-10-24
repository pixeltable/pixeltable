"""Rendering of table descriptions (`repr(table)`, `table.describe()`, `_repr_html_()`); the content comes from the
backend (`_core.tbl_describe()`)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


def _summarize_json(metadata: Any, max_elements: int = 5, max_chars: int = 100, indent: int = 4, interpose: int = 10) -> str:
    pad = ' ' * indent
    if isinstance(metadata, dict):
        parts = []
        for key, value in list(metadata.items())[:max_elements]:
            value_str = json.dumps(value)
            if len(value_str) > max_chars:
                value_str = value_str[: max_chars - interpose] + ' [...] ' + value_str[-interpose:]
            parts.append(f'{pad}"{key}": {value_str}')
        result = '{\n' + ',\n'.join(parts)
        if len(metadata) > max_elements:
            result += f',\n{pad}[...] ({len(metadata) - max_elements} more)'
        return result + '\n}'
    value_str = json.dumps(metadata)
    if len(value_str) > max_chars:
        value_str = value_str[: max_chars - interpose] + ' [...] ' + value_str[-interpose:]
    return value_str


def _blocks(d: dict[str, Any]) -> list[tuple[Any, list[int] | None]]:
    blocks: list[tuple[Any, list[int] | None]] = [(d['title'], None)]
    cols = pd.DataFrame(d['columns'], columns=['Column Name', 'Type', 'Source', 'Computed With', 'Comment'])
    blocks.append((cols, d['separators']))
    if len(d['indexes']) > 0:
        blocks.append((pd.DataFrame(d['indexes']), None))
    if d['comment']:
        blocks.append((f'Comment: {d["comment"]}', None))
    if d['custom_metadata']:
        blocks.append((f'Custom Metadata: {_summarize_json(d["custom_metadata"])}', None))
    return blocks


def _render_df(df: pd.DataFrame, separators: list[int] | None) -> str:
    df = df.copy()
    df.index = [''] * len(df)  # type: ignore[assignment]
    lines = df.to_string(max_colwidth=50).split('\n')
    width = max(len(line) for line in lines)
    seps = set(separators or [])
    result = [lines[0], '-' * width]
    for i, line in enumerate(lines[1:]):
        result.append(line)
        if i in seps:
            result.append('.' * width)
    return '\n'.join(result)


def to_string(d: dict[str, Any]) -> str:
    return '\n\n'.join(b if isinstance(b, str) else _render_df(b, seps) for b, seps in _blocks(d))


def to_html(d: dict[str, Any]) -> str:
    html = []
    for b, _ in _blocks(d):
        if isinstance(b, str):
            html.append(pd.DataFrame([b]).style.hide(axis='index').hide(axis='columns').to_html())
        else:
            html.append(b.style.hide(axis='index').to_html())
    return '\n'.join(html)
