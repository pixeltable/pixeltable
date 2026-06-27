"""Length-prefixed binary framing for proxy RPC bodies.

A framed body is a JSON head plus zero or more raw binary parts, laid out as:

    [u32 head_len][head bytes][u32 n_parts] ( [u32 part_len][part bytes] )*

All lengths are unsigned 32-bit big-endian.
"""

from __future__ import annotations

import struct

_U32 = struct.Struct('>I')


def encode_body(head: bytes, binary_parts: list[bytes]) -> bytes:
    out = [_U32.pack(len(head)), head, _U32.pack(len(binary_parts))]
    for part in binary_parts:
        out.append(_U32.pack(len(part)))
        out.append(part)
    return b''.join(out)


def decode_body(body: bytes) -> tuple[bytes, list[bytes]]:
    view = memoryview(body)
    offset = 0

    def take(n: int) -> bytes:
        nonlocal offset
        chunk = view[offset : offset + n]
        if len(chunk) != n:
            raise ValueError('truncated framed body')
        offset += n
        return bytes(chunk)

    def take_u32() -> int:
        return _U32.unpack(take(4))[0]

    head = take(take_u32())
    n_parts = take_u32()
    binary_parts = [take(take_u32()) for _ in range(n_parts)]
    return head, binary_parts
