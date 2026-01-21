"""
Pixeltable UDFs for `UUID`.
"""

import os
import threading
import time
import uuid

import sqlalchemy as sql

import pixeltable as pxt
from pixeltable.utils.code import local_public_names

# UUID7 implementation based on CPython's uuid.py (PSF License)
# https://github.com/python/cpython/blob/main/Lib/uuid.py
# State variables for monotonicity within a millisecond
_uuid7_lock = threading.Lock()
_last_timestamp_v7: int | None = None
_last_counter_v7: int = 0
_RFC_4122_VERSION_7_FLAGS = 0x7000_8000_0000_0000_0000


def _uuid7_get_counter_and_tail() -> tuple[int, int]:
    """Generate a random 42-bit counter (with MSB=0) and 32-bit tail."""
    rnd = int.from_bytes(os.urandom(10), 'big')
    counter = (rnd >> 32) & 0x1FF_FFFF_FFFF  # 41 bits (MSB is 0)
    tail = rnd & 0xFFFF_FFFF  # 32 bits
    return counter, tail


def _uuid7() -> uuid.UUID:
    """Generate a UUID from a Unix timestamp in milliseconds and random bits.

    UUIDv7 objects feature monotonicity within a millisecond.
    """
    global _last_timestamp_v7  # noqa: PLW0603
    global _last_counter_v7  # noqa: PLW0603

    with _uuid7_lock:
        nanoseconds = time.time_ns()
        timestamp_ms = nanoseconds // 1_000_000

        if _last_timestamp_v7 is None or timestamp_ms > _last_timestamp_v7:
            counter, tail = _uuid7_get_counter_and_tail()
        else:
            if timestamp_ms < _last_timestamp_v7:
                timestamp_ms = _last_timestamp_v7 + 1
            # advance the 42-bit counter
            counter = _last_counter_v7 + 1
            if counter > 0x3FF_FFFF_FFFF:
                # advance the 48-bit timestamp
                timestamp_ms += 1
                counter, tail = _uuid7_get_counter_and_tail()
            else:
                # 32-bit random data
                tail = int.from_bytes(os.urandom(4), 'big')

        unix_ts_ms = timestamp_ms & 0xFFFF_FFFF_FFFF
        counter_msbs = counter >> 30
        # keep 12 counter's MSBs and clear variant bits
        counter_hi = counter_msbs & 0x0FFF
        # keep 30 counter's LSBs and clear version bits
        counter_lo = counter & 0x3FFF_FFFF
        # ensure that the tail is always a 32-bit integer
        tail &= 0xFFFF_FFFF

        int_uuid_7 = unix_ts_ms << 80
        int_uuid_7 |= counter_hi << 64
        int_uuid_7 |= counter_lo << 32
        int_uuid_7 |= tail
        # by construction, the variant and version bits are already cleared
        int_uuid_7 |= _RFC_4122_VERSION_7_FLAGS

        # defer global update until all computations are done
        _last_timestamp_v7 = timestamp_ms
        _last_counter_v7 = counter

    return uuid.UUID(int=int_uuid_7)


@pxt.udf
def uuid4() -> uuid.UUID:
    """
    Generate a random UUID (version 4).

    Equivalent to [`uuid.uuid4()`](https://docs.python.org/3/library/uuid.html#uuid.uuid4).
    """
    return uuid.uuid4()


@uuid4.to_sql
def _() -> sql.ColumnElement:
    return sql.func.gen_random_uuid()  # Generates uuid version 4


@pxt.udf
def uuid7() -> uuid.UUID:
    """
    Generate a time-based UUID.

    Equivalent to [`uuid.uuid7()`](https://docs.python.org/3/library/uuid.html#uuid.uuid7).
    """
    if hasattr(uuid, 'uuid7'):
        return uuid.uuid7()
    return _uuid7()


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
