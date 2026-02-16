import datetime
import uuid
from zoneinfo import ZoneInfo

import numpy as np

import pixeltable as pxt

_DEFAULT_DATE = datetime.date(2026, 2, 10)
_DEFAULT_TIMESTAMP = datetime.datetime(2026, 2, 10, 21, 15, 0, tzinfo=ZoneInfo('UTC'))
_DEFAULT_BINARY = b'\xde\xad\xbe\xef'
_DEFAULT_ARRAY = np.ones(10, dtype=np.float64)
_DEFAULT_UUID = uuid.UUID('deadbeef-cafe-beef-beef-deadfacedead')


@pxt.udf()
def test_date_udf(date1: pxt.Date, date2: pxt.Date, date3: pxt.Date = _DEFAULT_DATE) -> int:
    return date1.toordinal() + date2.toordinal() + date3.toordinal()


@pxt.udf()
def test_timestamp_udf(ts1: pxt.Timestamp, ts2: pxt.Timestamp, ts3: pxt.Timestamp = _DEFAULT_TIMESTAMP) -> float:
    return ts1.timestamp() + ts2.timestamp() + ts3.timestamp()


@pxt.udf()
def test_binary_udf(b1: bytes, b2: bytes, b3: bytes = _DEFAULT_BINARY) -> bytes:
    return bytes(a ^ b ^ c for a, b, c in zip(b1, b2, b3))


@pxt.udf()
def test_array_udf(
    a: pxt.Array[np.float64, (10,)], b: pxt.Array[np.float64, (10,)], c: pxt.Array[np.float64, (10,)] = _DEFAULT_ARRAY
) -> pxt.Array[np.float64, (10,)]:
    return a + b + c


@pxt.udf()
def test_uuid_udf(uuid1: pxt.UUID, uuid2: pxt.UUID, uuid3: pxt.UUID = _DEFAULT_UUID) -> str:
    return str(uuid1) + str(uuid2) + str(uuid3)
