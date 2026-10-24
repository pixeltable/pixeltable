"""Fixtures for pixeltable-rs's own tests (run from pixeltable-rs/: `.venv/bin/python -m pytest tests_rs`)."""

import os
import tempfile
from typing import Iterator

import pytest

_home = tempfile.mkdtemp(prefix='pxtrs-test-')
os.environ['PIXELTABLE_HOME'] = _home
os.environ.setdefault('PIXELTABLE_TIME_ZONE', 'America/Los_Angeles')

import pixeltable as pxt  # noqa: E402
import pixeltable._testing  # noqa: E402

assert 'pixeltable-rs' in pxt.__file__, f'wrong pixeltable package: {pxt.__file__}'


@pytest.fixture(autouse=True)
def reset_db() -> Iterator[None]:
    pixeltable._testing.reset_db()
    yield
