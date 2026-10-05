"""`pxt db update` output behavior that needs no hosted database.

The diff answer is faked at `post_request`, so these exercise only what the client prints.
"""

import json
import os
import sys
from typing import BinaryIO, TextIO

import pytest

import pixeltable_cli.client.commands.db as db_cmd
from pixeltable_cli import types
from pixeltable_cli.client import utils

# the refusal code `confirm_or_exit` applies on a non-tty run without -f
EXIT_REFUSED = 3


def _pending_plan() -> dict:
    plan = types.DbPlan.from_ops(
        'pxt://acme:main',
        types.DbState.AVAILABLE,
        [
            types.DbChangeOp(
                target='archive',
                name='project',
                op='alter',
                severity='additive',
                description='the project files will be uploaded: app.py changed',
            )
        ],
    )
    return plan.model_dump(mode='json')


def _open_joined_pipe(monkeypatch: pytest.MonkeyPatch) -> tuple[BinaryIO, TextIO, TextIO]:
    """Stdout and stderr writing one pipe, buffered the way a non-tty pipe is.

    Stdout is block-buffered. Stderr is line-buffered, so a refusal line reaches the pipe
    while a plan print is still sitting in the stdout buffer.
    """
    read_fd, write_fd = os.pipe()
    out_fp = os.fdopen(write_fd, 'w', buffering=8192)
    err_fp = os.fdopen(os.dup(out_fp.fileno()), 'w', buffering=1)
    monkeypatch.setattr(sys, 'stdout', out_fp)
    monkeypatch.setattr(sys, 'stderr', err_fp)
    return os.fdopen(read_fd, 'rb'), out_fp, err_fp


def _read_joined(read_fp: BinaryIO, out_fp: TextIO, err_fp: TextIO) -> str:
    """Flush the buffers shutdown would flush, then read the one pipe."""
    err_fp.flush()
    out_fp.flush()
    err_fp.close()
    out_fp.close()
    merged = read_fp.read().decode()
    read_fp.close()
    return merged


class TestUpdateRefusal:
    def test_refusal_prints_plan_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A non-tty pipe shows the pending plan once, and that plan precedes the refusal line."""
        monkeypatch.setattr(db_cmd, 'post_request', lambda _path, _body: _pending_plan())
        monkeypatch.setattr(utils, 'stdin_is_a_tty', lambda: False)
        read_fp, out_fp, err_fp = _open_joined_pipe(monkeypatch)

        try:
            with pytest.raises(SystemExit, match=f'^{EXIT_REFUSED}$'):
                db_cmd.run(['update', 'pxt://acme:main'])
        finally:
            merged = _read_joined(read_fp, out_fp, err_fp)

        assert merged.count('Plan:') == 1, merged
        assert merged.index('Plan:') < merged.index('refusing to proceed'), merged

    def test_refusal_json_prints_pending_plan(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--json emits no pending plan up front, so the refusal still prints it as the one document."""
        monkeypatch.setattr(db_cmd, 'post_request', lambda _path, _body: _pending_plan())
        monkeypatch.setattr(utils, 'stdin_is_a_tty', lambda: False)

        with pytest.raises(SystemExit, match=f'^{EXIT_REFUSED}$'):
            db_cmd.run(['update', 'pxt://acme:main', '--json'])

        out, _ = capsys.readouterr()
        doc = json.loads(out)
        assert doc['resolution'] == 'update_additive', out
        assert doc['in_agreement'] is False
