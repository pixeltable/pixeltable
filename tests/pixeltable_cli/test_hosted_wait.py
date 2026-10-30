"""A hosted wait shows its progress without a terminal and stops on SIGTERM as on Ctrl-C (PXT-1614)."""

import signal
import sys
import threading
from typing import Any, Callable

import pytest

from pixeltable_cli.client import hosted
from pixeltable_cli.types import GenerationReceipt, ReceiptOutcome, ResourcePhase

_DB_URI = 'pxt://acme:main'
_STATUS = f'pxt db status {_DB_URI}'


def _receipt(phase: ResourcePhase | None = None, waiting_for: str | None = None) -> GenerationReceipt:
    return GenerationReceipt(
        kind='database', resource_id='db-uuid', generation=7, db='main', phase=phase, waiting_for=waiting_for
    )


def _serve(monkeypatch: pytest.MonkeyPatch, answers: list[Callable[[], GenerationReceipt]]) -> None:
    """Answers the n-th poll with answers[n]()."""
    polls = iter(answers)

    def post_request(path: str, body: dict[str, Any], *, raise_on_error: bool = False) -> Any:
        assert path == '/api/receipts'
        return {'receipts': [next(polls)().model_dump(mode='json')]}

    monkeypatch.setattr(hosted, 'post_request', post_request)
    monkeypatch.setattr(hosted, 'RECEIPT_POLL_INTERVAL', 0.0)


class TestProgressWithoutATerminal:
    def test_a_line_per_change(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        """capsys is not a terminal, as a coding agent's or CI's stderr is not."""
        building = _receipt(ResourcePhase.BLOCKED, 'the image build')
        done = _receipt().model_copy(update={'outcome': ReceiptOutcome.OBSERVED})
        _serve(monkeypatch, [lambda: building, lambda: building, lambda: done])

        settled = hosted.await_receipts(_DB_URI, [_receipt()], status_command=_STATUS)

        assert settled[0].observed
        out, err = capsys.readouterr()
        # nothing on stdout, which --json keeps for its document
        assert out == ''
        assert [line.rsplit('  ', 1)[0] for line in err.splitlines()] == [
            'main generation 7  PENDING',
            'main generation 7  BLOCKED, waiting for the image build',
        ]
        assert all(line.endswith('0:00:00') for line in err.splitlines())

    def test_a_line_each_interval_while_a_poll_blocks(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A poll that takes longer than the interval does not silence the wait: its caller can tell it is alive."""
        monkeypatch.setattr(hosted, '_PLAIN_PROGRESS_INTERVAL', 0.05)
        building = _receipt(ResourcePhase.BLOCKED, 'the image build')
        done = _receipt().model_copy(update={'outcome': ReceiptOutcome.OBSERVED})
        printed = threading.Semaphore(0)

        def counted_print(*args: Any, **kwargs: Any) -> None:
            print(*args, **kwargs)
            printed.release()

        monkeypatch.setattr(hosted, 'print', counted_print, raising=False)

        def slow() -> GenerationReceipt:
            # blocks for the first label and 3 repeats, not a set time: a loaded runner can starve the heartbeat thread
            assert all(printed.acquire(timeout=10) for _ in range(4)), 'the wait went silent during a blocking poll'
            return building

        _serve(monkeypatch, [slow, lambda: done])

        hosted.await_receipts(_DB_URI, [_receipt()], status_command=_STATUS)

        lines = capsys.readouterr().err.splitlines()
        # the first label, and its repeats from the heartbeat during the slow poll
        assert sum(1 for line in lines if 'PENDING' in line) >= 4


@pytest.mark.skipif(sys.platform == 'win32', reason='a tool timeout on Windows ends the process without a signal')
def test_sigterm_stops_the_wait_with_a_next_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A coding agent's tool timeout sends SIGTERM; the wait says the change continues, and how to follow it."""
    previous = signal.getsignal(signal.SIGTERM)

    def terminated() -> GenerationReceipt:
        signal.raise_signal(signal.SIGTERM)
        raise AssertionError('SIGTERM did not stop the wait')

    _serve(monkeypatch, [terminated])
    with pytest.raises(SystemExit) as exc:
        hosted.await_receipts(_DB_URI, [_receipt()], status_command=_STATUS)

    assert exc.value.code == hosted.EXIT_TERMINATED == 143
    err = capsys.readouterr().err
    assert 'pxt: stopped waiting; the change continues in Pixeltable Cloud' in err
    assert err.endswith(f'next: {_STATUS}  # shows its progress\n')
    # the wait gives SIGTERM back to its previous handler
    assert signal.getsignal(signal.SIGTERM) is previous
