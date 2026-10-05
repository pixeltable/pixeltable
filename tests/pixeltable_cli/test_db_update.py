"""`pxt db update` output behavior that needs no hosted database.

The diff answer is faked at `post_request`, so these exercise only what the client prints.
"""

import json

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


class TestUpdateRefusal:
    def test_refusal_prints_plan_once(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A non-tty run refuses; the pending plan printed before the prompt is not printed again."""
        monkeypatch.setattr(db_cmd, 'post_request', lambda _path, _body: _pending_plan())
        monkeypatch.setattr(utils, 'stdin_is_a_tty', lambda: False)

        with pytest.raises(SystemExit, match=f'^{EXIT_REFUSED}$'):
            db_cmd.run(['update', 'pxt://acme:main'])

        out, err = capsys.readouterr()
        assert out.count('Plan:') == 1, out
        assert 'refusing to proceed' in err

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
